from .._pycatan import get_trade_table
import torch
import torch.nn as nn
import torch.nn.functional as F

class XDimBlock(nn.Module):
    def __init__(self, ch_2d, dim_scalar, hidden_ch=None, hidden_dim=None):
        super().__init__()
        
        self.h_ch = hidden_ch if hidden_ch is not None else ch_2d
        self.h_dim = hidden_dim if hidden_dim is not None else dim_scalar

        # --- 2D Main Path ---
        self.conv_main = nn.Conv2d(ch_2d, self.h_ch, kernel_size=(3, 5), padding=(1, 2), bias=False)
        self.bn_2d = nn.BatchNorm2d(self.h_ch)

        # --- Scalar Main Path ---
        self.fc_main = nn.Linear(dim_scalar, self.h_dim, bias=False)
        self.bn_scalar = nn.BatchNorm1d(self.h_dim)

        # --- Cross Connections (Interactions) ---
        
        # 1. Scalar -> 2D (Inflation)
        # スカラを加工して2Dチャンネル数に合わせる層
        self.fc_s2i = nn.Linear(dim_scalar, self.h_ch, bias=False) 
        
        # 2. 2D -> Scalar (Deflation)
        # 2Dの特徴(平均+分散で2倍の数になる)を加工してスカラ次元に合わせる層
        # 入力次元は ch_2d * 2 (平均と分散)
        self.fc_i2s = nn.Linear(ch_2d * 2, self.h_dim, bias=False)

    def forward(self, x_2d, x_scalar):
        """
        x_2d: (Batch, C, H, W)
        x_scalar: (Batch, N)
        """
        B, C, H, W = x_2d.shape

        # --- 1. Compute Main Features ---
        # 2D Stream
        out_2d = self.conv_main(x_2d) # (B, h_ch, H, W)
        out_2d = self.bn_2d(out_2d)
        
        # Scalar Stream
        out_scalar = self.fc_main(x_scalar) # (B, h_dim)
        out_scalar = self.bn_scalar(out_scalar)

        # --- 2. Compute Cross Features ---
        
        # [Scalar -> 2D] Inflation
        # スカラを線形変換 -> (B, h_ch) -> (B, h_ch, 1, 1) -> (B, h_ch, H, W) に拡張
        s_inflated = self.fc_s2i(x_scalar)
        s_inflated = s_inflated.view(B, self.h_ch, 1, 1).expand(-1, -1, H, W)
        
        # [2D -> Scalar] Deflation
        # 論文: "each channel is reduced to two scalars: its average and variance" 
        # (B, C, H, W) -> flatten spatial -> (B, C, H*W)
        flat_2d = x_2d.view(B, C, -1).float()
        mean_2d = flat_2d.mean(dim=2) # (B, C)
        var_2d = flat_2d.var(dim=2, unbiased=False) # (B, C)
        deflated = torch.cat([mean_2d, var_2d], dim=1) # (B, C*2)
        
        i_deflated = self.fc_i2s(deflated) # (B, h_dim)
        
        # 2D Output: Main(2D) + Inflated(Scalar) + Input(Residual)
        next_2d = F.relu(out_2d + s_inflated + x_2d)
        
        # Scalar Output: Main(Scalar) + Deflated(2D) + Input(Residual)
        next_scalar = F.relu(out_scalar + i_deflated + x_scalar)

        return next_2d, next_scalar

class CatanXDimModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_channels = config.get("hidden_channels", 32) 
        self.scalar_hidden_dim = config.get("scalar_hidden_dim", 64)
        num_blocks = config.get("num_xdim_blocks", 4)

        board_channels = config["input_board_shape"][2]
        flat_dim = config["input_scalar_dim"]
        self.spatial_out_channels = config["num_players"] + 3  # MT + 建設
        num_spatial = len(config["board_logits_mask"][0]) # 空間に対応するアクションの数
        num_global_action = config["action_dim"] - num_spatial
        num_opponents = config["num_players"] - 1
        full_table = get_trade_table(player_count=config["num_players"])
        recipe_embeddings = torch.tensor(
            full_table[0::num_opponents, :5],
            dtype=torch.float32
        )
        self.register_buffer("recipe_embeddings", F.normalize(recipe_embeddings, dim=1))
        self.trade_temp = config.get("trade_temp", 10.0)

        # Initial layers
        self.conv_in = nn.Conv2d(board_channels, self.hidden_channels, kernel_size=(3, 5), padding=(1, 2), bias=False)
        self.bn_in_2d = nn.BatchNorm2d(self.hidden_channels)

        self.fc_in = nn.Linear(flat_dim, self.scalar_hidden_dim, bias=False)
        self.bn_in_scalar = nn.BatchNorm1d(self.scalar_hidden_dim)

        # X-dim Blocks
        self.blocks = nn.ModuleList([
            XDimBlock(
                ch_2d=self.hidden_channels, 
                dim_scalar=self.scalar_hidden_dim
            ) 
            for _ in range(num_blocks) # 指定された回数だけ積む
        ])

        self.spatial_head = nn.Conv2d(self.hidden_channels, self.spatial_out_channels, kernel_size=1) # 盤面に対応したlogits
        self.global_head = nn.Sequential(
            nn.Linear(self.scalar_hidden_dim, self.scalar_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.scalar_hidden_dim, num_global_action)
        ) # 盤面に依存しないlogits

        self.trade_query_head = nn.Sequential(
            nn.Linear(self.scalar_hidden_dim, self.scalar_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.scalar_hidden_dim, 5) # 資源種類数
        )

        self.value_head = nn.Sequential(
            nn.Linear(self.scalar_hidden_dim, self.scalar_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.scalar_hidden_dim, 1) ,
        )

        # board_logitsから取り出す用 maskはTranspose済み
        self.register_buffer("idx_c", config["board_logits_mask"][0])
        self.register_buffer("idx_y", config["board_logits_mask"][1])
        self.register_buffer("idx_x", config["board_logits_mask"][2])



    def forward(self, boards, flats):
        # 1. 最初の特徴抽出
        x = self.conv_in(boards)
        x = F.relu(self.bn_in_2d(x))
        f = self.fc_in(flats)
        f = F.relu(self.bn_in_scalar(f))
        
        # 2. X-dim Blocks でグルグル回す
        for block in self.blocks:
            x, f = block(x, f)
            
        # 3. Heads
        # Spatial Head
        spatial_logits_map = self.spatial_head(x)  # (B, spatial_out_channels, H, W)
        spatial_logits = spatial_logits_map[:, self.idx_c, self.idx_y, self.idx_x]  # (B, num_spatial)

        # Global Head
        global_logits = self.global_head(f)  # (B, num_flat_action)

        # Combine logits
        policy_logits = torch.cat([spatial_logits, global_logits], dim=1)

        # Trade Head
        trade_query = self.trade_query_head(f)
        query_norm = F.normalize(trade_query.float(), p=2, dim=1)
        cosine_scores = torch.matmul(query_norm, self.recipe_embeddings.t())
        trade_logits = cosine_scores * self.trade_temp  # スケーリングファクター

        # Value Head
        value = torch.tanh(self.value_head(f)).squeeze(-1)

        return policy_logits, value, trade_logits
    
class TradeExpectorNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        board_channels = config["input_board_shape"][2] 
        flat_dim = config["input_scalar_dim"] + config["trade_info_dim"]
        
        # --- 1. Board Encoder ---
        self.cnn = nn.Sequential(
            nn.Conv2d(board_channels, 32, kernel_size=(3, 5), padding=(1, 2), stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 5), padding=(1, 2), stride=(2, 2)),
            nn.ReLU()
        )
        
        # --- 2. Flat Encoder ---
        self.fc_flat = nn.Sequential(
            nn.Linear(flat_dim, 64),
            nn.ReLU()
        )
        
        # --- 3. Head ---
        self.head = nn.Sequential(
            nn.Linear(64 + 64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
    def encode_board(self, boards):
        """画像から特徴量抽出まで (CNN + Mean)"""
        x = self.cnn(boards)
        return x.float().mean(dim=(2, 3)) # (B, 64)

    def head_forward(self, board_feats, flat_inputs):
        """特徴量から確率計算まで (MLP)"""
        flat_feats = self.fc_flat(flat_inputs)
        combined = torch.cat([board_feats, flat_feats], dim=1)
        return self.head(combined)

    def forward(self, boards, flat_inputs):
        """通常の一括推論"""
        f = self.encode_board(boards)
        return self.head_forward(f, flat_inputs)


class CatanXDimModel_SimplePPO(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_channels = config.get("hidden_channels", 32) 
        self.scalar_hidden_dim = config.get("scalar_hidden_dim", 64)
        num_blocks = config.get("num_xdim_blocks", 4)

        board_channels = config["input_board_shape"][2]
        flat_dim = config["input_scalar_dim"]
        self.spatial_out_channels = config["num_players"] + 3  # MT + 建設
        num_spatial = len(config["board_logits_mask"][0]) # 空間に対応するアクションの数
        num_global_action = config["action_dim"] - num_spatial
        num_opponents = config["num_players"] - 1

        # Initial layers
        self.conv_in = nn.Conv2d(board_channels, self.hidden_channels, kernel_size=(3, 5), padding=(1, 2), bias=False)
        self.bn_in_2d = nn.BatchNorm2d(self.hidden_channels)

        self.fc_in = nn.Linear(flat_dim, self.scalar_hidden_dim, bias=False)
        self.bn_in_scalar = nn.BatchNorm1d(self.scalar_hidden_dim)

        # X-dim Blocks
        self.blocks = nn.ModuleList([
            XDimBlock(
                ch_2d=self.hidden_channels, 
                dim_scalar=self.scalar_hidden_dim
            ) 
            for _ in range(num_blocks) # 指定された回数だけ積む
        ])

        self.spatial_head = nn.Conv2d(self.hidden_channels, self.spatial_out_channels, kernel_size=1) # 盤面に対応したlogits
        self.global_head = nn.Sequential(
            nn.Linear(self.scalar_hidden_dim, self.scalar_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.scalar_hidden_dim, num_global_action)
        ) # 盤面に依存しないlogits

        self.value_head = nn.Sequential(
            nn.Linear(self.scalar_hidden_dim, self.scalar_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.scalar_hidden_dim, 1) ,
        )

        # board_logitsから取り出す用 maskはTranspose済み
        self.register_buffer("idx_c", config["board_logits_mask"][0])
        self.register_buffer("idx_y", config["board_logits_mask"][1])
        self.register_buffer("idx_x", config["board_logits_mask"][2])



    def forward(self, boards, flats):
        # 1. 最初の特徴抽出
        x = self.conv_in(boards)
        x = F.relu(self.bn_in_2d(x))
        f = self.fc_in(flats)
        f = F.relu(self.bn_in_scalar(f))
        
        # 2. X-dim Blocks でグルグル回す
        for block in self.blocks:
            x, f = block(x, f)
            
        # 3. Heads
        # Spatial Head
        spatial_logits_map = self.spatial_head(x)  # (B, spatial_out_channels, H, W)
        spatial_logits = spatial_logits_map[:, self.idx_c, self.idx_y, self.idx_x]  # (B, num_spatial)

        # Global Head
        global_logits = self.global_head(f)  # (B, num_flat_action)

        # Combine logits
        policy_logits = torch.cat([spatial_logits, global_logits], dim=1)

        # Value Head
        value = torch.tanh(self.value_head(f)).squeeze(-1)

        return policy_logits, value, None # PPO用なのでtrade_logitsは返さない


class CatanXDimModel_OmitRPV(nn.Module): # 資源選好ベクトルなし
    def __init__(self, config):
        super().__init__()
        self.hidden_channels = config.get("hidden_channels", 32) 
        self.scalar_hidden_dim = config.get("scalar_hidden_dim", 64)
        num_blocks = config.get("num_xdim_blocks", 4)

        board_channels = config["input_board_shape"][2]
        flat_dim = config["input_scalar_dim"]
        self.spatial_out_channels = config["num_players"] + 3  # MT + 建設
        num_spatial = len(config["board_logits_mask"][0]) # 空間に対応するアクションの数
        num_global_action = config["action_dim"] - num_spatial
        num_opponents = config["num_players"] - 1
        full_table = pycatan.get_trade_table(player_count=config["num_players"])
        num_recipes = full_table.shape[0] // num_opponents

        # Initial layers
        self.conv_in = nn.Conv2d(board_channels, self.hidden_channels, kernel_size=(3, 5), padding=(1, 2), bias=False)
        self.bn_in_2d = nn.BatchNorm2d(self.hidden_channels)

        self.fc_in = nn.Linear(flat_dim, self.scalar_hidden_dim, bias=False)
        self.bn_in_scalar = nn.BatchNorm1d(self.scalar_hidden_dim)

        # X-dim Blocks
        self.blocks = nn.ModuleList([
            XDimBlock(
                ch_2d=self.hidden_channels, 
                dim_scalar=self.scalar_hidden_dim
            ) 
            for _ in range(num_blocks) # 指定された回数だけ積む
        ])

        self.spatial_head = nn.Conv2d(self.hidden_channels, self.spatial_out_channels, kernel_size=1) # 盤面に対応したlogits
        self.global_head = nn.Sequential(
            nn.Linear(self.scalar_hidden_dim, self.scalar_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.scalar_hidden_dim, num_global_action)
        ) # 盤面に依存しないlogits

        self.trade_head = nn.Sequential(
            nn.Linear(self.scalar_hidden_dim, self.scalar_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.scalar_hidden_dim, num_recipes) # 資源レシピ数
        )

        self.value_head = nn.Sequential(
            nn.Linear(self.scalar_hidden_dim, self.scalar_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.scalar_hidden_dim, 1) ,
        )

        # board_logitsから取り出す用 maskはTranspose済み
        self.register_buffer("idx_c", config["board_logits_mask"][0])
        self.register_buffer("idx_y", config["board_logits_mask"][1])
        self.register_buffer("idx_x", config["board_logits_mask"][2])



    def forward(self, boards, flats):
        # 1. 最初の特徴抽出
        x = self.conv_in(boards)
        x = F.relu(self.bn_in_2d(x))
        f = self.fc_in(flats)
        f = F.relu(self.bn_in_scalar(f))
        
        # 2. X-dim Blocks でグルグル回す
        for block in self.blocks:
            x, f = block(x, f)
            
        # 3. Heads
        # Spatial Head
        spatial_logits_map = self.spatial_head(x)  # (B, spatial_out_channels, H, W)
        spatial_logits = spatial_logits_map[:, self.idx_c, self.idx_y, self.idx_x]  # (B, num_spatial)

        # Global Head
        global_logits = self.global_head(f)  # (B, num_flat_action)

        # Combine logits
        policy_logits = torch.cat([spatial_logits, global_logits], dim=1)

        # Trade Head
        trade_logits = self.trade_head(f)  # (B, num_recipes)

        # Value Head
        value = torch.tanh(self.value_head(f)).squeeze(-1)

        return policy_logits, value, trade_logits

