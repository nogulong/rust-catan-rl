import pycatan
import numpy as np
import torch

def get_board_logits_mask(num_players):
    # Rustから (W, H, C) のnumpy arrayをもらう
    # 値は action_id または -1
    raw_mask = pycatan.generate_board_mask(num_players) 

    # 1. 有効なアクションIDが存在する場所のインデックスを取得    
    w_indices, h_indices, c_indices = np.where(raw_mask != -1)
    
    # 2. その場所にある アクションID そのものを取得
    action_ids = raw_mask[w_indices, h_indices, c_indices]
    
    # 3. 「アクションID順」に並び替えるためのソート順を取得
    sort_order = np.argsort(action_ids)
    
    # 4. 並び替えた順序で座標を取得し、Tensorにする
    # PyTorchの入力 (B, C, H, W) に合わせるため、(c, y, x) = (c, h, w) として扱う
    sorted_c = torch.from_numpy(c_indices[sort_order]).long()
    sorted_y = torch.from_numpy(h_indices[sort_order]).long()
    sorted_x = torch.from_numpy(w_indices[sort_order]).long()

    # out[sorted_c, sorted_y, sorted_x] = logits の形で使える, そのあとflatとconcatする

    return sorted_c, sorted_y, sorted_x

def compile_with_tensorrt(model, device, board_shape, flat_dim, target_batch_size=640, max_batch_size=1024):
    import torch_tensorrt
    model = model.half()
    model.to(device)

    inputs = [
        torch_tensorrt.Input(
            min_shape=[1, board_shape[2], board_shape[1], board_shape[0]],
            opt_shape=[target_batch_size, board_shape[2], board_shape[1], board_shape[0]],
            max_shape=[max_batch_size, board_shape[2], board_shape[1], board_shape[0]],
            dtype=torch.half,
            name="board_input"
        ),
        torch_tensorrt.Input(
            min_shape=[1, flat_dim],
            opt_shape=[target_batch_size, flat_dim],
            max_shape=[max_batch_size, flat_dim],
            dtype=torch.half,
            name="flat_input"
        )
    ]
    enabled_precisions = {torch.half}
    input_example_board = torch.randn((target_batch_size, board_shape[2], board_shape[1], board_shape[0]), device=device)
    input_example_flat = torch.randn((target_batch_size, flat_dim), device=device)
    traced_model = torch.jit.trace(model, (input_example_board.half(), input_example_flat.half()))

    with torch.cuda.device(torch.device(device)):
        trt_ts_module = torch_tensorrt.compile(
            traced_model,
            inputs=inputs,
            enabled_precisions={torch.half},
            truncate_long_and_double=True,
            ir="ts",
        )

    warmup_board = torch.randn((target_batch_size, board_shape[2], board_shape[1], board_shape[0]), device=device, dtype=torch.half)
    warmup_flat = torch.randn((target_batch_size, flat_dim), device=device, dtype=torch.half)
    _ = trt_ts_module(warmup_board, warmup_flat)

    return trt_ts_module

class TradeDecisionHandler:
    def __init__(self, config, device, use_amp=False, use_gumbel_sampling=True):
        self.config = config
        self.device = device
        self.use_amp = use_amp
        
        # 定数のキャッシュ
        self.SUPPOSE_TRADE_ID = self.config["action_dim"] - 1
        self.TRADE_INFO_DIM = self.config["trade_info_dim"]
        self.LOOKAHEAD_BATCH_SIZE = self.config.get("lookahead_batch_size", 2048)

        num_players = self.config["num_players"]
        self.num_opponents = num_players - 1
        table_int8 = pycatan.get_trade_table(num_players)
        full_table = torch.from_numpy(table_int8).to(self.device, dtype=torch.float32)
        # A. flat更新用
        self.delta_res_table = full_table[:, :5] # (Actions, 5)
        self.delta_hidden_table = full_table[:, self.TRADE_INFO_DIM :]
        # B. NN入力用
        self.feat_table = full_table[:, : self.TRADE_INFO_DIM] # (Actions, TradeInfoDim)
        self.use_gumbel_sampling = use_gumbel_sampling

    def execute_lookahead(self, policy_model, trade_model, board_batch, flat_batch, original_actions, v_curr, trade_masks, trade_logits, fast_mode=False):

        batch_size = original_actions.shape[0]
        final_actions = original_actions.clone()
        final_trade_infos = np.zeros((batch_size, self.TRADE_INFO_DIM), dtype=np.float32)

        # 1. ~ 3. 準備
        is_trader = (original_actions == self.SUPPOSE_TRADE_ID)
        trader_indices = torch.where(is_trader)[0]
        num_traders = trader_indices.shape[0]
        if num_traders == 0: return final_actions, final_trade_infos

        trader_boards = board_batch[trader_indices]
        trader_flats = flat_batch[trader_indices].clone()

        with torch.no_grad(), torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=torch.bfloat16):
            trader_board_feats = trade_model.encode_board(trader_boards)

        current_recipe_logits = trade_logits[trader_indices]
        trader_action_logits = current_recipe_logits.repeat_interleave(self.num_opponents, dim=1)
        
        current_trade_masks = trade_masks[trader_indices]
        masked_logits = trader_action_logits.masked_fill(~current_trade_masks.bool(), -float('inf'))

        candidates = torch.nonzero(current_trade_masks, as_tuple=False)
        relative_indices = candidates[:, 0] 
        valid_trade_ids = candidates[:, 1]
        total_candidates = valid_trade_ids.shape[0]
        if total_candidates == 0: return final_actions, final_trade_infos

        if fast_mode: # シンプルにargmaxで決定, Archiveモード用, 受諾率で相手を決定
            best_trade_ids = torch.argmax(masked_logits, dim=1)
            best_recipe_indices = best_trade_ids // self.num_opponents
            start_ids = best_recipe_indices * self.num_opponents
            candidate_ids_matrix = start_ids.unsqueeze(1) + torch.arange(self.num_opponents, device=self.device).unsqueeze(0)
            batch_rel_idx = torch.arange(num_traders, device=self.device).repeat_interleave(self.num_opponents)
            target_trade_ids = candidate_ids_matrix.view(-1)
            chunk_board_feats = trader_board_feats[batch_rel_idx]
            chunk_flats = trader_flats[batch_rel_idx]
            trade_feat = self.feat_table[target_trade_ids]
            chunk_trade_input = torch.cat([chunk_flats, trade_feat], dim=1)
            with torch.no_grad(), torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=torch.bfloat16):
                p_logits = trade_model.head_forward(chunk_board_feats, chunk_trade_input)
                p_accept = torch.sigmoid(p_logits).squeeze(-1)

            p_accept_matrix = p_accept.view(num_traders, self.num_opponents)
            best_partner_rel_idx = torch.argmax(p_accept_matrix, dim=1)

            final_update_ids = self.SUPPOSE_TRADE_ID + (start_ids + best_partner_rel_idx)
            final_actions[trader_indices] = final_update_ids
            return final_actions, final_trade_infos # 学習対象ではないので, trade_infoは空でよい


        # ---------------------------------------------
        # 本格的一手読み処理
        # ---------------------------------------------

        # 3. Gumbelサンプリングで候補スコア計算
        candidate_scores = masked_logits[relative_indices, valid_trade_ids]
        if self.use_gumbel_sampling:
            uniform_noise = torch.rand_like(candidate_scores)
            eps = 1e-9
            gumbel_noise = -torch.log(-torch.log(uniform_noise + eps) + eps)
            candidate_scores += gumbel_noise
        # 4. top-K pruning
        K = self.config.get("trade_pruning_k", 10)
        counts = torch.bincount(relative_indices, minlength=num_traders).cpu().tolist()
        score_chunks = torch.split(candidate_scores, counts)
        id_chunks = torch.split(valid_trade_ids, counts)
        selected_global_indices = []
        selected_trade_ids = []
        for i, (scores, ids) in enumerate(zip(score_chunks, id_chunks)):
            if scores.numel() > 0:
                k_actual = min(K, scores.numel())
                _, idxs = torch.topk(scores, k_actual)
                selected_global_indices.extend([i] * k_actual)
                selected_trade_ids.append(ids[idxs])

        # 5. 受諾率計算
        target_rel_indices = torch.tensor(selected_global_indices, device=self.device, dtype=torch.long)
        target_trade_ids = torch.cat(selected_trade_ids)

        chunk_board_feats = trader_board_feats[target_rel_indices]
        chunk_flats = trader_flats[target_rel_indices]
        trade_feat = self.feat_table[target_trade_ids]
        chunk_trade_input = torch.cat([chunk_flats, trade_feat], dim=1)

        with torch.no_grad(), torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=torch.bfloat16):
            p_logits = trade_model.head_forward(chunk_board_feats, chunk_trade_input)
            target_p_accept = torch.sigmoid(p_logits).squeeze(-1)
        
        # 6. 一手読み価値計算
        target_global_traders = trader_indices[target_rel_indices]
        target_boards = board_batch[target_global_traders]
        target_flats_next = flat_batch[target_global_traders].clone()

        target_deltas_res = self.delta_res_table[target_trade_ids]
        target_deltas_hidden = self.delta_hidden_table[target_trade_ids]
        target_flats_next[:, 0:5] += target_deltas_res
        for p in range(self.num_opponents):
            target_flats_next[:, 29 + p*9] += target_deltas_hidden[:, p]
        
        with torch.no_grad(), torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=torch.bfloat16):
            _, v_next, _ = policy_model(target_boards, target_flats_next)
            v_next = v_next.squeeze(-1).float()
        
        # 7. 最終スコア計算 & 更新
        v_curr_target = v_curr[target_global_traders]
        if v_curr_target.ndim == 1: v_curr_target = v_curr_target.unsqueeze(-1)
        v_curr_target = v_curr_target.squeeze(-1)

        final_scores = target_p_accept * v_next + (1.0 - target_p_accept) * v_curr_target

        counts_k = torch.bincount(target_rel_indices, minlength=num_traders).cpu().tolist()
        score_chunks_k = torch.split(final_scores, counts_k)
        id_chunks_k = torch.split(target_trade_ids, counts_k)
        
        final_update_indices = []
        final_update_ids = []
        
        for i, (scores, ids) in enumerate(zip(score_chunks_k, id_chunks_k)):
            if scores.numel() > 0:
                best_idx = torch.argmax(scores)
                final_update_indices.append(trader_indices[i].item())
                final_update_ids.append(self.SUPPOSE_TRADE_ID + ids[best_idx])

        if final_update_indices:
            final_actions[final_update_indices] = torch.tensor(final_update_ids, device=self.device, dtype=torch.long)
            
            table_ids_for_feat = torch.tensor(final_update_ids, device=self.device) - self.SUPPOSE_TRADE_ID
            feats = self.feat_table[table_ids_for_feat].cpu().numpy()
            final_trade_infos[final_update_indices] = feats

        return final_actions, final_trade_infos