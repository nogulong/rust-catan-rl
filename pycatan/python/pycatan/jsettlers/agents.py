import numpy as np
from pycatan.utils import TradeDecisionHandler_onnx
from pycatan.inference import OnnxInfer, OnnxInfer_TradeExpector
from importlib import resources

# バッチサイズ1での動作を想定
class BaseAgent:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        pkg_root = resources.files("pycatan")
        self.main_model_path = str(pkg_root / "main_model.onnx")
        self.trade_model_path = str(pkg_root / "trade_model.onnx")
        self.policy_model = OnnxInfer(self.main_model_path, device)
        self.trade_model = OnnxInfer_TradeExpector(self.trade_model_path, device)

        self.TRADE_ACCEPT_ID = self.config["action_dim"] - 3  # 取引受諾アクションID
        self.TRADE_REJECT_ID = self.config["action_dim"] - 2  # 取引拒否アクションID
        self.TRADE_PROPOSE_ID = self.config["action_dim"] - 1 # 取引提案アクションID

        self.config["trade_pruning_k"] = 50  # 対戦時は大きめに, 全部読んでもいいかも, gumbel noiseはいれない
        self.trade_handler = TradeDecisionHandler_onnx(
            self.config, self.device
        )

    def act(self, board, flat, normal_mask, trade_mask=None):
        raise NotImplementedError

class NoTradeAgent(BaseAgent): # 提案しないし, 受諾もしない

    def act(self, board, flat, normal_mask, trade_mask=None):
        # 先頭にバッチ次元を追加
        board = board[np.newaxis, ...]  # (w, h, c) -> (1, w, h, c)
        flat = flat[np.newaxis, ...]   # (scalar_dim,) -> (1, scalar_dim)
        normal_mask = normal_mask[np.newaxis, ...]  # (action_dim,) -> (1, action_dim)

        board = np.transpose(board, (0, 3, 2, 1)) # (w, h, c) -> (c, h, w)

        policy_out = self.policy_model.infer_naive(board, flat)
        logits = policy_out[0]  # (action_dim,)
        masked_logits = np.where(normal_mask, logits, -np.inf)
        masked_logits[:, self.TRADE_PROPOSE_ID] = -np.inf
        masked_logits[:, self.TRADE_ACCEPT_ID] = -np.inf
        action = np.argmax(masked_logits, axis=1)

        return action.item()

class WithTradeAgent(BaseAgent): # 一手読みで取引を行うエージェント

    def act(self, board, flat, normal_mask, trade_mask):
        # バッチ次元の追加
        board = board[np.newaxis, ...]  # (w, h, c) -> (1, w, h, c)
        flat = flat[np.newaxis, ...]   # (scalar_dim,) -> (1, scalar_dim)
        normal_mask = normal_mask[np.newaxis, ...]  # (action_dim,) -> (1, action_dim)
        trade_mask = trade_mask[np.newaxis, ...]  # (trade_action_dim,) -> (1, trade_action_dim)

        board = np.transpose(board, (0, 3, 2, 1)) # (w, h, c) -> (c, h, w)

        policy_out = self.policy_model.infer_naive(board, flat)
        logits = policy_out[0]  # (action_dim,)
        masked_logits = np.where(normal_mask, logits, -np.inf)
        initial_action = np.argmax(masked_logits, axis=1)

        final_actions, _ = self.trade_handler.execute_lookahead(
            policy_model=self.policy_model,
            trade_model=self.trade_model,
            board_batch=board,
            flat_batch=flat,
            original_actions=initial_action, # Tensor shape (1,)
            v_curr=policy_out[1].reshape(-1),       # Tensor shape (1,)
            trade_masks=trade_mask,
            trade_logits=policy_out[2],
        )
            
        return final_actions.item()

