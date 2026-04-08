import numpy as np
from . import _pycatan
import json

class SingleVecEnvState:
    def __init__(self, num_envs, opponents=3, trade_activated=False, trade_limit=1):
        self.num_envs = num_envs
        self.opponents = opponents
        self.trade_activated = trade_activated
        self.trade_limit = trade_limit
        self.envs = _pycatan.SingleVecEnvironment.new(
            num_envs,
            _pycatan.PyObservationFormat(),
            opponents=opponents,
            trade_activated=trade_activated,
            trade_limit=trade_limit,
        )
        player_id, board, flat, normal_legal_mask, trade_legal_mask, position, done = self.envs.start(0)
        self.current_boards = np.zeros((num_envs, board.shape[0], board.shape[1], board.shape[2]), dtype=board.dtype)
        self.current_flats = np.zeros((num_envs, flat.shape[0]), dtype=flat.dtype)
        self.current_normal_legal_masks = np.zeros((num_envs, normal_legal_mask.shape[0]), dtype=normal_legal_mask.dtype)
        self.current_trade_legal_masks = np.zeros((num_envs, trade_legal_mask.shape[0]), dtype=trade_legal_mask.dtype)
        self.current_positions = np.zeros((num_envs,), dtype=np.int8)
        self.player_positions = np.zeros((num_envs, 4), dtype=np.int32)
        self.dones = np.array([True] * num_envs)
        self.player_ids = np.zeros((num_envs,), dtype=np.int32)
        self.current_step_counts = np.zeros(num_envs, dtype=np.int32)

        self.current_boards[0] = board
        self.current_flats[0] = flat
        self.current_normal_legal_masks[0] = normal_legal_mask
        self.current_trade_legal_masks[0] = trade_legal_mask
        self.current_positions[0] = position
        self.player_positions[0][player_id] = position
        self.dones[0] = done
        self.player_ids[0] = player_id

        for env_index in range(1, num_envs):
            self.start(env_index)
    
    def start(self, env_index):
        player_id, board, flat, normal_legal_mask, trade_legal_mask, position, done = self.envs.start(env_index)
        self.current_boards[env_index] = board
        self.current_flats[env_index] = flat
        self.current_normal_legal_masks[env_index] = normal_legal_mask
        self.current_trade_legal_masks[env_index] = trade_legal_mask
        self.current_positions[env_index] = position
        self.player_positions[env_index, player_id] = position
        self.dones[env_index] = done
        self.player_ids[env_index] = player_id
        self.current_step_counts[env_index] = 0
    
    def step(self, actions):
        player_ids, boards, flats, normal_legal_masks, trade_legal_masks, positions, dones = self.envs.play(actions)
        self.current_boards = boards
        self.current_flats = flats
        self.current_normal_legal_masks = normal_legal_masks
        self.current_trade_legal_masks = trade_legal_masks
        self.current_positions = positions
        self.player_positions[np.arange(self.num_envs), player_ids] = positions
        self.dones = dones
        self.player_ids = player_ids
        self.current_step_counts += 1
    
    def reset(self, env_index):
        player_id, board, flat, normal_legal_mask, trade_legal_mask, position, done = self.envs.reset(env_index,)
        self.current_boards[env_index] = board
        self.current_flats[env_index] = flat
        self.current_normal_legal_masks[env_index] = normal_legal_mask
        self.current_trade_legal_masks[env_index] = trade_legal_mask
        self.current_positions[env_index] = position
        self.player_positions[env_index, player_id] = position
        self.dones[env_index] = done
        self.player_ids[env_index] = player_id
        self.current_step_counts[env_index] = 0

    def get_history(self, env_index):
        history_and_summary = self.envs.export_history(env_index)
        positions_list = self.player_positions[env_index].tolist()
        return (history_and_summary, positions_list)
    
    def result(self, env_index):
        return self.envs.result(env_index)
    
    def close(self):
        self.envs.close()
                
        

class MultiVecEnvState:
    def __init__(self, num_envs, players=4, trade_activated=False, trade_limit=1):
        self.num_envs = num_envs
        self.players = players
        self.trade_activated = trade_activated
        self.trade_limit = trade_limit
        self.envs = _pycatan.MultiVecEnvironment.new(
            num_envs,
            _pycatan.PyObservationFormat(),
            players=players,
            trade_activated=trade_activated,
            trade_limit=trade_limit,
        )
        player_id, board, flat, normal_legal_mask, trade_legal_mask, position, done = self.envs.start(0)
        self.current_boards = np.zeros((num_envs, board.shape[0], board.shape[1], board.shape[2]), dtype=board.dtype)
        self.current_flats = np.zeros((num_envs, flat.shape[0]), dtype=flat.dtype)
        self.current_normal_legal_masks = np.zeros((num_envs, normal_legal_mask.shape[0]), dtype=normal_legal_mask.dtype)
        self.current_trade_legal_masks = np.zeros((num_envs, trade_legal_mask.shape[0]), dtype=trade_legal_mask.dtype)
        self.current_positions = np.zeros((num_envs,), dtype=np.int8)
        self.player_positions = np.tile(np.arange(4, dtype=np.int32), (num_envs, 1))
        self.dones = np.array([True] * num_envs)
        self.player_ids = np.zeros((num_envs,), dtype=np.int32)
        self.current_step_counts = np.zeros(num_envs, dtype=np.int32)

        self.current_boards[0] = board
        self.current_flats[0] = flat
        self.current_normal_legal_masks[0] = normal_legal_mask
        self.current_trade_legal_masks[0] = trade_legal_mask
        self.current_positions[0] = position
        self.player_positions[0][player_id] = position
        self.dones[0] = done
        self.player_ids[0] = player_id

        for env_index in range(1, num_envs):
            self.start(env_index)
    
    def start(self, env_index):
        player_id, board, flat, normal_legal_mask, trade_legal_mask, position, done = self.envs.start(env_index)
        self.current_boards[env_index] = board
        self.current_flats[env_index] = flat
        self.current_normal_legal_masks[env_index] = normal_legal_mask
        self.current_trade_legal_masks[env_index] = trade_legal_mask
        self.current_positions[env_index] = position
        self.player_positions[env_index, player_id] = position
        self.dones[env_index] = done
        self.player_ids[env_index] = player_id
        self.current_step_counts[env_index] = 0

    def step(self, actions):
        players_to_pass = self.player_ids.astype(np.uint8).tolist()
        player_ids, boards, flats, normal_legal_masks, trade_legal_masks, positions, dones = self.envs.play(players_to_pass, actions)
        self.current_boards = boards
        self.current_flats = flats
        self.current_normal_legal_masks = normal_legal_masks
        self.current_trade_legal_masks = trade_legal_masks
        self.current_positions = positions
        self.player_positions[np.arange(self.num_envs), player_ids] = positions
        self.dones = dones
        self.player_ids = player_ids
        self.current_step_counts += 1
    
    def reset(self, env_index):
        player_id, board, flat, normal_legal_mask, trade_legal_mask, position, done = self.envs.reset(env_index, self.player_ids[env_index].astype(np.uint8))
        self.current_boards[env_index] = board
        self.current_flats[env_index] = flat
        self.current_normal_legal_masks[env_index] = normal_legal_mask
        self.current_trade_legal_masks[env_index] = trade_legal_mask
        self.current_positions[env_index] = position
        self.player_positions[env_index, player_id] = position
        self.dones[env_index] = done
        self.player_ids[env_index] = player_id
        self.current_step_counts[env_index] = 0

    def get_history(self, env_index):
        history_and_summary = self.envs.export_history(env_index)
        positions_list = self.player_positions[env_index].tolist()
        return (history_and_summary, positions_list)
    
    def result(self, env_index):
        return self.envs.result(env_index)
    
    def close(self):
        self.envs.close()

def save_args(args, save_path):
    with open(save_path, 'w') as f:
        json.dump(args.__dict__, f, indent=4)

def load_args(args, load_path):
    with open(load_path, 'r') as f:
        saved_args = json.load(f)
    # 引数を上書き（ただし、resume用の引数などは除く）
    for k, v in saved_args.items():
        if k not in ['resume_dir', 'data_dir', 'model_dir', 'train_steps']: 
            setattr(args, k, v)
    return args

class TradeDecisionHandler_onnx: # torchに非依存, 推論もonnxで行う
    def __init__(self, config, device='cpu'):
        self.config = config
        self.device = device
        
        # 定数のキャッシュ
        self.SUPPOSE_TRADE_ID = self.config["action_dim"] - 1
        self.TRADE_INFO_DIM = self.config["trade_info_dim"]

        num_players = self.config["num_players"]
        self.num_opponents = num_players - 1
        full_table = _pycatan.get_trade_table(num_players).astype(np.float32)
        # A. flat更新用
        self.delta_res_table = full_table[:, :5] # (Actions, 5)
        self.delta_hidden_table = full_table[:, self.TRADE_INFO_DIM :]
        # B. NN入力用
        self.feat_table = full_table[:, : self.TRADE_INFO_DIM] # (Actions, TradeInfoDim)

    def execute_lookahead(self, policy_model, trade_model, board_batch, flat_batch, original_actions, v_curr, trade_masks, trade_logits):

        batch_size = original_actions.shape[0]
        final_actions = original_actions.copy()
        final_trade_infos = np.zeros((batch_size, self.TRADE_INFO_DIM), dtype=np.float32)

        # 1. ~ 3. 準備
        is_trader = (original_actions == self.SUPPOSE_TRADE_ID)
        trader_indices = np.where(is_trader)[0]
        num_traders = trader_indices.shape[0]
        if num_traders == 0: return final_actions, final_trade_infos

        trader_boards = board_batch[trader_indices]
        trader_flats = flat_batch[trader_indices].copy()

        current_recipe_logits = trade_logits[trader_indices]
        trader_action_logits = current_recipe_logits.repeat(self.num_opponents, axis=1)
        current_trade_masks = trade_masks[trader_indices]
        masked_logits = np.where(current_trade_masks > 0, trader_action_logits, -np.inf)

        candidates = np.argwhere(current_trade_masks > 0)
        relative_indices = candidates[:, 0] 
        valid_trade_ids = candidates[:, 1]
        total_candidates = valid_trade_ids.shape[0]
        if total_candidates == 0: return final_actions, final_trade_infos

        # ---------------------------------------------
        # 本格的一手読み処理
        # ---------------------------------------------

        # 3. Gumbelサンプリングで候補スコア計算 (対戦用なので省略)
        candidate_scores = masked_logits[relative_indices, valid_trade_ids]

        selected_global_indices = []
        selected_trade_ids = []

        # 4. top-K pruning
        K = self.config.get("trade_pruning_k", 10)
        for i in range(num_traders):
            mask = (relative_indices == i)
            i_scores = candidate_scores[mask]
            i_ids = valid_trade_ids[mask]
            k_actual = min(K, i_scores.shape[0])
            top_k_rel_indices = np.argsort(-i_scores)[:k_actual]
            selected_global_indices.extend([i] * k_actual)
            selected_trade_ids.append(i_ids[top_k_rel_indices])
        
        target_rel_indices = np.array(selected_global_indices, dtype=np.int64)
        target_trade_ids = np.concatenate(selected_trade_ids)

        chunk_boards = trader_boards[target_rel_indices]
        chunk_flats = trader_flats[target_rel_indices]
        trade_feats = self.feat_table[target_trade_ids]
        chunk_trade_input = np.concatenate([chunk_flats, trade_feats], axis=1)

        p_logits_out = trade_model.infer_naive(chunk_boards, chunk_trade_input)
        target_p_accept = 1.0 / (1.0 + np.exp(-p_logits_out[0].reshape(-1))) # sigmoid
        
        # 6. 一手読み価値計算
        target_global_traders = trader_indices[target_rel_indices]
        target_boards = board_batch[target_global_traders]
        target_flats_next = flat_batch[target_global_traders].copy().astype(np.float32)

        target_deltas_res = self.delta_res_table[target_trade_ids]
        target_deltas_hidden = self.delta_hidden_table[target_trade_ids]
        target_flats_next[:, 0:5] += target_deltas_res
        for p in range(self.num_opponents):
            target_flats_next[:, 29 + p*9] += target_deltas_hidden[:, p]
        
        policy_out = policy_model.infer_naive(target_boards, target_flats_next)
        v_next = policy_out[1].reshape(-1)
        
        # 7. 最終スコア計算 & 更新
        v_curr_target = v_curr[target_global_traders]
        v_curr_target = v_curr_target.reshape(-1)

        final_scores = target_p_accept * v_next + (1.0 - target_p_accept) * v_curr_target

        for i in range(num_traders):
                mask = (target_rel_indices == i)
                if not np.any(mask): continue
                
                i_final_scores = final_scores[mask]
                i_trade_ids = target_trade_ids[mask]
                
                best_idx = np.argmax(i_final_scores)
                global_idx = trader_indices[i]
                
                final_actions[global_idx] = self.SUPPOSE_TRADE_ID + i_trade_ids[best_idx]
                
                # trade_info の更新
                best_trade_id = i_trade_ids[best_idx]
                final_trade_infos[global_idx] = self.feat_table[best_trade_id]

        return final_actions, final_trade_infos
