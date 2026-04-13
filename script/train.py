import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["RAYON_NUM_THREADS"] = "4" 
# os.environ["OPENBLAS_NUM_THREADS"] = "1"

import ray
import numpy as np
import time
from collections import deque
import msgpack
import zstandard as zstd
import datetime
import argparse
import threading
import random
import json
import asyncio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler


from typing import Tuple, List, Dict

from pycatan import MultiVecEnvState
from pycatan.train import CatanXDimModel, TradeExpectorNet, CatanXDimModel_SimplePPO, CatanXDimModel_OmitRPV, get_board_logits_mask, TradeDecisionHandler
from pycatan.utils import load_args, save_args 
    
@ray.remote
class ReplayBuffer:
    def __init__(self, capacity=10):
        self.buffer = deque(maxlen=capacity)

    def add(self, chunk):
        self.buffer.append(chunk)
    
    def sample(self, required_chunks):
        if len(self.buffer) < required_chunks:
            return None
        
        chunk_to_send = []
        chunks_collected = 0
        
        while chunks_collected < required_chunks:
            chunk = self.buffer.popleft()
            chunk_to_send.append(chunk)
            chunks_collected += 1
        
        batch = {
            "boards": np.concatenate([c["boards"] for c in chunk_to_send], axis=0),
            "flats": np.concatenate([c["flats"] for c in chunk_to_send], axis=0),
            "actions": np.concatenate([c["actions"] for c in chunk_to_send], axis=0),
            "masks": np.concatenate([c["masks"] for c in chunk_to_send], axis=0),
            "trade_masks": np.concatenate([c["trade_masks"] for c in chunk_to_send], axis=0),
            "log_probs": np.concatenate([c["log_probs"] for c in chunk_to_send], axis=0),
            "player_ids": np.concatenate([c["player_ids"] for c in chunk_to_send], axis=0),
            "trade_infos": np.concatenate([c["trade_infos"] for c in chunk_to_send], axis=0),
            "rewards": np.concatenate([c["rewards"] for c in chunk_to_send], axis=0),
            "lengths": np.concatenate([c["lengths"] for c in chunk_to_send], axis=0),
        }
        return batch
    
    def size(self):
        return len(self.buffer)
    
@ray.remote
class InferenceServer:
    def __init__(self, learner_ref, config):
        self.learner_ref = learner_ref
        self.config = config
        self.device = torch.device(config["inference_device"])
        self.use_amp = self.device.type == "cuda"
        self.use_archive = config["use_archive"]

        match config["policy_model_class"]:
            case "CatanXDimModel":
                self.PolicyModel = CatanXDimModel
            case "CatanXDimModel_SimplePPO":
                self.PolicyModel = CatanXDimModel_SimplePPO
            case "CatanXDimModel_OmitRPV":
                self.PolicyModel = CatanXDimModel_OmitRPV
            case _:
                # どれにも当てはまらない場合のデフォルト設定
                self.PolicyModel = CatanXDimModel


        # モデル定義 (ここではまだ重みをロードしない)
        self.policy_model = self.PolicyModel(config).to(self.device)
        self.policy_model.eval()
        self.trade_model = TradeExpectorNet(config).to(self.device)
        self.trade_model.eval()
        
        # アーカイブモデルの準備
        if self.use_archive:
            self.archive_models = [self.PolicyModel(config).to(self.device) for _ in range(self.config["archive_size"])]
            self.archive_trade_models = [TradeExpectorNet(config).to(self.device) for _ in range(self.config["archive_size"])]
            self.active_slot = list(range(self.config["archive_slot_size"]))
        
            self.archive_save_paths = [os.path.join(
                self.config["exp_dir"],
                f"archive_model_slot_{i}.pth"
            ) for i in range(self.config["archive_size"])]

        self.server_state_path = os.path.join(self.config["exp_dir"], "inference_server_state.json")

        self.oldest_ptr = 0 
        self.last_weights_version = -1
        
        self.request_queue = deque()
        
        self.pending_weights_future = None
        self.archive_update_steps = 0
        self.slot_update_steps = 0
        self.steps = 0
        
        self.trade_handler = TradeDecisionHandler(
            self.config, self.device, self.use_amp
        )

    # 重みの初期ロードを行う非同期メソッド
    async def initialize_weights(self):
        print("[InferenceServer] Initializing weights...")
        initial_version, initial_weights_ref = await self.learner_ref.get_weights.remote()
        initial_weights = await initial_weights_ref
        
        self.policy_model.load_state_dict(initial_weights["policy"])
        self.trade_model.load_state_dict(initial_weights["trade"])
        self.last_weights_version = initial_version
        print(f"[InferenceServer] Loaded initial weights version {initial_version}")

        # アーカイブのロード
        if self.use_archive:
            if self.config["resume_dir"] is not None:
                if os.path.exists(self.server_state_path):
                    with open(self.server_state_path, "r") as f:
                        state = json.load(f)
                        self.oldest_ptr = state.get("oldest_ptr", 0)
                
                print("[InferenceServer] Loading archive models...")
                for i in range(self.config["archive_size"]):
                    load_path = self.archive_save_paths[i]
                    if os.path.exists(load_path):
                        checkpoint = torch.load(load_path, map_location=self.device)
                        
                        # 古い形式(Policyのみ)か新しい形式(Dict)か判定
                        if "policy" in checkpoint:
                            self.archive_models[i].load_state_dict(checkpoint["policy"])
                            self.archive_trade_models[i].load_state_dict(checkpoint["trade"])
                        else:
                            # 古いファイルならPolicyとしてロードし、Tradeは初期重み(or 現在の重み)
                            self.archive_models[i].load_state_dict(checkpoint)
                            self.archive_trade_models[i].load_state_dict(initial_weights["trade"])
                    else:
                        self.archive_models[i].load_state_dict(initial_weights["policy"])
                        self.archive_trade_models[i].load_state_dict(initial_weights["trade"])
                
                    self.archive_models[i].eval()
                    self.archive_trade_models[i].eval()
            else:
                for model in self.archive_models:
                    model.load_state_dict(initial_weights["policy"])
                    model.eval()
                for model in self.archive_trade_models:
                    model.load_state_dict(initial_weights["trade"])
                    model.eval()

    # async def にし、現在のループからFutureを作る
    @ray.method(num_returns=1)
    async def inference(self, boards, flats, normal_masks, trade_masks, versions):
        """Actorが呼ぶメソッド"""
        loop = asyncio.get_running_loop() # 現在のループを取得
        future = loop.create_future()     # そのループ用のFutureを作成
        
        self.request_queue.append({
            "future": future,
            "boards": boards,
            "flats": flats,
            "normal_masks": normal_masks,
            "trade_masks": trade_masks,
            "versions": versions
        })
        
        # 処理が終わるまで待機 (CPUを使わずに待てる)
        result = await future
        return result

    # データ処理後にFutureに結果をセットする
    def _process_batch_and_respond(self, batch_data):

        # 1. データ結合 (CPU)
        boards = np.concatenate([item["boards"] for item in batch_data], axis=0)
        flats = np.concatenate([item["flats"] for item in batch_data], axis=0)
        normal_masks = np.concatenate([item["normal_masks"] for item in batch_data], axis=0)
        trade_masks = np.concatenate([item["trade_masks"] for item in batch_data], axis=0)
        versions = np.concatenate([item["versions"] for item in batch_data], axis=0)
        
        sort_indices = np.argsort(versions)
        sorted_versions = versions[sort_indices]
        unique_versions, boundaries = np.unique(sorted_versions, return_index=True)
        boundaries = np.append(boundaries, len(versions))

        total_size = boards.shape[0]
        all_actions = np.zeros((total_size,), dtype=np.uint16)
        all_log_probs = np.zeros((total_size,), dtype=np.float32)
        all_trade_infos = np.zeros((total_size, self.config["trade_info_dim"]), dtype=np.float32)

        with torch.no_grad():
            for i, version in enumerate(unique_versions):
                start = boundaries[i]
                end = boundaries[i + 1]
                version_indices_sorted = sort_indices[start:end]

                # 2. GPU転送
                board_gpu = torch.from_numpy(boards[version_indices_sorted]).to(self.device, dtype=torch.float32).permute(0,3,2,1)
                flat_gpu = torch.from_numpy(flats[version_indices_sorted]).to(self.device, dtype=torch.float32)
                normal_mask_gpu = torch.from_numpy(normal_masks[version_indices_sorted]).to(self.device)
                trade_mask_gpu = torch.from_numpy(trade_masks[version_indices_sorted]).to(self.device)

                if version == self.config["archive_slot_size"] or not self.use_archive:
                    model = self.policy_model
                    trade_model = self.trade_model
                else:
                    model = self.archive_models[self.active_slot[version]]
                    trade_model = self.archive_trade_models[self.active_slot[version]]

                # 3. ベース推論 (Policy)
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=torch.bfloat16):
                    policy_logits, v_curr, trade_logits = model(board_gpu, flat_gpu)

                policy_logits = policy_logits.float()
                masked_logits = policy_logits.masked_fill(~normal_mask_gpu.bool(), -float('inf'))
                log_probs = F.log_softmax(masked_logits, dim=-1)
                actions = torch.multinomial(log_probs.exp(), num_samples=1).reshape(-1)
                log_probs = log_probs.gather(1, actions.unsqueeze(-1)).reshape(-1)

                if self.config["simple_ppo"]:
                    all_actions[version_indices_sorted] = actions.cpu().numpy().astype(np.uint16)
                    all_log_probs[version_indices_sorted] = log_probs.cpu().numpy().astype(np.float32)
                    continue
                else:
                    # 4. 一手読み (Lookahead)
                    v_curr_flat = v_curr.reshape(-1)

                    final_actions, final_trade_infos = self.trade_handler.execute_lookahead(
                        policy_model=model,
                        trade_model=trade_model,
                        board_batch=board_gpu,
                        flat_batch=flat_gpu,
                        original_actions=actions,
                        v_curr=v_curr_flat,
                        trade_masks=trade_mask_gpu,
                        trade_logits=trade_logits,
                        fast_mode=(not version == self.config["archive_slot_size"] and self.use_archive)
                    )

                    all_actions[version_indices_sorted] = final_actions.cpu().numpy().astype(np.uint16)
                    all_log_probs[version_indices_sorted] = log_probs.cpu().numpy().astype(np.float32)
                    all_trade_infos[version_indices_sorted] = final_trade_infos

        # 結果を返却
        cursor = 0
        for req in batch_data:
            size = req["boards"].shape[0]
            future = req["future"]
            
            actions = all_actions[cursor:cursor + size]
            log_probs = all_log_probs[cursor:cursor + size]
            trade_infos = all_trade_infos[cursor:cursor + size]
            
            if not future.cancelled():
                future.set_result((actions, log_probs, trade_infos))
            
            cursor += size

    async def run_loop(self):
        BATCH_SIZE_THRESHOLD = self.config["inference_batch_size"]

        while True:
            # --- 重み更新処理 ---
            if self.pending_weights_future is not None:
                ready_refs, remaining_refs = ray.wait([self.pending_weights_future], timeout=0)
                if ready_refs:
                    new_version, new_weights_ref = await ready_refs[0] # await
                    
                    if new_version > self.last_weights_version:
                        # アーカイブ更新
                        if self.use_archive and self.archive_update_steps >= self.config["archive_update_interval"]:
                            print("[InferenceServer] Updating archive models.")
                            self.archive_models[self.oldest_ptr].load_state_dict(self.policy_model.state_dict())
                            self.archive_trade_models[self.oldest_ptr].load_state_dict(self.trade_model.state_dict())
                            save_path = self.archive_save_paths[self.oldest_ptr]
                            save_dict = {
                                "policy": self.archive_models[self.oldest_ptr].state_dict(),
                                "trade": self.archive_trade_models[self.oldest_ptr].state_dict()
                            }
                            torch.save(save_dict, save_path)
                            self.oldest_ptr = (self.oldest_ptr + 1) % self.config["archive_size"]
                            with open(self.server_state_path, "w") as f:
                                json.dump({"oldest_ptr": self.oldest_ptr, "steps": self.steps}, f)
                            self.archive_update_steps = 0

                        new_weights = await new_weights_ref # await
                        self.policy_model.load_state_dict(new_weights["policy"])
                        self.trade_model.load_state_dict(new_weights["trade"])
                        self.last_weights_version = new_version
                    
                    self.pending_weights_future = None

            if self.pending_weights_future is None and (self.steps % self.config["load_interval"] == 0):
                self.pending_weights_future = self.learner_ref.get_weights.remote()

            # --- 推論処理 ---
            if not self.request_queue:
                await asyncio.sleep(0.0001) 
                continue
            
            collect_batch = []
            while self.request_queue and len(collect_batch) < BATCH_SIZE_THRESHOLD:
                collect_batch.append(self.request_queue.popleft())
            
            # 推論実行
            self._process_batch_and_respond(collect_batch)
            
            self.steps += 1
            
            if self.use_archive:
                self.archive_update_steps += 1
                self.slot_update_steps += 1

                if self.steps % self.config["slot_update_interval"] == 0:
                    print("[InferenceServer] Updating active archive slots.")
                    self.active_slot = random.sample(
                        range(self.config["archive_size"]),
                        self.config["archive_slot_size"]
                    )
    
# 棋譜生成プロセス 決められた数棋譜を生成 → ReplayBuffer に送信
@ray.remote
class Actor:
    def __init__(self, id, buffer_ref, learner_ref, inference_server_ref, config):
        try:
            self.id = id
            self.buffer_ref = buffer_ref
            self.learner_ref = learner_ref
            self.inference_server = inference_server_ref
            self.config = config
            self.device = torch.device(config["actor_device"])

            self.envs = MultiVecEnvState(
                num_envs=self.config["num_envs"],
                players=self.config["num_players"],
                trade_activated= not self.config["trade_deactivated"],
                trade_limit=self.config["trade_limit"],
            )
            self.file_count = 0
            self.data_dir = os.path.join(self.config["exp_dir"], "data")
            os.makedirs(self.data_dir, exist_ok=True)
            if self.config["resume_dir"] is not None:
                self.file_count = len([name for name in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, name)) and name.startswith("catan_batch_")])
            print(f"[Actor] Initialized with {self.config['num_envs']} environments.")

        except Exception as e:
            print("Actor encountered an exception during initialization:", e)
            raise e

        self.pending_weights_future = None
        self.last_weights_version = -1
    
    def _request_and_wait(self, boards, flats, normal_masks, trade_masks, model_versions):
        result_ref = self.inference_server.inference.remote(
            boards, flats, normal_masks, trade_masks, model_versions
        )
        
        # 結果が返ってくるまで待機
        actions, log_probs, trade_infos = ray.get(result_ref)
        
        return actions, log_probs, trade_infos
    
    def run_loop(self):
        try:
            games_buffer = [] # 送信待ちのtrajectories格納用
            histories_buffer = [] # fileへの書き込み待ちの棋譜格納用

            BUFFER_SIZE = self.config["max_episode_steps"] + 10
            NUM_ENVS = self.config["num_envs"]
            num_opponents = self.config["num_players"] - 1
            num_recipes = self.config["trade_action_dim"] // num_opponents

            board_buf = np.zeros((BUFFER_SIZE, NUM_ENVS, *self.config["input_board_shape"]), dtype=np.int8)
            flat_buf = np.zeros((BUFFER_SIZE, NUM_ENVS, self.config["input_scalar_dim"]), dtype=np.float32)
            masks_buf = np.zeros((BUFFER_SIZE, NUM_ENVS, self.config["action_dim"]), dtype=np.bool_)
            trade_masks_buf = np.zeros((BUFFER_SIZE, NUM_ENVS, num_recipes), dtype=np.bool_)
            act_buf = np.zeros((BUFFER_SIZE, NUM_ENVS), dtype=np.int16)
            log_prob_buf = np.zeros((BUFFER_SIZE, NUM_ENVS), dtype=np.float32)
            player_id_buf = np.zeros((BUFFER_SIZE, NUM_ENVS), dtype=np.int8)
            trade_info_buf = np.zeros((BUFFER_SIZE, NUM_ENVS, self.config["trade_info_dim"]), dtype=np.float32)
            # rewards_buf = np.zeros((BUFFER_SIZE, NUM_ENVS, self.config["num_players"]), dtype=np.float32) spaese 報酬には必要ない

            start_indices = np.zeros(NUM_ENVS, dtype=np.int32)
            current_ptr = 0 # (0 - BUFFER_SIZE)の範囲で循環
            player_models_versions = np.zeros((self.config["num_players"], ), dtype=np.int8) # 各環境の各プレイヤーのモデルバージョン管理用, player_id=0, 1 は必ずself.config["archive_slot_size"] (最新モデル)を指す 4人プレー以外では調整必要
            player_models_versions.fill(self.config["archive_slot_size"])
            if self.config["use_archive"]:
                player_models_versions[2:] = np.random.choice(
                    self.config["archive_slot_size"], 
                    size=self.config["num_players"] - 2, 
                    replace=False
                )


            while True:

                # InferenceServerへ推論リクエストを送信
                if self.config["simple_ppo"]:
                    legal_mask = np.concatenate([
                        self.envs.current_normal_legal_masks[:, :-1], 
                        self.envs.current_trade_legal_masks
                    ], axis=1)
                else:
                    legal_mask = self.envs.current_normal_legal_masks

                actions, log_probs, trade_infos = self._request_and_wait(
                    self.envs.current_boards,
                    self.envs.current_flats,
                    legal_mask,
                    self.envs.current_trade_legal_masks,
                    player_models_versions[self.envs.player_ids]
                )

                # trajectoryに追加
                board_buf[current_ptr] = self.envs.current_boards
                flat_buf[current_ptr] = self.envs.current_flats
                masks_buf[current_ptr] = legal_mask
                if not self.config["simple_ppo"]:
                    trade_masks_buf[current_ptr] = self.envs.current_trade_legal_masks[:, ::num_opponents]
                act_buf[current_ptr] = actions
                log_prob_buf[current_ptr] = log_probs
                player_id_buf[current_ptr] = self.envs.player_ids
                trade_info_buf[current_ptr] = trade_infos

                # 環境を進める
                self.envs.step(actions)

                # 各環境の終了判定
                for env_index in range(self.config["num_envs"]):
                    if self.envs.dones[env_index]:
                        # # エピソード終了時の処理
                        (history_bytes, dict), _positions = self.envs.get_history(env_index)
                        vps, winner_id = self.envs.result(env_index)
                        mean_vp = sum(vps) / len(vps)
                        vp_diff = [vp - mean_vp for vp in vps]
                        winner = winner_id # python側から見たプレイヤーに統一
                        histories_buffer.append(((history_bytes, dict), _positions))
                        # trajectoriesをgames_bufferに移動, あとでwinner情報を付与
                        head = start_indices[env_index]
                        tail = current_ptr
                        traj = {}
                        if head <= tail:
                            indices = slice(head, tail + 1)
                            traj["boards"] = board_buf[indices, env_index]
                            traj["flats"] = flat_buf[indices, env_index]
                            traj["masks"] = masks_buf[indices, env_index]
                            traj["trade_masks"] = trade_masks_buf[indices, env_index]
                            traj["actions"] = act_buf[indices, env_index]
                            traj["log_probs"] = log_prob_buf[indices, env_index]
                            traj["player_ids"] = player_id_buf[indices, env_index]
                            traj["trade_infos"] = trade_info_buf[indices, env_index]
                        else:
                            indices1 = slice(head, BUFFER_SIZE)
                            indices2 = slice(0, tail + 1)
                            traj["boards"] = np.concatenate((
                                board_buf[indices1, env_index],
                                board_buf[indices2, env_index]
                            ), axis=0)
                            traj["flats"] = np.concatenate((
                                flat_buf[indices1, env_index],
                                flat_buf[indices2, env_index]
                            ), axis=0)
                            traj["masks"] = np.concatenate((
                                masks_buf[indices1, env_index],
                                masks_buf[indices2, env_index]
                            ), axis=0)
                            traj["trade_masks"] = np.concatenate((
                                trade_masks_buf[indices1, env_index],
                                trade_masks_buf[indices2, env_index]
                            ), axis=0)
                            traj["actions"] = np.concatenate((
                                act_buf[indices1, env_index],
                                act_buf[indices2, env_index]
                            ), axis=0)
                            traj["log_probs"] = np.concatenate((
                                log_prob_buf[indices1, env_index],
                                log_prob_buf[indices2, env_index]
                            ), axis=0)
                            traj["player_ids"] = np.concatenate((
                                player_id_buf[indices1, env_index],
                                player_id_buf[indices2, env_index]
                            ), axis=0)
                            traj["trade_infos"] = np.concatenate((
                                trade_info_buf[indices1, env_index],
                                trade_info_buf[indices2, env_index]
                            ), axis=0)
                        traj["reward"] = np.array([self.config["vp_reward"] * vp_diff[pos] + (0.75 if pos == winner else -0.75/(self.config["num_players"] - 1)) for pos in range(self.config["num_players"])], dtype=np.float32)
                        games_buffer.append(traj)
                        # print(f"[Actor] Finished episode in env {env_index}, winner: Player {winner}, VPs: {vps}")

                        # 新しいエピソードの準備
                        start_indices[env_index] = (current_ptr + 1) % BUFFER_SIZE
                        self.envs.start(env_index)
                    elif start_indices[env_index] == (current_ptr + 1) % BUFFER_SIZE:
                        # バッファが一周したらリセット
                        # エピソード終了時の処理
                        head = start_indices[env_index]
                        tail = current_ptr
                        traj = {}
                        if head <= tail:
                            indices = slice(head, tail + 1)
                            traj["boards"] = board_buf[indices, env_index]
                            traj["flats"] = flat_buf[indices, env_index]
                            traj["masks"] = masks_buf[indices, env_index]
                            traj["trade_masks"] = trade_masks_buf[indices, env_index]
                            traj["actions"] = act_buf[indices, env_index]
                            traj["log_probs"] = log_prob_buf[indices, env_index]
                            traj["player_ids"] = player_id_buf[indices, env_index]
                            traj["trade_infos"] = trade_info_buf[indices, env_index]
                        else:
                            indices1 = slice(head, BUFFER_SIZE)
                            indices2 = slice(0, tail + 1)
                            traj["boards"] = np.concatenate((
                                board_buf[indices1, env_index],
                                board_buf[indices2, env_index]
                            ), axis=0)
                            traj["flats"] = np.concatenate((
                                flat_buf[indices1, env_index],
                                flat_buf[indices2, env_index]
                            ), axis=0)
                            traj["masks"] = np.concatenate((
                                masks_buf[indices1, env_index],
                                masks_buf[indices2, env_index]
                            ), axis=0)
                            traj["trade_masks"] = np.concatenate((
                                trade_masks_buf[indices1, env_index],
                                trade_masks_buf[indices2, env_index]
                            ), axis=0)
                            traj["actions"] = np.concatenate((
                                act_buf[indices1, env_index],
                                act_buf[indices2, env_index]
                            ), axis=0)
                            traj["log_probs"] = np.concatenate((
                                log_prob_buf[indices1, env_index],
                                log_prob_buf[indices2, env_index]
                            ), axis=0)
                            traj["player_ids"] = np.concatenate((
                                player_id_buf[indices1, env_index],
                                player_id_buf[indices2, env_index]
                            ), axis=0)
                            traj["trade_infos"] = np.concatenate((
                                trade_info_buf[indices1, env_index],
                                trade_info_buf[indices2, env_index]
                            ), axis=0)
                        traj["reward"] = np.array([self.config["draw_reward"] for _ in range(self.config["num_players"])], dtype=np.float32) # 長引きすぎた場合の報酬はconfigで設定
                        games_buffer.append(traj)
                            # 新しいエピソードの準備
                        start_indices[env_index] = (current_ptr + 1) % BUFFER_SIZE
                        self.envs.reset(env_index)
                        (history_bytes, dict), _positions = self.envs.get_history(env_index)
                        histories_buffer.append(((history_bytes, dict), _positions))
            
                # 一定数の棋譜が溜まったらチャンクにまとめてReplayBufferに送信
                while len(games_buffer) >= self.config["send_interval"]:
                    chunk_to_send = games_buffer[:self.config["send_interval"]]
                    games_buffer = games_buffer[self.config["send_interval"]:]
                    batch_to_send = {
                        "boards": np.concatenate([t["boards"] for t in chunk_to_send], axis=0),
                        "flats": np.concatenate([t["flats"] for t in chunk_to_send], axis=0),
                        "actions": np.concatenate([t["actions"] for t in chunk_to_send], axis=0),
                        "masks": np.concatenate([t["masks"] for t in chunk_to_send], axis=0),
                        "trade_masks": np.concatenate([t["trade_masks"] for t in chunk_to_send], axis=0),
                        "log_probs": np.concatenate([t["log_probs"] for t in chunk_to_send], axis=0),
                        "player_ids": np.concatenate([t["player_ids"] for t in chunk_to_send], axis=0),
                        "trade_infos": np.concatenate([t["trade_infos"] for t in chunk_to_send], axis=0),
                        # Rewardはゲーム単位のままスタック
                        "rewards": np.stack([t["reward"] for t in chunk_to_send], axis=0),
                        
                        # 長さ情報も
                        "lengths": np.array([len(t["boards"]) for t in chunk_to_send], dtype=np.int32)
                    }
                    # print("Actor sending batch of size:", len(games_buffer))
                    self.buffer_ref.add.remote(batch_to_send)
                
                # 一定数の棋譜が溜まったらfileに書き込み
                if len(histories_buffer) >= self.config["file_write_interval"]:
                    histories_to_send = histories_buffer[:self.config["file_write_interval"]]
                    histories_buffer = histories_buffer[self.config["file_write_interval"]:]
                    packed = msgpack.packb(histories_to_send)
                    compressed = zstd.ZstdCompressor().compress(packed)
                    file_path = os.path.join(self.data_dir, f"catan_batch_{self.file_count:08d}_{self.id}.msgpack.zst")
                    with open(file_path, "wb") as f:
                        f.write(compressed)
                    self.file_count += 1

                current_ptr = (current_ptr + 1) % BUFFER_SIZE
        except Exception as e:
            print("Actor encountered an exception:", e)
            raise e

@ray.remote
class Evaluator:
    def __init__(self, learner_ref, config):
        self.learner_ref = learner_ref
        self.config = config
        self.device = torch.device(config["evaluator_device"])
        self.use_amp = self.device.type == "cuda"
        self.trade_handler = TradeDecisionHandler(
            self.config, self.device, self.use_amp
        )

        match config["policy_model_class"]:
            case "CatanXDimModel":
                self.PolicyModel = CatanXDimModel
            case "CatanXDimModel_SimplePPO":
                self.PolicyModel = CatanXDimModel_SimplePPO
            case "CatanXDimModel_OmitRPV":
                self.PolicyModel = CatanXDimModel_OmitRPV
            case _:
                # どれにも当てはまらない場合のデフォルト設定
                self.PolicyModel = CatanXDimModel
        
        self.model = self.PolicyModel(config).to(self.device)
        self.trade_model = TradeExpectorNet(config).to(self.device)
        self.opponent_model = self.PolicyModel(config).to(self.device)
        self.opponent_trade_model = TradeExpectorNet(config).to(self.device)
        self.num_eval_envs = 32
        self.envs = MultiVecEnvState(
            num_envs=self.num_eval_envs,
            players=self.config["num_players"],
            trade_activated= not self.config["trade_deactivated"],
            trade_limit=self.config["trade_limit"],
        )
        # ログ設定
        self.step = 0
        self.opponent_ckpt_path = os.path.join(self.config["exp_dir"], "evaluator_opponent.pth")
        self.log_path = os.path.join(self.config["exp_dir"], "eval_log.csv")
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                f.write("Step,WinRate_vs_Archive,AvgPoints_vs_Archive,AvgGameLen,AvgDiceRolls\n")
        else: # resume
            try:
                with open(self.log_path, "r") as f:
                    # 行数を数える (ヘッダーがあるため -1 が現在の完了ステップ数)
                    lines = f.readlines()
                    self.step = max(0, len(lines) - 1)
                print(f"[Evaluator] Resumed log from step {self.step}")
            except Exception as e:
                print(f"[Evaluator] Warning: Failed to parse log file: {e}")
        
        self.last_weights_version = -1
        initial_version, initial_weights_ref = ray.get(self.learner_ref.get_weights.remote())
        initial_weights = ray.get(initial_weights_ref)
        self.model.load_state_dict(initial_weights["policy"])
        self.trade_model.load_state_dict(initial_weights["trade"])
        self.last_weights_version = initial_version
        self.model.eval()
        self.trade_model.eval()

        self.writer = SummaryWriter(log_dir=os.path.join(self.config["exp_dir"], "tensorboard", "evaluator"))

        # Opponentの復元
        if os.path.exists(self.opponent_ckpt_path):
            try:
                checkpoint = torch.load(self.opponent_ckpt_path, map_location=self.device)
                self.opponent_model.load_state_dict(checkpoint["policy"])
                self.opponent_trade_model.load_state_dict(checkpoint["trade"])
                print(f"[Evaluator] Resumed opponent")
            except Exception as e:
                print(f"[Evaluator] Failed to load opponent checkpoint: {e}. Using initial weights.")
                self.opponent_model.load_state_dict(initial_weights["policy"])
                self.opponent_trade_model.load_state_dict(initial_weights["trade"])
                self.opponent_version = initial_version
        else:
            # チェックポイントがなければ初期重みを使用
            self.opponent_model.load_state_dict(initial_weights["policy"])
            self.opponent_trade_model.load_state_dict(initial_weights["trade"])
            self.opponent_version = initial_version
        self.opponent_model.eval()
        self.opponent_trade_model.eval()

        self.log_dir = os.path.join(self.config["exp_dir"], "eval_games")
        os.makedirs(self.log_dir, exist_ok=True)

    def _inference(self, policy_model, trade_model, boards, flats, normal_masks, trade_masks):
        if boards.shape[0] == 0: return None

        # Tensor化
        board_gpu = torch.from_numpy(boards).to(self.device, dtype=torch.float32).permute(0,3,2,1)
        flat_gpu = torch.from_numpy(flats).to(self.device, dtype=torch.float32)

        if self.config["simple_ppo"]:
            normal_masks = np.concatenate([normal_masks[:, :-1], trade_masks], axis=1)
        normal_mask_gpu = torch.from_numpy(normal_masks).to(self.device)
        trade_mask_gpu = torch.from_numpy(trade_masks).to(self.device)
        
        # 安全なマスク値
        min_val = -1e9

        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=torch.bfloat16):
                # Flat Actionなので戻り値は logits, value の2つ想定 + trade_logits を追加
                logits, v_curr, trade_logits = policy_model(board_gpu, flat_gpu)
        
        logits = logits.float()

        # マスク処理
        masked_logits = logits.masked_fill(~normal_mask_gpu.bool(), min_val)
        
        # Argmax
        actions = torch.argmax(masked_logits, dim=-1)
        if not self.config["simple_ppo"]:
            v_curr = v_curr.reshape(-1)

            # trade_moduleの使用
            actions, _ = self.trade_handler.execute_lookahead(
                policy_model=policy_model,
                trade_model=trade_model,
                board_batch=board_gpu,
                flat_batch=flat_gpu,
                original_actions=actions,
                v_curr=v_curr,
                trade_masks=trade_mask_gpu,
                trade_logits=trade_logits,
                fast_mode=False, # 評価時は常に正確モード
            )

        return actions.cpu().numpy().astype(np.int32)
    
    def evaluate_loop(self):
        while True:
            weights_request_future = self.learner_ref.get_weights.remote()
            new_version, new_weights_ref = ray.get(weights_request_future)
            histories = []
            if new_version > self.last_weights_version:
                print(f"[Evaluator] Updating weights to version {new_version}")
                new_weights = ray.get(new_weights_ref)
                self.model.load_state_dict(new_weights["policy"])
                self.trade_model.load_state_dict(new_weights["trade"])
                self.last_weights_version = new_version
                self.model.eval()
                self.trade_model.eval()
            else:
                print(f"[Evaluator] Weights version {new_version} received is not newer. Skipping load.")
                
            wins = 0
            total_games = 0
            total_points = 0.0
            total_length = 0
            total_dice_rolls = 0
            # 2. 複数ゲームを並列で実行
            while total_games < 32:
                p_ids = self.envs.player_ids
                idx_player_mask = np.where(p_ids == 0)[0]
                idx_opponent_mask = np.where(p_ids != 0)[0]
                # プレイヤー0がself.model, その他がopponent_modelで行動選択
                all_actions = np.zeros((self.num_eval_envs,), dtype=np.uint16)
                if len(idx_player_mask) > 0:
                    policy_actions = self._inference(
                        self.model,
                        self.trade_model,
                        self.envs.current_boards[idx_player_mask],
                        self.envs.current_flats[idx_player_mask],
                        self.envs.current_normal_legal_masks[idx_player_mask],
                        self.envs.current_trade_legal_masks[idx_player_mask]
                    )
                    all_actions[idx_player_mask] = policy_actions
                if len(idx_opponent_mask) > 0:
                    opponent_actions = self._inference(
                        self.opponent_model,
                        self.opponent_trade_model,
                        self.envs.current_boards[idx_opponent_mask],
                        self.envs.current_flats[idx_opponent_mask],
                        self.envs.current_normal_legal_masks[idx_opponent_mask],
                        self.envs.current_trade_legal_masks[idx_opponent_mask]
                    )
                    all_actions[idx_opponent_mask] = opponent_actions
                self.envs.step(all_actions)
                for env_index in range(self.num_eval_envs):
                    if self.envs.dones[env_index]:
                        (history_bytes, dict), _positions = self.envs.get_history(env_index)
                        total_dice_rolls += dict.get("total_dice_rolls", 0)
                        histories.append(((history_bytes, dict), _positions))
                        result = self.envs.result(env_index)
                        if result[1] == 0:
                            wins += 1
                        total_points += result[0][0]
                        total_games += 1
                        total_length += self.envs.current_step_counts[env_index]
                        self.envs.start(env_index)
                    elif self.envs.current_step_counts[env_index] >= 3000: # 3000手で強制終了
                        total_games += 1
                        # pointはとりあえず0でカウント
                        total_length += self.envs.current_step_counts[env_index]
                        self.envs.reset(env_index)
                        (history_bytes, dict), _positions = self.envs.get_history(env_index)
                        total_dice_rolls += dict.get("total_dice_rolls", 0)
                        histories.append(((history_bytes, dict), _positions))
                        
                if total_games >= 32:
                    for env_index in range(self.num_eval_envs):
                        self.envs.reset(env_index)
                        _game_data = self.envs.get_history(env_index)
                    break
            
            # 3. ログ書き込み & 待機
            # 30秒おきくらいに評価 total_games=0を防止する
            if total_games > 0:
                win_rate = wins / total_games
                avg_points = total_points / total_games
                avg_length = total_length / total_games
                avg_dice_rolls = total_dice_rolls / total_games
                # --- TensorBoard への書き込み ---
                self.writer.add_scalar("Eval/WinRate_vs_Opponent", win_rate, self.step)
                self.writer.add_scalar("Eval/AvgPoints", avg_points, self.step)
                self.writer.add_scalar("Eval/AvgGameLength", avg_length, self.step)
                self.writer.add_scalar("Eval/AvgDiceRolls", avg_dice_rolls, self.step)
                self.writer.add_scalar("Eval/ModelVersion", self.last_weights_version, self.step)
                self.writer.add_scalar("Eval/OpponentModelVersion", self.opponent_version, self.step)
                with open(self.log_path, "a") as f:
                    f.write(f"{self.step}, {self.last_weights_version}, {self.opponent_version}, {win_rate},{avg_points},{avg_length},{avg_dice_rolls}\n")
                # 棋譜保存 Actorと同じ形式で保存
                packed = msgpack.packb(histories)
                compressed = zstd.ZstdCompressor().compress(packed)
                file_path = os.path.join(self.log_dir, f"eval_batch_{self.step:08d}.msgpack.zst")
                with open(file_path, "wb") as f:
                    f.write(compressed)
            
            self.step += 1
            if self.step % 30 == 0 or self.step == 5: # 5step目には特別にOpponent更新
                # 60分に1回くらいでOpponent更新
                print(f"[Evaluator] Updating opponent model at step {self.step}")
                self.opponent_model.load_state_dict(self.model.state_dict())
                self.opponent_trade_model.load_state_dict(self.trade_model.state_dict())
                torch.save(
                    {
                        "policy": self.opponent_model.state_dict(),
                        "trade": self.opponent_trade_model.state_dict()
                    },
                    self.opponent_ckpt_path
                )
            time.sleep(120)

# @torch.jit.script
def _compute_targets_impl(
    T: int, B: int,
    rewards: torch.Tensor,        # [T, B, num_players]
    values: torch.Tensor,      # [T, B] Target V
    turn_player_ids: torch.Tensor, # [T, B] 手番プレイヤーID
    masks: torch.Tensor,       # [T, B]
    num_players: int,
    reward_gamma: float,
    lam: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    # 結果格納用
    advantages = torch.zeros((T, B), dtype=values.dtype, device=values.device) # Q_targets用
    v_targets = torch.zeros((T, B), dtype=values.dtype, device=values.device) # V_targets用
    
    # 再帰バッファ (t+1 の状態)
    gae = torch.zeros((B, num_players), dtype=values.dtype, device=values.device)
    v_net_next = torch.zeros((B, num_players), dtype=values.dtype, device=values.device)
    acc_reward = torch.zeros((B, num_players), dtype=values.dtype, device=values.device)
    
    for t in range(T - 1, -1, -1):
        mask_t = masks[t].unsqueeze(-1) # [B, 1]
        turn_id = turn_player_ids[t] # [B]

        r_t = rewards[t] # [B, num_players]
        r_t_act = torch.gather(r_t, 1, turn_id.unsqueeze(-1)).reshape(-1) + acc_reward.gather(1, turn_id.unsqueeze(-1)).reshape(-1) # [B]
        
        v_curr = values[t] # [B]
        
        # --- 未来情報の取得 ---
        v_next_act = torch.gather(v_net_next, 1, turn_id.unsqueeze(-1)).reshape(-1) # [B]
        delta = r_t_act + reward_gamma * v_next_act - v_curr
        curr_gae = torch.gather(gae, 1, turn_id.unsqueeze(-1)).reshape(-1) # [B]
        new_gae = delta + reward_gamma * lam * curr_gae
        v_targets[t] = new_gae + v_curr
        advantages[t] = new_gae
        
        # --- 変数更新 ---
        turn_mask = torch.zeros((B, num_players), dtype=torch.bool, device=values.device)
        turn_mask.scatter_(1, turn_id.unsqueeze(-1), torch.ones((B, 1), dtype=torch.bool, device=values.device))
        
        # 手番: V更新
        # v_net_next = torch.where(turn_mask, curr_v.unsqueeze(-1), v_net_next)
        v_net_next = torch.where(turn_mask, v_curr.unsqueeze(-1), v_net_next)
        gae = torch.where(turn_mask, new_gae.unsqueeze(-1), gae)
        v_net_next *= mask_t
        acc_reward += r_t
        # 手番だったプレイヤーのみ0にリセット
        acc_reward = torch.where(turn_mask, torch.zeros((B, num_players), dtype=values.dtype, device=values.device), acc_reward)
        acc_reward *= mask_t

    return v_targets, advantages

_compute_targets_jit = None

def _compute_targets(
    T: int, B: int,
    rewards: torch.Tensor,
    values: torch.Tensor,
    turn_player_ids: torch.Tensor,
    masks: torch.Tensor,
    num_players: int,
    reward_gamma: float,
    lam=0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Wrapper function that uses JIT-compiled version when available."""
    global _compute_targets_jit
    if _compute_targets_jit is None:
        # Lazy JIT compilation on first use
        try:
            _compute_targets_jit = torch.jit.script(_compute_targets_impl)
        except Exception as e:
            print(f"Warning: JIT compilation failed, using non-JIT version: {e}")
            _compute_targets_jit = _compute_targets_impl
    return _compute_targets_jit(T, B, rewards, values, turn_player_ids, masks, num_players, reward_gamma, lam)


@ray.remote
class Learner:
    def __init__(self, buffer_ref, config):
        self.buffer_ref = buffer_ref
        self.config = config
        self.device = torch.device(config["learner_device"])
        self.num_opponents = config["num_players"] - 1

        self.save_dir = os.path.join(self.config["exp_dir"], "checkpoints")
        self.log_path = os.path.join(self.config["exp_dir"], "train_log.csv")
        os.makedirs(self.save_dir, exist_ok=True)
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                f.write("Step,Loss,PolicyLoss,ValueLoss,Entropy\n")
        
        match config["policy_model_class"]:
            case "CatanXDimModel":
                self.PolicyModel = CatanXDimModel
            case "CatanXDimModel_SimplePPO":
                self.PolicyModel = CatanXDimModel_SimplePPO
            case "CatanXDimModel_OmitRPV":
                self.PolicyModel = CatanXDimModel_OmitRPV
            case _:
                # どれにも当てはまらない場合のデフォルト設定
                self.PolicyModel = CatanXDimModel

        self.model = self.PolicyModel(config).to(self.device)
        self.trade_model = TradeExpectorNet(config).to(self.device)
        # self.target_net = PolicyModel(config).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["lr"], weight_decay=self.config["weight_decay"], betas=(self.config["beta_adam_1"], self.config["beta_adam_2"]))
        # とりあえずハイパーパラメータはLearnerと同じに
        self.trade_optimizer = torch.optim.AdamW(self.trade_model.parameters(), lr=self.config["lr"], weight_decay=self.config["weight_decay"])
        self.steps = 0

        if self.config["resume_dir"] is not None:
            self.load_checkpoint(os.path.join(self.save_dir, "latest_checkpoint.pth"))

        # self.target_net.load_state_dict(self.model.state_dict())
        # self.target_net.eval()


    
        self.use_amp = self.device.type == "cuda"
        self.scaler = GradScaler(enabled=False)  # いったん無効化
        self.trade_scaler = GradScaler(enabled=False)  # いったん無効化
        self.gamma_averaging = self.config.get("gamma_averaging", 0.001)
        self.c_clip_gradient = self.config.get("c_clip_gradient", 10000.0)
        self.c_clip_neur_d = self.config.get("c_clip_neur_d", 100.0)

        self.latest_version = 0
        self.latest_weights = {
            "policy": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
            "trade": {k: v.detach().cpu().clone() for k, v in self.trade_model.state_dict().items()}
        }
        self.latest_weights_ref = ray.put(self.latest_weights)

        self.writer = SummaryWriter(log_dir=os.path.join(self.config["exp_dir"], "tensorboard", "learner"))

    
    def save_checkpoint(self, filename):
        path = os.path.join(self.save_dir, filename)
        
        if hasattr(self, "latest_weights"):
            model_state = self.latest_weights["policy"]
            trade_model_state = self.latest_weights["trade"]
        else:
            model_state = self.model.state_dict()
            trade_model_state = self.trade_model.state_dict()
        # 必要な全状態を保存
        state = {
            'steps': self.steps,
            'model_state': model_state,
            'trade_model_state': trade_model_state,
            'optimizer_state': self.optimizer.state_dict(),
            'trade_optimizer_state': self.trade_optimizer.state_dict(),
            # 'target_net_state': self.target_net.state_dict(),
        }
        torch.save(state, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path):
        print(f"Loading checkpoint from {path}...")
        if not os.path.exists(path):
            print(f"Checkpoint not found at {path}. Starting from scratch.")
            return

        checkpoint = torch.load(path, map_location=self.device)
        
        # 全状態を復元
        self.steps = checkpoint['steps']
        self.model.load_state_dict(checkpoint['model_state'])
        self.trade_model.load_state_dict(checkpoint['trade_model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.trade_optimizer.load_state_dict(checkpoint['trade_optimizer_state'])
        # self.target_net.load_state_dict(checkpoint['target_net_state'])
        
        print(f"Successfully resumed from step {self.steps}")

    def _publish_latest(self):
        with torch.no_grad():
            self.latest_weights = {
                "policy": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
                "trade": {k: v.detach().cpu().clone() for k, v in self.trade_model.state_dict().items()}
            }
            self.latest_version += 1
            self.latest_weights_ref = ray.put(self.latest_weights)

    def get_weights(self):
        return (self.latest_version, self.latest_weights_ref)
    
    # def update_target_network(self):
    #     with torch.no_grad():
    #         # for param, target_param in zip(self.model.parameters(), self.target_net.parameters()):
    #             target_param.data.mul_(1.0 - self.gamma_averaging)
    #             target_param.data.add_(self.gamma_averaging * param.data)
    
    def _batched_inference(self, network, boards_gpu, flats_gpu, batch_size=2048):
        logits_list = []
        values_list = []
        trade_logits_list = []
        total = boards_gpu.shape[0]
        network.eval() # batchnorm等をevalモードに
        
        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=torch.bfloat16):
                for i in range(0, total, batch_size):
                    b_chunk = boards_gpu[i:i+batch_size].to(dtype=torch.float32)
                    f_chunk = flats_gpu[i:i+batch_size]
                    l, v, t = network(b_chunk, f_chunk)
                    
                    logits_list.append(l.float())
                    if v is not None:
                        values_list.append(v.float())

                    if t is not None:
                        trade_logits_list.append(t.float())
        
        cat_logits = torch.cat(logits_list, dim=0)
        cat_values = None
        cat_trade_logits = torch.cat(trade_logits_list, dim=0) if trade_logits_list else None
        if len(values_list) > 0:
            cat_values = torch.cat(values_list, dim=0)
        return cat_logits, cat_values, cat_trade_logits
    
    def train_loop(self):
        SUPPOSE_TRADE_ID = self.config["action_dim"] - 1 # 要確認
        DECLINE_ID = self.config["action_dim"] - 2
        ACCEPT_ID = self.config["action_dim"] - 3
        while True:
            # Data Collection
            batch_data = ray.get(self.buffer_ref.sample.remote(self.config["trajectory_batch_size"]))
            if not batch_data:
                time.sleep(1.0)
                continue
            # print("Learner received data batch.")

            # --- Phase 1: Inference & Reg Policy ---
            all_boards = torch.from_numpy(batch_data["boards"]).to(self.device).permute(0,3,2,1)  # (N,W,H,C) -> (N,C,H,W) # 大きいのでint8で保持
            all_flats = torch.from_numpy(batch_data["flats"]).to(self.device, dtype=torch.float32)
            all_log_probs = torch.from_numpy(batch_data["log_probs"]).to(self.device, dtype=torch.float32)
            all_trade_infos = torch.from_numpy(batch_data["trade_infos"]).to(self.device, dtype=torch.float32)

            raw_actions = torch.from_numpy(batch_data["actions"]).to(self.device, dtype=torch.long)
            if self.config["simple_ppo"]:
                all_actions = raw_actions
            else:
                all_actions = torch.where(
                    raw_actions > SUPPOSE_TRADE_ID, 
                    torch.tensor(SUPPOSE_TRADE_ID, device=self.device), 
                    raw_actions
                )
            all_player_ids = torch.from_numpy(batch_data["player_ids"]).to(self.device, dtype=torch.long)

            all_masks = torch.from_numpy(batch_data["masks"]).to(self.device, dtype=torch.bool)
            all_trade_masks = torch.from_numpy(batch_data["trade_masks"]).to(self.device, dtype=torch.bool)
            all_rewards = torch.from_numpy(batch_data["rewards"]).to(self.device, dtype=torch.float32) # [N, T, num_players]
            lengths = batch_data["lengths"]

            # trade_moduleのトレーニング
            if not self.config["simple_ppo"]:
                current_trade_loss = 0.0
                num_trade_updates = 0
                trade_acceptance_rate = 0.0
                with torch.no_grad():
                    is_proposal = (all_actions == SUPPOSE_TRADE_ID)
                    next_actions = torch.roll(all_actions, -1)

                    is_accept = (next_actions == ACCEPT_ID)
                    is_decline = (next_actions == DECLINE_ID)
                    next_is_response = is_accept | is_decline
                    trade_mask = is_proposal & next_is_response
                    trade_mask[-1] = False
                    trade_indices = torch.where(trade_mask)[0]
                    
                    trade_targets_all = is_accept[trade_indices].float().unsqueeze(-1) # (N_trade, 1)
                    num_trade_samples = trade_indices.shape[0]
                if num_trade_samples > 0:
                    trade_acceptance_rate = is_accept[trade_indices].float().mean().item()
                    # 入力データの抽出
                    perm = torch.randperm(num_trade_samples, device=self.device)
                    trade_indices = trade_indices[perm]
                    trade_targets_all = trade_targets_all[perm]
                    self.trade_model.train()
                    
                    for start_idx in range(0, num_trade_samples, self.config["batch_size"]):
                        # ミニバッチの切り出し
                        end_idx = start_idx + self.config["batch_size"]
                        batch_idx = trade_indices[start_idx:end_idx]
                        batch_targets = trade_targets_all[start_idx:end_idx]
                        
                        # 必要なデータだけ抽出
                        b_boards = all_boards[batch_idx].float() # float32へ
                        b_flats = all_flats[batch_idx]
                        b_vecs = all_trade_infos[batch_idx]
                        
                        # 入力結合
                        trade_input = torch.cat([b_flats, b_vecs], dim=1)

                        # --- Update Step ---
                        self.trade_optimizer.zero_grad()
                        
                        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=torch.bfloat16):
                            preds_logit = self.trade_model(b_boards, trade_input)
                            loss = F.binary_cross_entropy_with_logits(preds_logit, batch_targets)
                        
                        self.trade_scaler.scale(loss).backward()
                        self.trade_scaler.step(self.trade_optimizer)
                        self.trade_scaler.update()
                        
                        current_trade_loss += loss.item()
                        num_trade_updates += 1

                    current_trade_loss /= max(1, num_trade_updates)
                self.writer.add_scalar("Learner/TradeLoss", current_trade_loss, self.steps)
                self.writer.add_scalar("Learner/TradeAcceptanceRate", trade_acceptance_rate, self.steps)
                
            # PPOのトレーニング準備

            with torch.no_grad():
                # Current Policy (Pi_theta)
                logits, values, trade_logits = self._batched_inference(self.model, all_boards, all_flats, self.config["batch_size"])
                values = values.reshape(-1)  # [N, 1] -> [N]
                min_val = -1e9
                masked_logits = logits.masked_fill(~all_masks.bool(), min_val)
                curr_log_probs = F.log_softmax(masked_logits, dim=-1) #[N, A]
                
                # Target Network (Value & Pi for V-trace)
                # _, tgt_values = self._batched_inference(self.target_net, all_boards, all_flats, self.config["batch_size"])
                # tgt_values = tgt_values.reshape(-1)  # [N, 1] -> [N]
                
            
            # --- Phase 2: Preparation ---
            batch_size = len(lengths)
            max_len = int(np.max(lengths))
            action_dim = logits.shape[-1]
            
            # プレースホルダー確保 (省略)
            padded_tgt_values = torch.zeros(max_len, batch_size, device=self.device) #[T, B]
            padded_actions = torch.zeros(max_len, batch_size, dtype=torch.long, device=self.device) #[T, B]
            padded_turn_ids = torch.zeros(max_len, batch_size, dtype=torch.long, device=self.device) #[T, B]
            padded_masks = torch.zeros(max_len, batch_size, device=self.device) #[T, B]
            padded_rewards = torch.zeros(max_len, batch_size, self.config["num_players"], device=self.device) #[T, B, num_players]

            # 重要な変数を埋めるループ
            cursor = 0
            for b, L in enumerate(lengths):
                sl = slice(cursor, cursor + L)
                acts = all_actions[sl] #[L]
                padded_actions[:L, b] = acts #[T, B]
                # padded_tgt_values[:L, b] = tgt_values[sl] #[T, B]
                padded_tgt_values[:L, b] = values[sl] #[T, B] オンポリシーに変更

                padded_turn_ids[:L, b] = all_player_ids[sl] #[T, B]
                padded_masks[:L, b] = 1.0 #[T, B]
                padded_rewards[L-1, b] = all_rewards[b] # 最終ステップに報酬をセット #[T, B, num_players]

                cursor += L
            

            v_target_all, advantage_all = _compute_targets(
                T=max_len, B=batch_size,
                rewards=padded_rewards,
                values=padded_tgt_values,
                turn_player_ids=padded_turn_ids,
                masks=padded_masks,
                num_players=self.config["num_players"],
                reward_gamma=self.config["reward_gamma"],
                lam=self.config.get("gae_lambda", 0.95)
            )

            flat_adv_list = []
            flat_v_list = []
            
            # Game-Major (all_boardsの順序) に合わせて切り出す
            for b, L in enumerate(lengths):
                flat_adv_list.append(advantage_all[:L, b])
                flat_v_list.append(v_target_all[:L, b])
            
            # 結合
            flat_adv_targets = torch.cat(flat_adv_list, dim=0)
            flat_v_targets = torch.cat(flat_v_list, dim=0)

            is_learner = (all_player_ids == 0) | (all_player_ids == 1)
            all_boards = all_boards[is_learner]
            all_flats = all_flats[is_learner]
            all_masks = all_masks[is_learner]
            all_actions = all_actions[is_learner]
            all_log_probs = all_log_probs[is_learner] # Bufferから来た過去の確率
            flat_adv_targets = flat_adv_targets[is_learner]
            flat_v_targets = flat_v_targets[is_learner]

            all_raw_actions = raw_actions[is_learner]
            all_trade_masks = all_trade_masks[is_learner]

            self.model.train()

            # adv, v の平均値, 標準偏差をプリント
            if self.steps % 50 == 0:
                print(f"Mean adv-target: {flat_adv_targets.mean().item():.4f}, Mean V-target: {flat_v_targets.mean().item():.4f}")
                print(f"Std adv-target: {flat_adv_targets.std().item():.4f}, Std V-target: {flat_v_targets.std().item():.4f}")
                print(f"Mean V-prediction: {values.mean().item():.4f}, Std V-prediction: {values.std().item():.4f}")
                print(f"Mean actual episode length: {lengths.mean():.2f}, Max: {lengths.max()}, Min: {lengths.min()}")

            acc_total_loss= 0.0
            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_entropy_loss = 0.0
            total_l2_loss = 0.0
            total_trade_distill_loss = 0.0
            num_iterations = 0
            for epoch in range(self.config["ppo_epochs"]):
                num_samples = all_boards.shape[0]
                indices = torch.randperm(num_samples, device=self.device) # GPU上でシャッフル

                # training loop
                for i in range(0, num_samples, self.config["batch_size"]):
                    b_indices = indices[i:i + self.config["batch_size"]]
                    if len(b_indices) < 2:
                        continue
                    b_boards = all_boards[b_indices].to(dtype=torch.float32)
                    b_flats = all_flats[b_indices]
                    b_masks = all_masks[b_indices]
                    b_actions = all_actions[b_indices]
                    b_adv_tgt = flat_adv_targets[b_indices]
                    b_v_tgt = flat_v_targets[b_indices]
                    b_actor_log_probs = all_log_probs[b_indices]

                    b_raw_actions = all_raw_actions[b_indices]
                    b_trade_masks = all_trade_masks[b_indices]

                    self.optimizer.zero_grad()
                    
                    with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=torch.bfloat16):
                        logits, val_pred, trade_logits = self.model(b_boards, b_flats)
                        logits = logits.float()
                        val_pred = val_pred.float()

                        # value_pred_clipped = b_v_tgt + (val_pred - b_v_tgt).clamp(-self.config["epsilon_clip"], self.config["epsilon_clip"])
                        # value_loss_1 = (val_pred - b_v_tgt).pow(2)
                        # value_loss_2 = (value_pred_clipped - b_v_tgt).pow(2)
                        # value_loss = 0.5 * self.config["value_coef"] * torch.max(value_loss_1, value_loss_2).mean()

                        min_val = -1e9
                        masked_logits = logits.masked_fill(~b_masks.bool(), min_val)

                        value_loss = F.mse_loss(val_pred.reshape(-1), b_v_tgt)
                        # value_loss = self.config["value_coef"] * F.huber_loss(val_pred.reshape(-1), b_v_tgt, delta=1.0)
                        
                        log_p_all = F.log_softmax(masked_logits, dim=-1)
                        selected_log_p = log_p_all.gather(1, b_actions.unsqueeze(-1)).reshape(-1)
                        # policy_loss = -self.config["policy_coef"] * (selected_log_p * b_adv_tgt).mean()
                        probs_all = F.softmax(masked_logits, dim=-1)

                        old_log_probs = b_actor_log_probs

                        ratio = torch.exp(selected_log_p - old_log_probs)
                        clipped_ratio = torch.clamp(ratio, 1.0 - self.config["epsilon_clip"], 1.0 + self.config["epsilon_clip"])
                        surrogate1 = ratio * b_adv_tgt
                        surrogate2 = clipped_ratio * b_adv_tgt  
                        policy_loss = -torch.min(surrogate1, surrogate2).mean()

                        # entropy loss (optional)
                        safe_entropy_term = -(probs_all * log_p_all) * b_masks.float()
                        entropy = safe_entropy_term.sum(dim=-1).mean()
                        entropy_loss = -entropy

                        # L2 regularization on logits (optional)
                        l2_loss = (logits ** 2).sum(dim=-1).mean()

                        is_trade = (b_raw_actions >= SUPPOSE_TRADE_ID)
                        if is_trade.any() and not self.config["simple_ppo"]:
                            active_logits = trade_logits[is_trade]
                            active_masks = b_trade_masks[is_trade]
                            active_targets = (b_raw_actions[is_trade] - SUPPOSE_TRADE_ID) // self.num_opponents
                            masked_trade_logits = active_logits.masked_fill(~active_masks.bool(), min_val)

                            loss_trade_distill = F.cross_entropy(masked_trade_logits, active_targets)
                        else:
                            loss_trade_distill = 0.0

                        total_loss = self.config["policy_coef"] * policy_loss + self.config["value_coef"] * value_loss + self.config["entropy_coef"] * entropy_loss + self.config["L2_coef"] * l2_loss + self.config["trade_distill_coef"] * loss_trade_distill

                        total_policy_loss += policy_loss.item()
                        total_value_loss += value_loss.item()
                        total_entropy_loss += entropy_loss.item()
                        total_l2_loss += l2_loss.item()
                        total_trade_distill_loss += (loss_trade_distill.item() if isinstance(loss_trade_distill, torch.Tensor) else 0.0)

                        acc_total_loss += total_loss.item()
                        num_iterations += 1
                    
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.config["c_clip_gradient"])
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["c_norm_clip_gradient"])
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    # self.update_target_network()
            
            avg_total_loss = acc_total_loss / max(1, num_iterations)
            avg_policy_loss = total_policy_loss / max(1, num_iterations)
            avg_value_loss = total_value_loss / max(1, num_iterations)
            avg_entropy_loss = -total_entropy_loss / max(1, num_iterations)
            avg_l2_loss = total_l2_loss / max(1, num_iterations)
            avg_trade_distill_loss = total_trade_distill_loss / max(1, num_iterations)

            with open(self.log_path, "a") as f:
                f.write(f"{self.steps},{avg_total_loss},{avg_policy_loss},{avg_value_loss},{avg_entropy_loss}\n")
            # TensorBoardログ
            self.writer.add_scalar("Loss/Total", avg_total_loss, self.steps)
            self.writer.add_scalar("Loss/Policy", avg_policy_loss, self.steps)
            self.writer.add_scalar("Loss/Value", avg_value_loss, self.steps)
            self.writer.add_scalar("Loss/Entropy", avg_entropy_loss, self.steps)
            self.writer.add_scalar("Loss/L2", avg_l2_loss, self.steps)
            self.writer.add_scalar("Loss/TradeDistill", avg_trade_distill_loss, self.steps)
            self.writer.add_scalar("Train/MeanEpisodeLength", lengths.mean(), self.steps)
            self.writer.add_scalar("Train/MeanVTarget", flat_v_targets.mean().item(), self.steps)
            self.writer.add_scalar("Train/MeanAdvantage", flat_adv_targets.mean().item(), self.steps)

            self._publish_latest()
            print(f"Loss at step {self.steps}: Total {avg_total_loss:.4f}, Policy {avg_policy_loss:.4f}, Value {avg_value_loss:.4f}")
            self.steps += 1
            
            # チェックポイント保存
            if self.steps % 10 == 0:
                self.save_checkpoint("latest_checkpoint.pth")
            if self.steps % 500 == 0:
                self.save_checkpoint(f"checkpoint_{self.steps:08d}.pth")
            print(f"Learner completed step {self.steps}.")
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Catan AI Training Script")
    parser.add_argument("--num_players", type=int, default=4, help="プレイヤー数")
    parser.add_argument("--experiment_root", type=str, default="./experiments", help="実験データ保存ルート")
    parser.add_argument("--experiment_name", type=str, default=None, help="実験名 (指定なければ日時)")
    parser.add_argument("--resume_dir", type=str, default=None, help="再開する実験ディレクトリのパス")
    parser.add_argument("--gpu_actor", type=float, default=0.0)
    parser.add_argument("--gpu_inference_server", type=float, default=1.0)
    parser.add_argument("--gpu_learner", type=float, default=1.0)
    parser.add_argument("--gpu_evaluator", type=float, default=0.0)
    parser.add_argument("--num_actors", type=int, default=50, help="Actor数")
    parser.add_argument("--num_envs", type=int, default=128, help="Actor並列環境数")
    # ablation 用設定
    parser.add_argument("--trade_deactivated", action="store_true", help="取引を無効化")
    parser.add_argument("--use_archive", action="store_true", help="InferenceServerでアーカイブを使用")
    parser.add_argument("--simple_ppo", action="store_true", help="平坦なPPOモデルを使用")
    parser.add_argument("--omit_rpv", action="store_true", help="RPV(資源選好ベクトル)を省略")

    parser.add_argument("--archive_size", type=int, default=16, help="InferenceServerアーカイブモデル数")
    parser.add_argument("--archive_slot_size", type=int, default=16, help="InferenceServerアーカイブスロット数")
    parser.add_argument("--slot_update_interval", type=int, default=20000, help="InferenceServerスロット更新間隔(step数)")
    parser.add_argument("--archive_update_interval", type=int, default=40000, help="InferenceServerアーカイブ更新間隔(step数)")
    parser.add_argument("--inference_batch_size", type=int, default=16, help="InferenceServerバッチサイズ(Actor数)")
    parser.add_argument("--send_interval", type=int, default=16, help="Actor送信頻度(棋譜数)")
    parser.add_argument("--load_interval", type=int, default=10, help="Actor重み更新頻度(step数)")
    parser.add_argument("--max_episode_steps", type=int, default=1000, help="1エピソード最大ステップ数")
    parser.add_argument("--vp_reward", type=float, default=0.02, help="勝利点報酬係数")
    parser.add_argument("--draw_reward", type=float, default=-1/3, help="引き分け報酬")
    parser.add_argument("--reward_gamma", type=float, default=1.0, help="報酬割引率")
    parser.add_argument("--epsilon", type=float, default=0.0, help="ε-greedyのε値")
    parser.add_argument("--buffer_capacity", type=int, default=10, help="ReplayBuffer容量(チャンク数)")
    parser.add_argument("--trajectory_batch_size", type=int, default=4, help="Learner収集単位(chunk数)")
    parser.add_argument("--batch_size", type=int, default=1024, help="SGDミニバッチサイズ")
    parser.add_argument("--file_write_interval", type=int, default=2000, help="Actorファイル書き込み頻度(棋譜数)")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="エントロピー正則化係数")
    parser.add_argument("--L2_coef", type=float, default=1e-4, help="L2正則化係数")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=2.0, help="NeurDのクリッピング閾値")
    parser.add_argument("--value_coef", type=float, default=0.5, help="Value lossの重み")
    parser.add_argument("--policy_coef", type=float, default=1.0, help="Policy lossの重み")
    parser.add_argument("--gamma_averaging", type=float, default=0.001)
    parser.add_argument("--beta_adam_1", type=float, default=0.0)
    parser.add_argument("--beta_adam_2", type=float, default=0.999)
    parser.add_argument("--c_norm_clip_gradient", type=float, default=1.0)
    parser.add_argument("--c_clip_gradient", type=float, default=10000.0)
    parser.add_argument("--c_clip_neur_d", type=float, default=100.0)
    parser.add_argument("--epsilon_clip", type=float, default=0.2)
    parser.add_argument("--ppo_epochs", type=int, default=1, help="PPOエポック数")
    parser.add_argument("--trade_limit", type=int, default=1, help="取引制限回数")
    parser.add_argument("--num_xdim_blocks", type=int, default=8, help="PolicyModelのブロック数")
    parser.add_argument("--lookahead_batch_size", type=int, default=4096, help="Actor先読みバッチサイズ")
    parser.add_argument("--trade_pruning_k", type=int, default=20, help="取引プルーニングのk値")
    parser.add_argument("--trade_distill_coef", type=float, default=0.0025, help="取引蒸留損失の重み")

    args = parser.parse_args()
    
    # dummy 環境を作成してboard_shape, action_dimを取得
    fmt = pycatan.PyObservationFormat()
    opponents = args.num_players - 1
    dummy_env = pycatan.SingleEnvironment.new(
        format=fmt,
        opponents=opponents,
        trade_activated=True,
    )
    player_id, board, flat, normal_legal_mask, trade_legal_mask, position, done = dummy_env.start()
    dummy_env.close()
    args.input_board_shape = board.shape
    args.input_scalar_dim = flat.shape[0]
    args.action_dim = normal_legal_mask.shape[0]
    args.trade_action_dim = trade_legal_mask.shape[0]

    args.policy_model_class = "CatanXDimModel"

    if args.simple_ppo:
        args.policy_model_class = "CatanXDimModel_SimplePPO"
        args.action_dim = (args.action_dim - 1) + args.trade_action_dim # 取引アクションを全て平坦化して末尾に追加する, 「取引を提案」のアクション分を除いている点に注意
        args.trade_action_dim = 0 # 取引モジュールは使用しない
    
    if args.omit_rpv:
        args.policy_model_class = "CatanXDimModel_OmitRPV"

    if args.resume_dir:
        # Resumeモード
        EXP_DIR = os.path.abspath(args.resume_dir)
        if not os.path.exists(EXP_DIR):
            raise FileNotFoundError(f"Resume directory {EXP_DIR} does not exist.")
        
        print(f"Resuming experiment from: {EXP_DIR}")
        # 設定の復元（オプション）
        args_path = os.path.join(EXP_DIR, "args.json")
        if os.path.exists(args_path):
            args = load_args(args, args_path)
            print("Loaded arguments from args.json")
    else:
        # 新規実験モード
        if args.experiment_name is not None:
            exp_id = args.experiment_name
        else:
            exp_id = f"exp_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        EXP_DIR = os.path.join(os.path.abspath(args.experiment_root), exp_id)
        os.makedirs(EXP_DIR, exist_ok=True)
        print(f"Starting new experiment: {EXP_DIR}")
        args.exp_dir = EXP_DIR
        save_args(args, os.path.join(EXP_DIR, "args.json"))
    

    actor_device = "cuda" if args.gpu_actor > 0 else "cpu"
    learner_device = "cuda" if args.gpu_learner > 0 else "cpu"
    evaluator_device = "cuda" if args.gpu_evaluator > 0 else "cpu"
    inference_server_device = "cuda" if args.gpu_inference_server > 0 else "cpu"
    actor_num_gpus = args.gpu_actor if args.gpu_actor > 0 else 0
    learner_num_gpus = args.gpu_learner if args.gpu_learner > 0 else 0
    evaluator_num_gpus = args.gpu_evaluator if args.gpu_evaluator > 0 else 0
    inference_server_num_gpus = args.gpu_inference_server if args.gpu_inference_server > 0 else 0

    args.actor_device = actor_device
    args.learner_device = learner_device
    args.evaluator_device = evaluator_device
    args.inference_device = inference_server_device
    config = vars(args)
    config["board_logits_mask"] = get_board_logits_mask(args.num_players)
    config["trade_info_dim"] = 5 + (args.num_players - 1)
    RAY_TEMP_DIR = os.path.join(os.path.expanduser("~"), "ray_temp") 
    os.makedirs(RAY_TEMP_DIR, exist_ok=True)

    ray.init(_temp_dir=RAY_TEMP_DIR)

    buffer = ReplayBuffer.remote(capacity=config["buffer_capacity"])
    learner = Learner.options(num_gpus=learner_num_gpus, max_concurrency=2).remote(buffer, config)
    inference_server = InferenceServer.options(num_gpus=inference_server_num_gpus, max_concurrency=1000).remote(learner, config)
    ray.get(inference_server.initialize_weights.remote())
    actors = [
        Actor.options(num_cpus=0.5, num_gpus=0).remote(i, buffer, learner, inference_server, config)
        for i in range(args.num_actors)
    ]
    evaluator = Evaluator.options(num_gpus=evaluator_num_gpus).remote(learner, config)

    inference_server.run_loop.remote()
    learner_loop_ref = learner.train_loop.remote()
    evaluator.evaluate_loop.remote()
    for actor in actors:
        actor.run_loop.remote()
        time.sleep(3)  # スタートを少しずらす
    ray.get(learner_loop_ref)
