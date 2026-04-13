import subprocess
import time
import sys
import os
import random
import json
import datetime
import argparse

# 設定
HOST = "localhost"
PORT = 8880

def main():
    parser = argparse.ArgumentParser(description='JSettlers Tournament Runner')
    parser.add_argument('--main_model', default=None, help='Path to the model file(.onnx) for MyBot_1')
    parser.add_argument('--trade_model', default=None, help='Path to the trade model file(.onnx) for MyBot_1')
    parser.add_argument('--id', required=True, help='Experiment ID (used for output filename)')
    parser.add_argument('--games', type=int, default=100, help='Number of games to play')
    parser.add_argument('--trade', action='store_true', help='Enable trade actions in the agents')
    parser.add_argument('--is_fast', action='store_true', help='Use fast Java bots instead of smart ones')
    parser.add_argument('--result_dir', default='results', help='Directory to save result logs')
    parser.add_argument('--port', type=int, default=8880, help='Port number of JSettlers server')
    args = parser.parse_args()
    
    global PORT
    PORT = args.port
    
    num_games = args.games
    experiment_id = args.id
    main_model_path = args.main_model
    trade_model_path = args.trade_model

    # 出力ファイル名: results_<experiment_id>.jsonl
    os.makedirs(args.result_dir, exist_ok=True)
    log_file = os.path.join(args.result_dir, f"results_{experiment_id}.jsonl")
    
    # 既存ファイルチェック
    if os.path.exists(log_file):
        print(f"⚠️  Warning: Output file '{log_file}' already exists. Appending results.")
    
    # 統計用
    bot_prefix = "droid" if args.is_fast else "robot"
    stats = {
        "MyBot_1": {"wins": 0, "total_score": 0},
        f"{bot_prefix}_1": {"wins": 0, "total_score": 0},
        f"{bot_prefix}_2": {"wins": 0, "total_score": 0},
        f"{bot_prefix}_3": {"wins": 0, "total_score": 0},
        "TIMEOUT": {"wins": 0, "total_score": 0},
        "Unknown": {"wins": 0, "total_score": 0}
    }
    
    # ゲーム名を固定 (短くする)
    base_game_name = f"Exp{int(time.time()) % 10000}"
    
    print(f"Starting Tournament: {num_games} games")
    print(f"Experiment ID: {experiment_id}")
    print(f"Game Name: {base_game_name}")
    print(f"Main Model: {main_model_path}")
    print(f"Trade Model: {trade_model_path}")
    print(f"Logging to {log_file}")

    # MyBotプロセスを起動 (Interactive Mode)
    print("🚀 Launching MyBot process...")
    cmd_mybot = [
        sys.executable, "-m", "pycatan.jsettlers.run_bot",
        HOST, str(PORT), "MyBot_1", "cookie",
        "--interactive", "--create",
    ]
    if main_model_path:
        cmd_mybot.extend(["--main_model", main_model_path])
    if trade_model_path:
        cmd_mybot.extend(["--trade_model", trade_model_path])

        
    if args.trade:
        cmd_mybot.append("--enable_trade")
        
    # stdin=PIPE, stdout=PIPE で起動
    mybot_proc = subprocess.Popen(
        cmd_mybot, 
        stdin=subprocess.PIPE, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        bufsize=1
    )
    
    # MyBotの起動完了待ち ("READY" を待つ)
    while True:
        line = mybot_proc.stdout.readline()
        if not line:
            print("❌ MyBot failed to start.")
            return
        print(f"[MyBot Init] {line.strip()}")
        if "READY" in line:
            break

    try:
        with open(log_file, "a") as f:
            for i in range(num_games):
                match_id = i + 1
                game_name = f"{base_game_name}_M{match_id}"
                print(f"=== Starting Match {match_id}: {game_name} ===")
                
                # # 1. ゲームのリセット (初回以外)
                # if i > 0:
                #     print("  Resetting game...")
                #     # MyBotにリセットさせる
                #     reset_cmd = f"RESET {game_name}\n"
                #     print(f"[SEND] {reset_cmd.strip()}")
                #     mybot_proc.stdin.write(reset_cmd)
                #     mybot_proc.stdin.flush()
                #     time.sleep(1.0) # リセット完了待ち

                # 2. 席の決定
                mybot_seat = (match_id - 1) % 4
                seat_assignments = {
                    "MyBot_1": mybot_seat,
                    f"{bot_prefix}_1": (mybot_seat + 1) % 4,
                    f"{bot_prefix}_2": (mybot_seat + 2) % 4,
                    f"{bot_prefix}_3": (mybot_seat + 3) % 4
                }
                seat_to_player = {v: k for k, v in seat_assignments.items()}
                print(f"  Seating: {seat_assignments}")

                # 3. MyBotに参加指示 (CREATE or JOIN)
                create_cmd = f"CREATE {game_name} {mybot_seat}\n"
                mybot_proc.stdin.write(create_cmd)
                mybot_proc.stdin.flush()

                # Wait for MyBot to be seated
                print("  Waiting for MyBot to be seated...")
                while True:
                    line = mybot_proc.stdout.readline()
                    if not line:
                        print("❌ MyBot process died while waiting for MyBot to seat.")
                        return
                    if "Player MyBot_1 sat at seat" in line and "Total: 1/4" in line:
                        print("  MyBot seated. Now inviting bots...")
                        break

                # 4. Java Bot 起動 (サーバー内部のボットを使用)
                # ADDBOT <game_name> <type> <seat>
                bot_type = "FAST" if args.is_fast else "SMART"
                
                # Bot 1
                seat1 = seat_assignments[f"{bot_prefix}_1"]
                addbot1_cmd = f"ADDBOT {game_name} {bot_type} {seat1}\n"
                mybot_proc.stdin.write(addbot1_cmd)
                mybot_proc.stdin.flush()
                # Wait for bot to be seated
                while True:
                    line = mybot_proc.stdout.readline()
                    if not line:
                        print("❌ MyBot process died while waiting for bot 1.")
                        return
                    if f"Player {bot_prefix}" in line and "sat at seat" in line:
                        break

                # Bot 2
                seat2 = seat_assignments[f"{bot_prefix}_2"]
                addbot2_cmd = f"ADDBOT {game_name} {bot_type} {seat2}\n"
                mybot_proc.stdin.write(addbot2_cmd)
                mybot_proc.stdin.flush()
                # Wait for bot to be seated
                while True:
                    line = mybot_proc.stdout.readline()
                    if not line:
                        print("❌ MyBot process died while waiting for bot 2.")
                        return
                    if f"Player {bot_prefix}" in line and "sat at seat" in line:
                        break

                # Bot 3
                seat3 = seat_assignments[f"{bot_prefix}_3"]
                addbot3_cmd = f"ADDBOT {game_name} {bot_type} {seat3}\n"
                mybot_proc.stdin.write(addbot3_cmd)
                mybot_proc.stdin.flush()
                # Wait for bot to be seated
                while True:
                    line = mybot_proc.stdout.readline()
                    if not line:
                        print("❌ MyBot process died while waiting for bot 3.")
                        return
                    if f"Player {bot_prefix}" in line and "sat at seat" in line:
                        break

                print("  Bots invited via ADDBOT command. Waiting for result...")

                # 5. 結果監視
                match_result = {
                    "match_id": match_id,
                    "game_name": game_name,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "seat_assignments": seat_assignments,
                    "start_player": seat_to_player[0],
                    "winner": None,
                    "winner_seat": -1,
                    "scores": {},
                    "model": model_path_1,
                    "trade_logs": [],
                    "bank_trades": [0, 0, 0] # [1:4, 1:3, 1:2]
                }

                start_time = time.time()
                while True:
                    if time.time() - start_time > 600: # 10分タイムアウト
                        print("  Timeout!")
                        match_result["winner"] = "TIMEOUT"
                        break
                        
                    # MyBotの出力を監視
                    line = mybot_proc.stdout.readline()
                    if not line:
                        if mybot_proc.poll() is not None:
                            print("❌ MyBot process died!")
                            return
                        continue
                    
                    # ログ出力制御（必要なら）
                    if "[Time]" in line:
                        print(line.strip())
                    
                    if "[TRADE_LOG]" in line:
                        try:
                            log_data = json.loads(line.split("[TRADE_LOG]")[1].strip())
                            match_result["trade_logs"].append(log_data)
                        except Exception as e:
                            print(f"Error parsing trade log: {e}")

                    if "[BANK_TRADE]" in line: # [BANK_TRADE] 1:4 など
                        try:
                            trade_type = line.split("[BANK_TRADE]")[1].strip()
                            if trade_type == "1:4":
                                match_result["bank_trades"][0] += 1
                            elif trade_type == "1:3":
                                match_result["bank_trades"][1] += 1
                            elif trade_type == "1:2":
                                match_result["bank_trades"][2] += 1
                        except Exception as e:
                            print(f"Error parsing bank trade log: {e}")

                    if "Game Stats:" in line:
                        try:
                            score_str = line.split("Game Stats:")[1].strip()
                            scores = json.loads(score_str)
                            for seat, score in enumerate(scores):
                                match_result["scores"][seat_to_player.get(seat, f"Seat{seat}")] = score
                        except:
                            pass

                    if "RESULT: WINNER" in line:
                        try:
                            parts = line.strip().split()
                            winner_seat = int(parts[2])
                            winner_name = seat_to_player.get(winner_seat, "Unknown")
                            
                            print(f"  🏆 Winner is Seat {winner_seat} ({winner_name})")
                            match_result["winner"] = winner_name
                            match_result["winner_seat"] = winner_seat
                        except:
                            print(f"  Parse Error: {line}")
                        break
                
                # 6. Java Bot 終了 (内部ボットなのでプロセス終了不要)
                # for p in java_bots:
                #     if p.poll() is None:
                #         p.terminate()
                #         try:
                #             p.wait(timeout=1)
                #         except subprocess.TimeoutExpired:
                #             p.kill()

                # 結果保存
                f.write(json.dumps(match_result) + "\n")
                f.flush()
                
                # 統計更新
                winner = match_result["winner"]
                if winner in stats:
                    stats[winner]["wins"] += 1
                else:
                    stats["Unknown"]["wins"] += 1
                    
                for player, score in match_result["scores"].items():
                    if player in stats:
                        stats[player]["total_score"] += score

                # 中間報告
                print(f"--- Stats after {i+1} games ---")
                for name, data in stats.items():
                    if data["wins"] > 0 or data["total_score"] > 0:
                        avg_score = data["total_score"] / (i + 1)
                        win_rate = (data["wins"] / (i + 1)) * 100
                        print(f"  {name}: {data['wins']} wins ({win_rate:.1f}%), Avg Score: {avg_score:.1f}")
                print("-------------------------------")
                
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n👋 Interrupted by user.")
    finally:
        print("🛑 Terminating MyBot...")
        if mybot_proc.poll() is None:
            mybot_proc.terminate()
            mybot_proc.wait()

if __name__ == "__main__":
    main()
