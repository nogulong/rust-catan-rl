import subprocess
import time
import sys
import os
import argparse
from concurrent.futures import ProcessPoolExecutor
from pycatan.jsettlers import start_server

def run_worker(worker_id, port, args, gpu_id=None):
    """1つのワーカー（実験ランナー）を実行"""
    
    # 実験IDをワーカーごとにユニークにする
    worker_exp_id = f"{args.id}_w{worker_id}"
    
    # run_experiment_1vs3.py をサブプロセスとして実行
    cmd = [
        sys.executable, "run_experiment_1vs3.py",
        "--id", worker_exp_id,
        "--games", str(args.games_per_worker),
        "--port", str(port),
        "--result_dir", args.result_dir
    ]
    
    if args.model:
        cmd.extend(["--model", args.model])
        # cmd.extend(["--blocks", str(args.blocks)])
    if args.trade:
        cmd.append("--trade")
    if args.is_fast:
        cmd.append("--is_fast")
        
    # 環境変数の設定
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    elif args.cpu:
        env["CUDA_VISIBLE_DEVICES"] = "" # Force CPU
        
    # 実行
    subprocess.run(cmd, env=env)
    # print(f"✅ Worker {worker_id} finished.")

def main():
    parser = argparse.ArgumentParser(description='Parallel JSettlers Experiment Runner')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--start_port', type=int, default=8880, help='Starting port number')
    parser.add_argument('--gpus', type=str, default=None, help='Comma separated list of GPU IDs to use (e.g. "0,1")')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    
    # run_experiment_1vs3.py に渡す引数
    parser.add_argument('--model', default=None, help='Path to model file')
    parser.add_argument('--id', required=True, help='Experiment ID prefix')
    parser.add_argument('--games', type=int, default=100, help='Total number of games')
    parser.add_argument('--trade', action='store_true', help='Enable trade')
    parser.add_argument('--is_fast', action='store_true', help='Use fast bots')
    parser.add_argument('--result_dir', default='results', help='Result directory')
    
    args = parser.parse_args()
    
    # 1ワーカーあたりのゲーム数
    args.games_per_worker = args.games // args.workers
    if args.games_per_worker == 0:
        args.games_per_worker = 1
        args.workers = args.games # ゲーム数よりワーカーが多い場合は調整
        
    print(f"=== Parallel Experiment: {args.games} games, {args.workers} workers, {args.games_per_worker} games/worker, Ports {args.start_port}-{args.start_port + args.workers - 1}")
    
    gpu_list = []
    if args.gpus:
        gpu_list = [int(x) for x in args.gpus.split(',')]
        print(f"Using GPUs: {gpu_list}")
    elif args.cpu:
        print("Using CPU only")
    else:
        print("Using default device configuration (likely GPU 0 if available)")
    
    servers = []
    
    try:
        # 1. サーバー群を起動
        for i in range(args.workers):
            port = args.start_port + i
            server_proc = start_server(port)
            servers.append(server_proc)
        print("⏳ Waiting for servers to initialize (5s)...")
        time.sleep(5)
        # 2. ワーカー群を並列実行
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = []
            for i in range(args.workers):
                port = args.start_port + i
                # GPU割り当て
                gpu_id = None
                if gpu_list:
                    gpu_id = gpu_list[i % len(gpu_list)]
                futures.append(executor.submit(run_worker, i, port, args, gpu_id))
            # 全完了待ち
            for f in futures:
                f.result()
        # 3. 各workerのjsonl/jsonをまとめる
        import glob, json
        all_results = []
        # jsonl優先、なければjson
        pattern_jsonl = os.path.join(args.result_dir, f"results_{args.id}_w*.jsonl")
        pattern_json = os.path.join(args.result_dir, f"results_{args.id}_w*.json")
        files = glob.glob(pattern_jsonl)
        if not files:
            files = glob.glob(pattern_json)
        for file in files:
            with open(file, "r") as f:
                if file.endswith(".jsonl"):
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                all_results.append(json.loads(line))
                            except Exception as e:
                                print(f"⚠️  Failed to parse line in {file}: {e}")
                else:
                    try:
                        data = json.load(f)
                        if isinstance(data, list):
                            all_results.extend(data)
                        else:
                            all_results.append(data)
                    except Exception as e:
                        print(f"⚠️  Failed to parse {file}: {e}")
        # 保存
        merged_file = os.path.join(args.result_dir, f"results_{args.id}_all.json")
        with open(merged_file, "w") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"✅ Merged all worker results into {merged_file}")
        # 元ファイルを削除
        for file in files:
            try:
                os.remove(file)
                print(f"🗑️  Deleted {file}")
            except Exception as e:
                print(f"⚠️  Failed to delete {file}: {e}")
    except KeyboardInterrupt:
        print("\n👋 Interrupted. Shutting down...")
    finally:
        print("🛑 Stopping servers...")
        for p in servers:
            p.terminate()
            try:
                p.wait(timeout=1)
            except:
                p.kill()
        print("Done.")

if __name__ == "__main__":
    main()
