import sys
import subprocess
import random
from importlib import resources
from .server import start_server, open_gui


def play_match():
    port = 8880
    pkg_root = resources.files("pycatan")
    wait_time = 1.0 # before selecting an action, wait for this seconds, to make the GUI more visible.

    # サーバー起動
    server_process = start_server(port)

    # ゲームを作るBotを起動

    cmd_bot = [
        sys.executable, "-m", "pycatan.jsettlers.run_bot",
        "localhost", str(port), "PycatanBot_1", "cookie",
        "--interactive", "--create", "--enable_trade", "--wait_time", str(wait_time)
    ]
        
    # stdin=PIPE, stdout=PIPE で起動
    bot_proc = subprocess.Popen(
        cmd_bot, 
        stdin=subprocess.PIPE, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        bufsize=1
    )
    
    # PycatanBot_1の起動完了待ち ("READY" を待つ)
    while True:
        line = bot_proc.stdout.readline()
        if not line:
            print("❌ PycatanBot_1 failed to start.")
            return
        
        if "READY" in line:
            break

    cmd_bot2 = [
        sys.executable, "-m", "pycatan.jsettlers.run_bot",
        "localhost", str(port), "PycatanBot_2", "cookie",
        "--interactive", "--enable_trade", "--wait_time", str(wait_time)
    ]

    bot2_proc = subprocess.Popen(
        cmd_bot2,
        stdin=subprocess.PIPE, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        bufsize=1
    )

    while True:
        line = bot2_proc.stdout.readline()
        if not line:
            print("❌ PycatanBot_2 failed to start.")
            return
        
        if "READY" in line:
            break

    cmd_bot3 = [
        sys.executable, "-m", "pycatan.jsettlers.run_bot",
        "localhost", str(port), "PycatanBot_3", "cookie",
        "--interactive", "--enable_trade", "--wait_time", str(wait_time)
    ]

    bot3_proc = subprocess.Popen(
        cmd_bot3,
        stdin=subprocess.PIPE, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        bufsize=1
    )

    while True:
        line = bot3_proc.stdout.readline()
        if not line:
            print("❌ PycatanBot_3 failed to start.")
            return
        
        if "READY" in line:
            break

    game_name = "pycatan_game"
    seats = [0, 1, 2, 3]
    random.shuffle(seats) # 席をシャッフル

    # PycatanBot_1がゲームを作成
    bot_proc.stdin.write(f"CREATE {game_name} {seats[0]}\n")
    bot_proc.stdin.flush()

    while True:
        line = bot_proc.stdout.readline()
        if not line:
            print("❌ PycatanBot_1 process died while waiting for it to seat.")
            return
        if "Player PycatanBot_1 sat at seat" in line and "Total: 1/4" in line:
            break

    # PycatanBot_2がゲームに参加
    bot2_proc.stdin.write(f"JOIN {game_name} {seats[1]}\n")
    bot2_proc.stdin.flush()

    while True:
        line = bot2_proc.stdout.readline()
        if not line:
            print("❌ PycatanBot_2 process died while waiting for it to seat.")
            return
        if "Player PycatanBot_2 sat at seat" in line and "Total: 2/4" in line:
            break

    # PycatanBot_3がゲームに参加
    bot3_proc.stdin.write(f"JOIN {game_name} {seats[2]}\n")
    bot3_proc.stdin.flush()

    while True:
        line = bot3_proc.stdout.readline()
        if not line:
            print("❌ PycatanBot_3 process died while waiting for it to seat.")
            return
        if "Player PycatanBot_3 sat at seat" in line and "Total: 3/4" in line:
            break
    
    # GUIを開く
    gui_proc = open_gui(port)

    try:
        # GUIプロセスが終了するまでPythonを終了させない
        gui_proc.wait() 
    except KeyboardInterrupt:
        print("\n👋 Detected Ctrl+C. Shutting down...")
    finally:
        # プロセスの後片付け
        for p in [bot_proc, bot2_proc, bot3_proc, server_process, gui_proc]:
            if p: p.terminate()

if __name__ == "__main__":
    play_match()