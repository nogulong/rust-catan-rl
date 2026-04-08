"""
JSettlersサーバーに接続するPythonボット
"""
import random
import string
import socket
import traceback
from pycatan import PyTricellState
import torch
import numpy as np
from typing import Optional
import subprocess
import time
import json

from .jsettler_utils import write_java_utf, read_java_utf, parse_message, parse_board_layout_1084


class JSettlersBot:
    """
    JSettlersサーバーに接続するPythonボット
    """
    
    def __init__(self, host: str, port: int, nickname: str, cookie: str, agent, device=None, create=False):
        """
        Args:
            host: JSettlersサーバーのホスト
            port: ポート番号
            nickname: ボットの名前
            cookie: サーバーのセキュリティクッキー
            agent: PyTorchエージェント（predict_action メソッドを持つ）
        """
        self.host = host
        self.port = port
        self.nickname = nickname
        self.cookie = cookie
        self.agent = agent
        
        self.sock: Optional[socket.socket] = None
        self.player_count = 4  # 通常4人用, 3人用とかへの対応は後で
        self.game_state = PyTricellState.new(players=self.player_count)
        self.current_game: Optional[str] = None
        self.game_started = False
        self.device = device if device is not None else torch.device("cpu")
        self.is_creator = create
    
    def create_game(self, game_name):
        print(f"🛠️ Creating new game: {game_name}")
        
        # 存在しない名前で 1013 を送ると、サーバーが新規作成してくれる
        # 形式: 1013 | ニックネーム | - | - | ゲーム名
        msg = f"1013|{self.nickname},-,-,{game_name}"
        write_java_utf(self.sock, msg)
        print(f"→ {msg}")
        
        # current_game をセットしておく（サーバーからの承認待ち）
        self.current_game = game_name

    def reset_game(self, game_name):
        """ゲームをリセットする"""
        print(f"🔄 Resetting game: {game_name}")
        # 1073|GAME_NAME (RESETBOARDREQUEST)
        msg = f"1073|{game_name}"
        write_java_utf(self.sock, msg)
        print(f"→ {msg}")

        
    def connect(self):
        """サーバーに接続"""
        # print(f"🤖 Connecting to {self.host}:{self.port}...")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        # print("✓ Connected")
        
    def authenticate(self):
        """ロボットとして認証"""
        # print("🔐 Authenticating as robot...")
        
        # VERSIONメッセージを送信
        version_msg = "9998|2700,2.7.00,JM20251205,;6pl;sb;,en_US"
        write_java_utf(self.sock, version_msg)
        # print(f"→ {version_msg}")
        
        # IMAROBOTメッセージを送信
        robot_msg = f"1022|{self.nickname},{self.cookie},python.bot.PyTorchAgent"
        write_java_utf(self.sock, robot_msg)
        # print(f"→ {robot_msg}")
        # print("✓ Authenticated")
        
    def run(self, game_name, mode="create", seat_num=0, num_games=100):
        """メインループ"""
        
        # 設定を保存しておく
        games_played = 0
        self.requested_seat_num = seat_num # 座席番号を保存
        self.is_creator = (mode == "create")
        self.seated_players = set()

        if not self.sock:
            self.connect()
            self.authenticate()
        try:
            current_game_name = f"{game_name}"
            # self.reset_internal_state() # いらないかも
            if mode == "create":
                self.create_game(current_game_name)

            elif mode == "join":
                self.current_game = current_game_name
                self.join_game(current_game_name)
        
            while True:
                # メッセージを受信
                message = read_java_utf(self.sock)
                print(f"← {message}")
                
                # メッセージを処理
                if self.handle_message(message):
                    print("🏁 Game Over. Exiting loop.")
                    break
            
            games_played += 1
                    
        except KeyboardInterrupt:
            print("\n👋 Shutting down...")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # このあたりでcreatorに勝敗報告とかを送らせる
            if self.sock:
                self.sock.close()

    def run_interactive(self):
        """標準入力からのコマンドで制御するモード"""
        import select
        import sys
        
        if not self.sock:
            self.connect()
            self.authenticate()
        
        print("READY") # 親プロセスへの合図
        sys.stdout.flush()
        
        try:
            while True:
                # ソケットと標準入力を監視
                inputs = [self.sock, sys.stdin]
                readable, _, _ = select.select(inputs, [], [])
                
                for r in readable:
                    if r is self.sock:
                        message = read_java_utf(self.sock)
                        if not message:
                            print("❌ Server disconnected")
                            return
                        # print(f"[RECV] {message}") 
                        if self.handle_message(message):
                            # Game Over
                            pass
                        sys.stdout.flush()
                            
                    elif r is sys.stdin:
                        line = sys.stdin.readline()
                        if not line:
                            return # EOF
                        line = line.strip()
                        if not line:
                            continue
                        self.handle_command(line)
        except KeyboardInterrupt:
            print("\n👋 Shutting down...")
        finally:
            if self.sock:
                self.sock.close()

    def handle_command(self, command_line):
        parts = command_line.split()
        if not parts:
            return
        cmd = parts[0].upper()

        if cmd in ["JOIN", "CREATE"]:
            print(f"♻️ Reconnecting for new game command: {cmd}")
            if self.sock:
                try:
                    self.sock.close()
                except:
                    pass
                self.sock = None
            
            # 再接続と再認証
            self.connect()
            self.authenticate()
            # 内部状態もクリア
            self.game_state = PyTricellState.new(players=self.player_count)
            self.seated_players = set()
            self.player_id = -1
            self.game_started = False
        
        if cmd == "JOIN":
            # JOIN game_name seat_num
            if len(parts) < 3:
                print("Usage: JOIN <game_name> <seat_num>")
                return
            game_name = parts[1]
            seat_num = int(parts[2])
            self.requested_seat_num = seat_num
            self.current_game = game_name
            
            # 内部状態のリセット
            self.game_state = PyTricellState.new(players=self.player_count)
            self.seated_players = set()
            self.player_id = -1
            self.game_started = False

            
            self.join_game(game_name)
            
        elif cmd == "ADDBOT":
            # ADDBOT game_name type seat
            if len(parts) < 4:
                print("Usage: ADDBOT <game_name> <type> <seat>")
                return
            game_name = parts[1]
            bot_type = parts[2] # FAST or SMART
            seat = parts[3]
            
            # Send *ADDBOT command as chat
            # 1010|GAME\0NICKNAME\0TEXT
            # TEXT = *ADDBOT TYPE SEAT
            text = f"*ADDBOT {bot_type} {seat}"
            msg = f"1010|{game_name}\x00-\x00{text}"
            write_java_utf(self.sock, msg)
            print(f"→ {msg}")

        elif cmd == "CREATE":
            # CREATE game_name seat_num
            if len(parts) < 3:
                print("Usage: CREATE <game_name> <seat_num>")
                return
            game_name = parts[1]
            seat_num = int(parts[2])
            self.requested_seat_num = seat_num
            self.current_game = game_name
            
            # 内部状態のリセット
            self.game_state = PyTricellState.new(players=self.player_count)
            self.seated_players = set()
            self.player_id = -1
            self.game_started = False


            self.create_game(game_name)

        elif cmd == "RESET":
            # RESET game_name
            if len(parts) < 2:
                print("Usage: RESET <game_name>")
                return
            game_name = parts[1]
            self.reset_game(game_name)
            
        elif cmd == "EXIT":
            import sys
            sys.exit(0)

    
    def handle_message(self, message: str):
        """メッセージを処理"""
        parsed = parse_message(message)
        msg_type = parsed["type"]
        
        # メッセージタイプに応じて処理
        if msg_type == "1071":
            print("✓ Robot parameters updated")

        elif msg_type == "1009": # PutPiece
            # 何かの駒を置いた通知 1009|testplay,3,0,201
            args = parsed.get("args", [])
            if len(args) >= 4:
                p_num = int(args[1])
                piece_type = int(args[2])
                coord = int(args[3])
                self.game_state.put_piece(p_num, piece_type, coord)

        
        elif msg_type == "1012":
            # ログ表示や自分の着席確認だけ行い、リトライ処理は削除してOK
            args = parsed.get("args", [])
            if len(args) >= 3:
                name = args[1]
                seat = int(args[2])

                self.seated_players.add(seat)
                print(f"🪑 Player {name} sat at seat {seat}. Total: {len(self.seated_players)}/4")
                
                if name == self.nickname: # または "-" から変換された名前
                    self.game_state.set_player_id(seat)
                    self.player_id = seat
                    print(f"✅ Success! I am Player {seat}!")
                
                if len(self.seated_players) == 4 and self.is_creator:
                     print("🚀 All players seated. Starting game...")
                     self.start_game()
        
        elif msg_type == "1013": # JOINGAME 誰かがゲームに入室した
            args = parsed.get("args", [])
            if len(args) >= 1:
                name = args[0]
                
                # 自分が入室した通知が来たら、1回だけ座るリクエストを送る
                if name == self.nickname and self.player_id == -1:
                    print("🚀 Join complete. Requesting auto-seat assignment...")
                    # ここで1回だけ呼ぶ！
                    self.sit_down(self.current_game)# とりあえず,常に0を指定
        
        elif msg_type == "1014": # BOARDLAYOUT (Classic)
            # 1014|gameName,hl[0]..hl[36],nl[0]..nl[36],rh
            args = parsed.get("args", [])
            if len(args) >= 76: # 1 + 37 + 37 + 1
                hl = [int(x) for x in args[1:38]]
                nl_mapped = [int(x) for x in args[38:75]]
                rh = int(args[75])
                
                # Map dice numbers (0-9 -> 2-12)
                # -1 -> 0 (Water/Desert)
                dice_map = {0:2, 1:3, 2:4, 3:5, 4:6, 5:8, 6:9, 7:10, 8:11, 9:12}
                nl = [dice_map.get(x, 0) for x in nl_mapped]
                
                self.game_state.update_board_layout(hl, nl, rh)

        elif msg_type == "1018": # GAMESTART
            self.game_started = True
                
        elif msg_type == "1021": # JOINGAMEAUTH
            # ゲーム参加成功
            self.current_game = parsed.get("game")
            self.player_id = int(parsed.get("playerNumber", -1))
            print(f"✓ Joined game: {self.current_game} as player {self.player_id}")

        # elif msg_type == "1023": #BOTJOINGAMEREQUEST
        #     # ゲーム参加要求
        #     game_name = parsed.get("game")
        #     if game_name:
        #         self.join_game(game_name)
        
        elif msg_type == "1024": # PLAYERELEMENTS　発展カードの使用関連
            # ← 1024|testplay,1,100,19,1
            # ← 1024|testplay,1,101,15,1
            # ← 1024|testplay,2,100,4,0,Y　4(麦)を0にセット
            # ← 1063|testplay,2,6
            # ← 1024|testplay,3,100,4,0,Y　monopolyでの増減もある
            # ← 1063|testplay,3,3

            args = parsed.get("args", [])
            if len(args) >= 5:
                p_num = int(args[1])
                if p_num == -1:
                    return
                action = int(args[2])
                element = int(args[3])
                amount = int(args[4])
                self.game_state.update_player_elements(p_num, action, element, amount)
        
        elif msg_type == "1025": # GAMESTATE, pycatanでいうphase変更
            args = parsed.get("args", [])
            if len(args) >= 2:
                phase = int(args[1])
                if phase == 0 or phase == 51 or phase == 50:
                    return # pregame, choose victim, discarding は無視
                if phase == 1000:
        
                    return
                self.game_state.update_phase(phase)
 
            if self.is_my_turn() and self.game_started:
                print("🚀 It's my turn! (From GAMESTATE msg)")
                self.make_decision()
            
        elif msg_type == "1026": # TURN
            # 形式: 1026 | ゲーム名 | プレイヤー番号 | アクションID(ボード状態)
            args = parsed.get("args", [])
            
            if len(args) >= 2:
                try:
                    next_player = int(args[1])
                    phase = int(args[2])
                    # args[2]はうえのphaseと同じ
                    # 同様のphase変更処理が必要
                    
                    print(f"🔄 Turn changed to Player {next_player}")
                    self.game_state.update_player_turn(next_player)
                    self.game_state.update_phase(phase)

                    # もし自分の番なら行動開始！
                    if self.is_my_turn() and self.game_started:
                        print("🚀 It's my turn! (From TURN msg)")
                        self.make_decision()
                        
                except ValueError:
                    print(f"⚠️ Failed to parse turn player number: {args}")
        
        elif msg_type == "1028": # DiceRollResult
            pass # サイコロの出目通知, 資源変動は後続の1092で処理
        
        elif msg_type == "1029":
            # カードを捨てる要求にこたえる
            self.game_state.set_discard_phase()
            self.make_decision()


        elif msg_type == "1030":
            # サイコロを振る要求, 来てないかも
            pass

        elif msg_type == "1033": # DISCARD
            # カードを捨てた結果通知
            # 1033|testplay,p1,0,0,0,0,0,4
            args = parsed.get("args", [])
            idx = 1
            player_num = -1

            if args[idx].startswith('p'):
                player_num = int(args[idx][1:])
                idx += 1
            discarded = [int(x) for x in args[idx:idx+6]] # 6番目はジョーカー
            self.game_state.execute_discard(player_num, discarded)

        elif msg_type == "1034": # MT
            # player と hex, 1034|testplay,1,183
            args = parsed.get("args", [])
            if len(args) >= 3:
                p_num = int(args[1])
                hex_coord = int(args[2])
                self.game_state.put_piece(p_num, 3, hex_coord)
        
        elif msg_type == "1036": # ChooseVictim
            self.choose_victim()

        elif msg_type == "1038": # trade終わり, 1038 → 1042の場合はアクションをとる必要がある
            self.game_state.update_phase(20) # main turnに戻す
            # 次のメッセージを受け取って, 1042かつ自分のターンならmake_decisionを呼ぶ, 1042以外は普通にhandle_messageで処理
            message = read_java_utf(self.sock)
            print(f"← {message}")
            parsed = parse_message(message)
            msg_type = parsed["type"]
            if msg_type == "1042":
                if self.is_my_turn() and self.game_started:
                    print("🚀 It's my turn! (After trade end)")
                    self.make_decision()
            else:
                self.handle_message(message)

        elif msg_type == "1039": # Accept
            #受諾者, 提案者, 資源オファー, 資源リクエスト offerとrequestが逆かも, 要確認
            args = parsed.get("args", [])
            if len(args) >= 13:
                accepter = int(args[1])
                proposer = int(args[2])
                offer = [int(x) for x in args[3:8]]
                request = [int(x) for x in args[8:13]]
                self.game_state.execute_trade(accepter, proposer, offer, request, is_bank=False)
        
        elif msg_type == "1040": # BankTrade 1040|testplay,4,0,0,0,0,0,0,0,0,1,2
            # 最後プレイヤー
            args = parsed.get("args", [])
            if len(args) >= 12:
                offer = [int(x) for x in args[1:6]]
                request = [int(x) for x in args[6:11]]
                p_num = int(args[11])
                self.game_state.execute_trade(0, p_num, offer, request, is_bank=True)
                if p_num == self.player_id:
                    self.make_decision() # 銀行トレード後の判断を行う
        
        elif msg_type == "1041":
            # トレード提案を受信
            # 提案者, 提案相手(bool), 資源オファー, 資源リクエスト
            # 1041|testplay,3,false,false,true,false,0,1,0,0,0,1,0,0,0,0
            args = parsed.get("args", [])
            if args[2 + self.player_id] == "true": # 自分宛の提案なら
                offer = [int(x) for x in args[2 + self.player_count: 7 + self.player_count]]
                request = [int(x) for x in args[7 + self.player_count: 12 + self.player_count]]
                
                # ログ出力
                try:
                    hand = self.game_state.get_hand(self.player_id)
                    self._log_trade({
                        "type": "RECEIVE",
                        "offer": offer,
                        "request": request,
                        "proposer": int(args[1]),
                        "hand": hand
                    })
                except Exception as e:
                    print(f"Error logging trade receive: {e}")

                self.game_state.set_trade_offer(int(args[1]), offer, request)
                self.make_decision() # トレード受諾/拒否の判断を行う
                self.game_state.update_phase(20) # main turnに戻す

        elif msg_type == "1042":
            self.game_state.update_phase(20) # main turnに戻す

        elif msg_type == "1046": 
            # BuyDevelopの結果　1046|testplay,3,0,0　最後がどのカードを引いたか
            # useの場合, 1046|testplay,1,1,3 1番が3(Monopoly)をuse(1)した
            args = parsed.get("args", [])
            if len(args) >= 4:
                p_num = int(args[1])
                action = int(args[2]) # 0=buy, 1=use
                element = int(args[3])
                if p_num == -1 or action == 4:
                    return
                self.game_state.update_develop_card(p_num, action, element)
        
        elif msg_type == "1035": # いったん保留
            # プレイヤーを選択（盗賊で奪う相手）
            # self.handle_choose_player(parsed)
            pass

        elif msg_type == "1052": # PICKRESOURCESRESULT
            # 1052|GameName, Brick, Ore, Wool, Grain, Lumber, PlayerNum, cause
            args = parsed.get("args", [])
            if len(args) >= 8:
                res_changes = [int(x) for x in args[1:6]]
                player = int(args[6])
                
                self.game_state.update_resources(player, 101, res_changes) # 101 = add

        elif msg_type == "1061":# 勝敗などの記録に使える, いったん保留
            args = parsed.get("args", [])
            if len(args) >= 5:
                if args[1].startswith('t'): # 無視してよい
                    return
                try:
                    scores = [int(x) for x in args[1:5]]
                    print(f"📊 Game Stats: {scores}")
                    
                    # Check for winner (>= 10 points)
                    for i, score in enumerate(scores):
                        if score >= 10:
                            print(f"🏆 WINNER DETECTED: Player {i} with {score} points")
                            print(f"RESULT: WINNER {i}") # Machine readable
                            # ここでLEAVEGAME(1011)を送信
                            if self.current_game:
                                msg = f"1011|{self.current_game}"
                                write_java_utf(self.sock, msg)
                                print(f"→ {msg} (LEAVEGAME)")
                            return True # Signal game over
                except ValueError:
                    pass

        elif msg_type == "1072": # ROLLDICEPROMPT
            # 多分早くダイスを振れと言われてる
            pass

        elif msg_type == "1084": # BOARDLAYOUT2
        # 1084|testplay,1,HL,[37,9,6,69,6,6,5,3,2,67,8,4,5,4,2,6,6,4,1,1,4,3,84,8,1,3,2,5,6,6,3,0,5,12,18,6,97,6,RH,149,NL,[37,-1,-1,-1,-1,-1,12,11,4,-1,-1,9,6,3,8,-1,-1,10,5,11,10,5,-1,-1,8,4,9,2,-1,-1,3,-1,6,-1,-1,-1,-1,-1
            layout = parse_board_layout_1084(message)
            hl = layout.get("HL", [])
            nl = layout.get("NL", [])
            rh = layout.get("RH", -1)
            # hl と nl の長さは一致している
            self.game_state.update_board_layout(hl, nl, rh)

        elif msg_type == "1086": # リソース使用関係 1086|testplay|3|102|1|1|5|1
            args = parsed.get("args", [])
            player = int(args[1])
            action_type = int(args[2])
            length = (len(args) - 3) // 2
            res_changes = [0] * 5
            for i in range(length):
                res_type = int(args[3 + i * 2])
                if res_type == 0 or res_type > 5:
                    continue
                amount = int(args[4 + i * 2])
                res_changes[res_type-1] += amount
            self.game_state.update_resources(player, action_type, res_changes)

        elif msg_type == "1090": # 発展カード枚数の更新　1090|testplay,3,1,24,0, 1046のbuyのところで反映するのでいらない
            pass
            
        elif msg_type == "1092": # DICERESULTRESOURCES　1092|testplay|2|2|8|1|1|0|3|3|1|2
            # 獲得人数, (playerid, player所有数, (prod, resType)*gain_res)*gain_player
            args = parsed.get("args", [])
            player_count = int(args[1])
            idx = 2
            for _ in range(player_count):
                p_num = int(args[idx])
                owned_count = int(args[idx + 1])
                idx += 2
                res_changes = [0] * 5
                while idx < len(args) and args[idx] != "0":
                    prod = int(args[idx])
                    res_type = int(args[idx + 1])
                    res_changes[res_type-1] += prod
                    idx += 2
                self.game_state.update_resources(p_num, 101, res_changes)
                idx += 1 # skip the trailing 0

        elif msg_type == "1102": # ROBBERYRESULT, 1102|testplay,1,3,R,6,1,T
            args = parsed.get("args", [])
            player = int(args[1])
            victim = int(args[2])
            res_type = int(args[4])
            self.game_state.execute_robbery(player, victim, res_type)
            
            
    def join_game(self, game_name: str):
        """ゲームに参加"""
        print(f"📥 Joining game: {game_name}")
        join_msg = f"1013|{self.nickname},-,-,{game_name}"
        write_java_utf(self.sock, join_msg)
        print(f"→ {join_msg}")
    
    def sit_down(self, game_name: str):
        """席に座るリクエスト (1012)"""
        seat = getattr(self, 'requested_seat_num', 0)
        print(f"🪑 Requesting to sit in game: {game_name} at seat {seat}")
        
        msg = f"1012|{game_name},-,{seat},true"
        
        write_java_utf(self.sock, msg)
        print(f"→ {msg}")

    def start_game(self):
        """ゲームを開始する (1018)"""
        # 1018|gameName,0
        msg = f"1018|{self.current_game},0"
        write_java_utf(self.sock, msg)
        print(f"→ {msg}")
    
    def is_my_turn(self) -> bool:
        """自分のターンかどうか"""
        return self.game_state.is_my_turn()
    def make_decision(self):
        """エージェントで決定を行い、アクションを実行"""
        try:
            # ゲーム状態をObservationに変換
            board, flat, legal_mask, trade_mask = self.game_state.to_observation()
            
            # エージェントでアクションを予測
            action = self.agent.act(board, flat, legal_mask, trade_mask)
            
            print(f"🧠 Agent predicted action: {action}")
            
            # アクションを実行
            self.execute_action(action)
            
        except Exception as e:
            print(f"⚠️  Error in decision making: {e}")
            traceback.print_exc()
            # フォールバック: ターンを終了
            self.end_turn()
    
    def execute_action(self, action: int):
        """
        アクションを実行
        
        Args:
            action: エージェントが出力したアクション
        
        注意: アクション空間は実際のエージェントに合わせてカスタマイズしてください
        """
        action_type, param0, param1, res0, res1 = self.game_state.decode_action(action)
        if not self.current_game:
            return
        
        # 例: 簡単なアクションマッピング
        # 実際のエージェントのアクション空間に合わせて変更してください
        
        if action_type == 0:
            # ターンを終了
            self.end_turn()
        elif action_type == 1:
            self.roll_dice()
        elif action_type == 2:
            self.move_thief(param0, param1) # param0: hex coord, param1: victim player number
        elif action_type == 3:
            self.build_road(param0) # param0: road coord
        elif action_type == 4:
            self.build_settlement(param0) # param0: settlement coord
        elif action_type == 5:
            self.build_city(param0) # param0: city coord
        elif action_type == 6:
            self.trade_bank(res0, res1) # res0: offered resource type, res1: requested resource type
        elif action_type == 7:
            self.buy_development_card()
        elif action_type == 8:
            self.development_knight()
        elif action_type == 9:
            self.development_road()
        elif action_type == 10:
            self.development_year_of_plenty(param0, param1)
        elif action_type == 11:
            self.development_monopoly(param0) # param0: resource type
        elif action_type == 12:
            self.discard_cards(res0)
        elif action_type == 13:
            pass # exit
        elif action_type == 14:
            self.offer_trade(param0, res0, res1) # param0: target player number, res0: offered resource type, res1: requested resource type
        elif action_type == 15:
            self.accept_trade(param0)
        elif action_type == 16:
            self.decline_trade()
        else:
            # デフォルト: ターンを終了
            self.end_turn()
    
    def roll_dice(self):
        """サイコロを振る"""
        msg = f"1031|{self.current_game}"
        write_java_utf(self.sock, msg)
        print(f"→ {msg}")
    
    def end_turn(self):
        """ターンを終了"""
        msg = f"1032|{self.current_game}"
        write_java_utf(self.sock, msg)
        print(f"→ {msg} (ending turn)")
    
    def move_thief(self, hex_coord: int, victim_player: int):
        """盗賊を移動"""
        msg = f"1034|{self.current_game},{self.player_id},{hex_coord}"
        write_java_utf(self.sock, msg)
        print(f"→ {msg}")
        self.victim_player = victim_player  # 次に盗む相手を保存しておく

    def choose_victim(self):
        """盗む相手を選ぶ"""
        msg = f"1035|{self.current_game},{self.victim_player}"
        write_java_utf(self.sock, msg)
        print(f"→ {msg} (choosing victim)")
            
    
    def build_road(self, coord: int):
        """道を建設"""
        msg = f"1009|{self.current_game},{self.player_id},0,{coord}"
        write_java_utf(self.sock, msg)
        print(f"→ {msg}")
        self.game_state.set_development_played() # もし道建設カードによるものなら, phaseを変える

    
    def build_settlement(self, coord: int):
        """集落を建設"""
        msg = f"1009|{self.current_game},{self.player_id},1,{coord}"
        write_java_utf(self.sock, msg)
        print(f"→ {msg}")
    
    def build_city(self, coord: int):
        """都市を建設"""
        msg = f"1009|{self.current_game},{self.player_id},2,{coord}"
        write_java_utf(self.sock, msg)
        print(f"→ {msg}")
    
    def trade_bank(self, offer_res, request_res):
        """銀行とトレード"""
        offer_str = f"{offer_res[0]},{offer_res[1]},{offer_res[2]},{offer_res[3]},{offer_res[4]}"
        request_str = f"{request_res[0]},{request_res[1]},{request_res[2]},{request_res[3]},{request_res[4]}"
        msg = f"1040|{self.current_game},{offer_str},{request_str}"
        write_java_utf(self.sock, msg)
        print(f"→ {msg}")

        rate = sum(offer_res)
        print(f"[BANK_TRADE] 1:{rate}")

    def buy_development_card(self):
        """発展カードを購入"""
        """購入リクエストを送る"""
        msg = f"1045|{self.current_game}"
        write_java_utf(self.sock, msg)
        print(f"→ {msg}")
    
    def development_knight(self):
        """騎士カードを使用"""
        msg = f"1049|{self.current_game},9"
        write_java_utf(self.sock, msg)
        print(f"→ {msg}")
        while True:
            message = read_java_utf(self.sock)
            print(f"← {message}")
            parsed = parse_message(message)
            msg_type = parsed["type"]
            if msg_type == "1038" or msg_type == "1042":
                continue
            self.handle_message(message)
            if msg_type == "1046":
                break
    
    def development_road(self):
        """道建設カードを使用"""
        msg = f"1049|{self.current_game},1"
        write_java_utf(self.sock, msg)
        print(f"→ {msg}")
        while True:
            message = read_java_utf(self.sock)
            print(f"← {message}")
            parsed = parse_message(message)
            msg_type = parsed["type"]
            if msg_type == "1038" or msg_type == "1042":
                continue
            self.handle_message(message)
            if msg_type == "1046":
                break

    
    def development_year_of_plenty(self, res_type1, res_type2):
        """豊穣の年カードを使用"""
        msg = f"1049|{self.current_game},2"
        write_java_utf(self.sock, msg)
        print(f"→ {msg}")
        """GAMESTATEを受け取る"""
        while True:
            message = read_java_utf(self.sock)
            print(f"← {message}")
            parsed = parse_message(message)
            msg_type = parsed["type"]
            if msg_type == "1025": 
                args = parsed.get("args", [])
                if int(args[1]) == 52:
                    break
            if msg_type == "1038" or msg_type == "1042":
                continue
            self.handle_message(message)
            
        """次に, 資源指定を送る"""
        res_list = [0, 0, 0, 0, 0]
        res_list[res_type1 - 1] += 1
        res_list[res_type2 - 1] += 1
        res_str = f"{res_list[0]},{res_list[1]},{res_list[2]},{res_list[3]},{res_list[4]}"
        msg = f"1052|{self.current_game},{res_str}"
        write_java_utf(self.sock, msg)
        print(f"→ {msg}")

    def development_monopoly(self, res_type):
        """独占カードを使用"""
        msg = f"1049|{self.current_game},3"
        write_java_utf(self.sock, msg)
        print(f"→ {msg}")
        """GAMESTATEを受け取る"""
        while True:
            message = read_java_utf(self.sock)
            print(f"← {message}")
            parsed = parse_message(message)
            msg_type = parsed["type"]
            if msg_type == "1025": 
                args = parsed.get("args", [])
                if int(args[1]) == 53:
                    break
            if msg_type == "1038" or msg_type == "1042":
                continue
            self.handle_message(message)
        
        """次に, 資源指定を送る"""
        msg = f"1053|{self.current_game},{res_type}"
        write_java_utf(self.sock, msg)
        print(f"→ {msg}")
    
    def discard_cards(self, resources):
        """カードを捨てる"""
        # DISCARD -> 1033
        res_str = f"{resources[0]},{resources[1]},{resources[2]},{resources[3]},{resources[4]},0"
        msg = f"1033|{self.current_game},{res_str}"
        write_java_utf(self.sock, msg)
        print(f"→ {msg}")
        print(f"   Discarding: {resources}")

    def _log_trade(self, data):
        print(f"[TRADE_LOG] {json.dumps(data)}")
    
    def offer_trade(self, target_player: int, offer_res, request_res):
        """トレードを提案 (1041)"""
        self.game_state.increment_trade_count()
        
        # 【重要】NumPy型をPython標準のint型に変換する
        # list() だけでは不十分で、中身を int() する必要があります
        safe_offer = [int(x) for x in offer_res[:5]]
        safe_request = [int(x) for x in request_res[:5]]
        
        # 文字列作成 (safe_offerを使う)
        offer_str = ",".join(map(str, safe_offer))
        request_str = ",".join(map(str, safe_request))
        
        # Toフラグの作成
        to_flags = []
        my_num = self.player_id
        
        for i in range(self.player_count):
            if i == my_num:
                to_flags.append("false") # 自分には送らない
            elif target_player == -1 or target_player == i:
                to_flags.append("true")
            else:
                to_flags.append("false")
                
        to_str = ",".join(to_flags)
        
        msg = f"1041|{self.current_game},{my_num},{to_str},{offer_str},{request_str}"
        write_java_utf(self.sock, msg)
        print(f"→ {msg}")

        # 手札情報の取得と安全な変換
        try:
            raw_hand = self.game_state.get_hand(self.player_id)
            safe_hand = [int(x) for x in raw_hand] if raw_hand is not None else []
        except:
            safe_hand = []

        # ログ用データの構築 (全て変換済みのsafe変数を使う)
        trade_log_data = {
            "type": "OFFER",
            "result": "FAILED",
            "offer": safe_offer,
            "request": safe_request,
            "target_player": int(target_player), # これも念のためint化
            "hand": safe_hand
        }

        """返事が来るまで待つ"""
        while True:
            message = read_java_utf(self.sock)
            print(f"← {message}")
            parsed = parse_message(message)
            msg_type = parsed["type"]
            
            if msg_type == "1041": # counter offer
                args = parsed.get("args", [])
                if args[2 + self.player_id] == "false": # 自分のオファーがブロードキャストされて戻ってきただけなら無視して待機
                    continue
                
                # カウンターオファーが来た＝自分の提案は実質失敗（あるいは交渉継続）
                trade_log_data["result"] = "COUNTERED" 
                self._log_trade(trade_log_data)
                
                self.handle_message(message)
                args = parsed.get("args", [])
                return

            elif msg_type == "1037" or msg_type == "1038" or msg_type == "1039":
                # 手札が変わっている可能性があるので再取得してもいいが、
                # オファー時点の手札を知りたいなら trade_log_data['hand'] はそのままでOK
                
                if msg_type == "1039":
                    trade_log_data["result"] = "ACCEPTED"
                else:
                    trade_log_data["result"] = "DECLINED"
                
                # ここでログ出力！
                self._log_trade(trade_log_data)

                self.handle_message(message)
                self.game_state.update_phase(20) # main turnに戻す
                self.make_decision() # トレード後の判断を行う
                return
            
    
    def accept_trade(self, proposer: int):
        """トレードを受け入れる"""
        self._log_trade({"type": "RECEIVE_RESPONSE", "action": "ACCEPT"})
        # ACCEPTOFFER -> 1036
        msg = f"1039|{self.current_game},{self.player_id},{proposer}"
        write_java_utf(self.sock, msg)
        print(f"→ {msg}")
    
    def decline_trade(self):
        """トレードを断る"""
        self._log_trade({"type": "RECEIVE_RESPONSE", "action": "DECLINE"})
        # REJECTOFFER -> 1037
        msg = f"1037|{self.current_game},{self.player_id}"
        write_java_utf(self.sock, msg)
        print(f"→ {msg}")
    
    # def handle_choose_player(self, params: dict):
    #     """プレイヤーを選択（盗賊で奪う相手）"""
    #     choices = params.get("choices", "")
    #     print(f"👤 Choosing player to rob from: {choices}")
        
    #     # 選択肢がある場合は最初のプレイヤーを選択
    #     if choices:
    #         choice_list = [int(x) for x in choices.split(",") if x.strip().isdigit()]
    #         if choice_list:
    #             chosen = choice_list[0]
    #             self.choose_player(chosen)
    #         else:
    #             # 選択肢がない場合は-1（誰も奪わない）
    #             self.choose_player(-1)
    #     else:
    #         self.choose_player(-1)
    
    # def choose_player(self, player_number: int):
    #     """プレイヤーを選択"""
    #     # CHOOSEPLAYER -> 1035
    #     msg = f"1035|game={self.current_game}|choice={player_number}"
    #     write_java_utf(self.sock, msg)
        # print(f"→ {msg}")
