import argparse
from pycatan import PyObservationFormat, SingleEnvironment
from .agents import WithTradeAgent, NoTradeAgent
from .jsettlers_bot import JSettlersBot

def run_bot():
    parser = argparse.ArgumentParser(description='JSettlers Python Bot')
    parser.add_argument('host', help='Server Host')
    parser.add_argument('port', type=int, help='Server Port')
    parser.add_argument('nickname', help='Bot Nickname')
    parser.add_argument('cookie', help='Auth Cookie (dummy)')
    parser.add_argument('--enable_trade', action='store_true', help='Enable trade actions in the agent')
    
    parser.add_argument('--create', action='store_true', help='Create a new game instead of joining')
    parser.add_argument('--reset', action='store_true', help='Reset the game')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode (controlled via stdin)')
    parser.add_argument('--game', default='pycatan_game', help='Game name to join (if not creating)')
    
    parser.add_argument('--seat', type=int, default=0, help='Seat number to request (0-3)')

    args = parser.parse_args()
    device = "cpu"
    fmt = PyObservationFormat()
    dummy_env = SingleEnvironment.new(
        format=fmt,
        opponents=3,
        trade_activated=True,
    )
    player_id, board, flat, legal_mask, trade_mask, position, done = dummy_env.start()
    dummy_env.close()
    args.action_dim = legal_mask.shape[0]
    config = vars(args)
    config["num_players"] = 4
    config["trade_info_dim"] = 5 + (config["num_players"] - 1)
    config["trade_pruning_k"] = 50

    # ボット作成
    if args.enable_trade:
        agent = WithTradeAgent(config, device)
    else:
        agent = NoTradeAgent(config, device)

    bot = JSettlersBot(args.host, args.port, args.nickname, args.cookie, agent, device=device, create=args.create)
    
    bot.connect()
    bot.authenticate() 

    # 実行モードの指定

    if args.interactive:
        # print(f"🎮 Mode: INTERACTIVE")
        bot.run_interactive()
        return

    if args.create:
        # print(f"🛠️  Mode: CREATOR")
         # 作成モード: CPUボットの数も渡す
        bot.run(mode="create", game_name=args.game, seat_num=args.seat)
        
    elif args.game:
        # print(f"👉 Mode: JOINER (Joining '{args.game}')")
        # 参加モード: 既存ゲームに入るだけ
        bot.run(mode="join", game_name=args.game, seat_num=args.seat)
        
    else:
        # デフォルト動作（ランダム作成、CPU3体）
        bot.run(mode="create", num_robots=3)

if __name__ == "__main__":
    run_bot()