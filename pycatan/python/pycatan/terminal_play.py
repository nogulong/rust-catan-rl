from pycatan import TerminalEnvironment, PyObservationFormat, OnnxInfer, TradeDecisionHandler_onnx
from pycatan.jsettlers.agents import WithTradeAgent
from importlib import resources

def main():
    fmt = PyObservationFormat()
    env = TerminalEnvironment.new(format=fmt, players=3)

    player_id, board, flat, normal_legal_mask, trade_legal_mask, position, done = env.start()
    agent = WithTradeAgent(config={
        "action_dim": normal_legal_mask.shape[0],
        "trade_info_dim": 8,
        "num_players": 3,
    }, device="cpu")
    while not done:
        action = agent.act(board, flat, normal_legal_mask, trade_legal_mask)
        player_id, board, flat, normal_legal_mask, trade_legal_mask, position, done = env.play(player_id, action)
    
if __name__ == "__main__":
    main()