import pycatan
import argparse
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="変換するモデルパス")
    parser.add_argument("--num_players", type=int, default=4, help="プレイヤー数")
    parser.add_argument("--num_xdim_blocks", type=int, default=8, help="CatanXDimModelのブロック数")
    args = parser.parse_args()

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
    config = vars(args)
    config["board_logits_mask"] = pycatan.get_board_logits_mask(args.num_players)
    config["trade_info_dim"] = 5 + (args.num_players - 1)

    model = pycatan.CatanXDimModel(config)
    trade_model = pycatan.TradeExpectorNet(config)
    device = "cpu"

    checkpoint = torch.load(config["model_path"], map_location=device)
        
    model.load_state_dict(checkpoint['model_state'])
    trade_model.load_state_dict(checkpoint['trade_model_state'])
    filename = "main_model.onnx"
    filename_trade = "trade_model.onnx"

    pycatan.export_onnx(model, config, device, filename, remove_aux_head=False)
    pycatan.export_onnx_trade(trade_model, config, device, filename_trade)

if __name__ == "__main__":
    main()