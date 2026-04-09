pub use catan_player::action_parser::parse_action;

use catan::game::Game;
use catan::player::Randomy;

use catan_player::TerminalPlayer;

fn main() {
    println!("[START]");

    catan::player::init_global_possible_actions(2);

    let mut game = Game::new();
    game.add_player(Box::new(TerminalPlayer::new()));
    game.add_player(Box::new(Randomy::new_player(false)));
    game.setup_and_play();

    println!("[END]");
}
