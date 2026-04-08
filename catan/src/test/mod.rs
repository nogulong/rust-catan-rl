use crate::game::{Game, GameResult};
use crate::player::Randomy;
use crate::state::PlayerId;

#[test]
fn play_random_game() {
      // Initialize global possible actions before creating players (for DetailedIdGameHistory)
   crate::player::init_global_possible_actions(4);

   let mut game = Game::new();
   game.add_player(Box::new(Randomy::new_player(false)));
   game.add_player(Box::new(Randomy::new_player(false)));
   game.add_player(Box::new(Randomy::new_player(false)));
   game.add_player(Box::new(Randomy::new_player(false)));

   let (history, result) = game.setup_and_play();
   match result {
      GameResult::Finished { winner } => {
         assert_ne!(winner, PlayerId::NONE);
         // Winner should be one of the 4 players
         assert!(winner.to_u8() < 4);
      },
      _ => panic!("Game should finish with a winner"),
   }
   drop(history);

   // trade_activated = trueの場合もテスト
   let mut game = Game::new();
   game.add_player(Box::new(Randomy::new_player(true)));
   game.add_player(Box::new(Randomy::new_player(true)));
   game.add_player(Box::new(Randomy::new_player(true)));
   game.add_player(Box::new(Randomy::new_player(true)));
   let (history, result) = game.setup_and_play();
   match result {
      GameResult::Finished { winner } => {
         assert_ne!(winner, PlayerId::NONE);
         // Winner should be one of the 4 players
         assert!(winner.to_u8() < 4);
      },
      _ => panic!("Game should finish with a winner"),
   }
   drop(history);
}

#[test]
fn test_game_initialization() {
   let game = Game::new();
   assert_eq!(game.players.len(), 0);
}

#[test]
fn test_game_add_players() {
   // Initialize global possible actions
   crate::player::init_global_possible_actions(4);

   let mut game = Game::new();
   game.add_player(Box::new(Randomy::new_player(false)));
   assert_eq!(game.players.len(), 1);

   game.add_player(Box::new(Randomy::new_player(false)));
   assert_eq!(game.players.len(), 2);
}

#[test]
fn test_game_coherence_check() {
   use crate::state::TricellState;
   use crate::board::setup;
   use rand::SeedableRng;
   use rand::rngs::SmallRng;

   let seed = 12345u64;
   let mut rng = SmallRng::seed_from_u64(seed);
   let state = setup::random_default::<TricellState, SmallRng>(&mut rng, 4);

   // Verify basic state properties
   assert_eq!(state.player_count(), 4);

   // Check that resources in bank are reasonable
   use crate::utils::Resource;
   let bank = state.get_bank_resources();
   for res in Resource::ALL.iter() {
      let val = bank[*res];
      assert!(val >= 0 && val <= 19, "Bank resource {} should be between 0 and 19, got {}", res, val);
   }
}

#[test]
fn test_multiple_games_sequential() {
    // Initialize global possible actions once
   crate::player::init_global_possible_actions(4);

    // Play multiple games sequentially to ensure state cleanup works
    for _ in 0..2 {
        let mut game = Game::new();
        game.add_player(Box::new(Randomy::new_player(false)));
        game.add_player(Box::new(Randomy::new_player(false)));
        game.add_player(Box::new(Randomy::new_player(false)));
        game.add_player(Box::new(Randomy::new_player(false)));

        let (_, result) = game.setup_and_play();

        match result {
            GameResult::Finished { winner: _ } => (),
            _ => panic!("Game should finish with a winner"),
        }
    }
}