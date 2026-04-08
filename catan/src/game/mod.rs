pub mod action;
mod error;
mod phase;
mod notification;
pub mod apply;
pub mod legal;

#[cfg(test)]
mod legal_tests;
#[cfg(test)]
mod apply_tests;

pub use error::Error;
pub use action::{Action, ActionCategory};
pub use phase::{Phase, TurnPhase, DevelopmentPhase};
pub use notification::Notification;
use crate::state::{State, TricellState};
use crate::board::setup;
use crate::state::PlayerId;
use crate::player::CatanPlayer;
use crate::history::{GameHistory, DetailedIdGameHistory, GameMetadata};

// --------------------------------------------------------------------------------------------- //

use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::Rng;

use apply::apply;

pub struct Game {
    pub players: Vec<Box<dyn CatanPlayer>>,
    pub history: GameHistory,
}

pub enum GameResult {
    Finished { winner: PlayerId },
    Interrupted,
    Reseted,
}

impl Game {
    pub fn new() -> Game {
        Game {
            players: Vec::new(),
            history: Box::new(DetailedIdGameHistory::new()),
        }
    }

    pub fn add_player(&mut self, player: Box<dyn CatanPlayer>) {
        self.players.push(player);
    }

    fn notify_all(&mut self, notification: Notification) {
        for player in self.players.iter_mut() {
            player.notify(&notification);
        }
    }

    pub fn setup_and_play(&mut self) -> (GameHistory, GameResult) {
        let player_count = self.players.len();
        let seed: u64 = rand::rng().random();
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut state = setup::random_default::<TricellState, SmallRng>(&mut rng, player_count as u8);
        let mut players_order: Vec<usize> = (0..player_count).collect();
        players_order.shuffle(&mut rng);
        let metadata = GameMetadata {
            player_count: player_count as u8,
            seed,
        };
        self.history.set_metadata(metadata);
        let result = self.play(&mut rng, &mut state, players_order);
        let new_history = self.history.create_new();
        (std::mem::replace(&mut self.history, new_history), result)
    }

    pub fn play(&mut self, rng: &mut SmallRng, state: &mut State, players_order: Vec<usize>) -> GameResult {
        let mut phase = Phase::START_GAME;
        self.history.push_snapshot(state);

        for (i, player) in players_order.iter().enumerate() {
            self.players[*player].new_game(PlayerId::from(i), &state);
        }
        loop {
            // If the game is finished, exit
            if let Phase::FinishedGame { winner } = phase {
                for player in players_order.iter() {
                    self.players[*player].results(&state, winner);
                }
                self.history.set_winner(winner);
                return GameResult::Finished { winner };
            }

            // Get the player object that is supposed to be making a decision
            let player = &mut self.players[players_order[phase.player().to_u8() as usize]];
            let mut action;
            loop {
                // Ask player to take action
                action = player.pick_action(&phase, &state);
                if action == Action::Exit {
                    return GameResult::Interrupted;
                } else if action == Action::Reset {
                    return GameResult::Reseted;
                }

                // Checks if action is legal
                let result = legal::legal(&phase, &state, action, true);
                if let Err(error) = result {
                    // Tells player if action was invalid
                    player.bad_action(error);
                } else {
                    break;
                }
            }

            // Notifies every player of action played
            let prev_phase = phase;
            let prev_player = phase.player();
            let discards_clone = state.peek_discards().clone();
            // Applies action
            let record = apply(&mut phase, state, action, rng);
            // Discard と tradeのアクセプトの場合、まとめて通知 & 履歴記録
            // prev_phaseがdiscardかつphaseがdiscardでない場合はnotifyしない
            if action == Action::RollDice {
                self.notify_all(Notification::ResourcesRolled {
                    roll: record.dice_roll.unwrap(),
                    resources: record.resource_changes.clone()
                });
            } else if matches!(prev_phase, Phase::Turn { turn_phase: TurnPhase::Discard(_),.. }) {
                if !matches!(phase, Phase::Turn { turn_phase: TurnPhase::Discard(_), .. }) {
                    self.notify_all(Notification::Discards {
                        discards: discards_clone.into_iter().map(|(p, r)| (p, r)).collect()
                    });
                }
            } else if action == Action::TradePlayersAccept {
                self.notify_all(Notification::TradeAccepted);
            } else if action == Action::TradePlayersDecline {
                self.notify_all(Notification::TradeDeclined);
            } else {
                self.notify_all(Notification::ActionPlayed { by: prev_player, action });
            }
            self.history.push_turn_record(record);
            self.history.push_snapshot(state);
            let coherence = check_coherence(state);
            if coherence.is_err() {
                println!("[INCOHERENCE] {:?} --({:?})-> {:?}", prev_phase, action, phase);
                panic!("{:?}", coherence.err());
            }
        }
    }
}

use crate::utils::{Resource, Resources};

fn check_coherence(state: &State) -> Result<(),String> {
    let mut players_resources = Resources::ZERO;
    for p in 0..state.player_count() {
        let player = PlayerId::from(p);
        let hand = state.get_player_hand(player).resources;
        for res in Resource::ALL.iter() {
            let v = hand[*res];
            if v > 19 || v < 0 {
                return Err(format!("Player {:?} has {} of {}", player, v, res));
            }
        }
        players_resources += hand;
    }
    let bank_resources = state.get_bank_resources();
    for res in Resource::ALL.iter() {
        let v = bank_resources[*res];
        let pv = players_resources[*res];
        if v > 19 || v < 0 || pv+v != 19 {
            return Err(format!("For resource {}: Bank has {} / Players have {}", res, v, pv));
        }
    }
    Ok(())
}
