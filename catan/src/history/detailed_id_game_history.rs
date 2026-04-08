use termion::{clear, cursor};
use std::io::{stdout, stdin, Write};
use crate::state::{State, TricellState};
use crate::state::PlayerId;
use crate::display::utils::grid_display;
use crate::display::{PrettyGridDisplay, pretty_public_player_hand};
use crate::player::get_global_possible_actions;
use crate::game::{Action, Phase};
use super::{TurnRecord, GameHistoryTrait, GameHistory, GameMetadata, GameSummary};
use serde::{Serialize, Deserialize};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use crate::board::setup;
use crate::game::apply::{apply, light_apply};

#[derive(Serialize, Deserialize)]
pub struct DetailedIdGameHistory{
    metadata: GameMetadata,
    actions: Vec<(usize, PlayerId)>, // グローバルアクションインデックスとプレイヤーIDのペア
    dev_turns: Vec<usize>, // BuyDevelopmentアクションのターンインデックス
    mthief_turns: Vec<usize>, // MoveThiefアクションのターンインデックス
    monopoly_turns: Vec<usize>, // DevelopmentMonopoleアクションのターンインデックス
    winner: Option<PlayerId>,
    num_dice_rolls: usize,
}

#[typetag::serde]
impl GameHistoryTrait for DetailedIdGameHistory{
    fn set_metadata(&mut self, metadata: GameMetadata){
        self.metadata = metadata;
    }

    fn set_start_info(&mut self, _state: &State, _player_count_order: &Vec<usize>){
    }

    fn push_snapshot(&mut self, _state: &State){}

    fn push_turn_record(&mut self, record: TurnRecord){
        let action_index = get_global_possible_actions(true).iter().position(|a| *a == record.action.unwrap())
            .expect("Action not found in global possible actions");
        self.actions.push((action_index, record.player_id));
        match record.action.unwrap() {
            Action::BuyDevelopment => {
                self.dev_turns.push(self.actions.len() - 1);
            },
            Action::MoveThief { .. } => {
                self.mthief_turns.push(self.actions.len() - 1);
            },
            Action::DevelopmentMonopole { .. } => {
                self.monopoly_turns.push(self.actions.len() - 1);
            },
            Action::RollDice { .. } => {
                self.num_dice_rolls += 1;
            },
            _ => {}
        }
    }

    fn display_test(&self) {
        let player_count = self.player_count() as usize;
        
        // 初期状態生成
        let mut rng = SmallRng::seed_from_u64(self.metadata.seed);
        let initial_state = setup::random_default::<TricellState, SmallRng>(&mut rng, player_count as u8);
        let mut players_order: Vec<usize> = (0..player_count).collect();
        players_order.shuffle(&mut rng);
        
        // 各ターンの状態を再構築
        let mut states: Vec<State> = Vec::new();
        states.push(initial_state.clone_box());
        
        let mut base = initial_state.clone_box();
        let mut phase = Phase::START_GAME;

        for (action_index, _) in &self.actions {
            let action = get_global_possible_actions(true)[*action_index];
            apply(&mut phase, &mut base, action, &mut rng);
            states.push(base.clone_box());
        }
        
        let mut current_turn = 0;

        loop {
            let mut screen = stdout();
            write!(screen, "{}{}", clear::All, cursor::Goto(1, 1))
                .expect("Failed to clear screen");

            // 盤面表示
            grid_display(&PrettyGridDisplay::INSTANCE, &mut screen, &states[current_turn])
                .expect("Failed to draw grid");

            // プレイヤー手札表示
            if self.player_count() > 0 {
                writeln!(screen, "\n════════ PLAYERS HANDS ════════").ok();
                for p in 0..self.player_count() {
                    let pid = PlayerId::from(p);
                    pretty_public_player_hand(&mut screen, pid, &states[current_turn])
                        .expect("Failed to print player hand");
                }
                writeln!(screen, "════════════════════════════════").ok();
            }

            // ヘッダー・前ターン情報
            println!("═════════════════════════════════════════════════════════════════");
            println!("🎮 CATAN GAME HISTORY VIEWER (Smallest)");
            println!("📊 Turn: {} / {} | Total Actions: {}",
                    current_turn, self.actions.len(), self.actions.len());
            
            if current_turn > 0 && current_turn <= self.actions.len() {
                let (prev_action_index, prev_player) = self.actions[current_turn - 1];
                let prev_action = get_global_possible_actions(true)[prev_action_index];
                println!("⏮ Prev Turn #{}  Action: {:?} Player: {:?}", current_turn - 1, prev_action, prev_player);
            }
            
            println!("─────────────────────────────────────────────────────────────────");
            println!("🎮 Controls: n(next)  p(prev)  j <n>  f(first)  l(last)  q(quit)");
            println!("─────────────────────────────────────────────────────────────────");
            print!("Command: ");
            screen.flush().expect("Failed to flush output");

            // 入力処理
            let mut input = String::new();
            stdin().read_line(&mut input).expect("Failed to read input");
            let input = input.trim().to_lowercase();

            match input.as_str() {
                "n" | "next" | "" => {
                    if current_turn + 1 < states.len() {
                        current_turn += 1;
                    } else if input.is_empty() {
                        println!("(last turn)");
                        self.wait_for_enter();
                    }
                },
                "p" | "prev" => {
                    if current_turn > 0 {
                        current_turn -= 1;
                    } else {
                        println!("(first turn)");
                        self.wait_for_enter();
                    }
                },
                "f" | "first" => current_turn = 0,
                "l" | "last" => current_turn = states.len() - 1,
                "q" | "quit" | "exit" => break,
                _ => {
                    if input.starts_with("j ") {
                        let jump_str = &input[2..];
                        match jump_str.parse::<usize>() {
                            Ok(t) if t < states.len() => current_turn = t,
                            Ok(t) => {
                                println!("invalid turn {t}");
                                self.wait_for_enter();
                            },
                            Err(_) => {
                                println!("parse error");
                                self.wait_for_enter();
                            }
                        }
                    } else {
                        println!("unknown command");
                        self.wait_for_enter();
                    }
                }
            }
        }
        println!("viewer closed");
    }
    
    fn reconstruct_turn(&self, turn_index: usize) -> Option<(State, Phase, PlayerId, Action)> {
        if turn_index >= self.actions.len() {
            return None;
        }
        let player_count = self.player_count() as usize;
        let mut rng = SmallRng::seed_from_u64(self.metadata.seed);
        let mut state = setup::random_default::<TricellState, SmallRng>(&mut rng, player_count as u8);
        let mut players_order: Vec<usize> = (0..player_count).collect();
        players_order.shuffle(&mut rng);
        let mut phase = Phase::START_GAME;
        for i in 0..turn_index {
            let action = get_global_possible_actions(true)[self.actions[i].0];
            light_apply(&mut phase, &mut state, action, &mut rng);
        }
        let position = phase.player();
        let action = get_global_possible_actions(true)[self.actions[turn_index].0];
        Some((state, phase, position, action))
    }

    fn player_count(&self) -> u8 {
        self.metadata.player_count
    }

    fn create_new(&self) -> GameHistory {
        Box::new(Self::new())
    }

    fn set_winner(&mut self, winner: PlayerId) {
        self.winner = Some(winner);
    }

    fn get_winner(&self) -> Option<PlayerId> {
        self.winner
    }

    fn history_len(&self) -> usize {
        self.actions.len()
    }

    fn get_turns_of_player(&self, player: PlayerId) -> Vec<usize> {
        let mut turns = Vec::new();
        for (i, (_, p)) in self.actions.iter().enumerate() {
            if *p == player {
                turns.push(i);
            }
        }
        turns
    }
    fn get_summary(&self) -> GameSummary {
        GameSummary {
            total_turns: self.history_len(),
            total_dice_rolls: self.num_dice_rolls,
            dev_turns: self.dev_turns.len(),
            mthief_turns: self.mthief_turns.len(),
            monopoly_turns: self.monopoly_turns.len(),
            winner: match self.get_winner() {
                Some(p) => p.to_usize(),
                None => usize::MAX,
            },
        }
    }

    fn get_development_turn(&self, development_index: usize) -> Option<usize> {
        if development_index < self.dev_turns.len() {
            Some(self.dev_turns[development_index])
        } else {
            None
        }
    }

    fn get_thief_turn(&self, thief_index: usize) -> Option<usize> {
        if thief_index < self.mthief_turns.len() {
            Some(self.mthief_turns[thief_index])
        } else {
            None
        }
    }

    fn get_monopole_turn(&self, monopole_index: usize) -> Option<usize> {
        if monopole_index < self.monopoly_turns.len() {
            Some(self.monopoly_turns[monopole_index])
        } else {
            None
        }
    }

}

impl DetailedIdGameHistory{
    pub fn new() -> Self{
        Self{
            metadata: GameMetadata::default(),
            actions: Vec::new(),
            dev_turns: Vec::new(),
            mthief_turns: Vec::new(),
            monopoly_turns: Vec::new(),
            winner: None,
            num_dice_rolls: 0,
        }
    }
    // ヘルパーメソッド：エラー時の一時停止
    fn wait_for_enter(&self) {
        println!("Press Enter to continue...");
        let mut input = String::new();
        stdin().read_line(&mut input).expect("Failed to read input");
    }
}

impl Default for DetailedIdGameHistory{
    fn default() -> Self {
        Self::new()
    }
}