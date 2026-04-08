// mod full_game_history;
// mod small_game_history;
// mod smallest_game_history;
// mod id_game_history;
mod detailed_id_game_history;

// pub use full_game_history::FullGameHistory;
// pub use small_game_history::SmallGameHistory;
// pub use smallest_game_history::SmallestGameHistory;
// pub use id_game_history::IdGameHistory;
pub use detailed_id_game_history::DetailedIdGameHistory;


use crate::state::State;
use crate::state::PlayerId;
use serde::{Serialize, Deserialize};
use crate::game::{Action, Phase};
use crate::utils::{Coord, Resources, DevelopmentCard};

pub type GameHistory = Box<dyn GameHistoryTrait>;

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct GameMetadata {
    pub player_count: u8,
    pub seed: u64,
}
impl Default for GameMetadata {
    fn default() -> Self {
        Self {
            player_count: 0,
            seed: 0,
        }
    }
}

pub struct GameSummary {//history_channelでの伝達用クラス
    pub total_turns: usize,
    pub total_dice_rolls: usize,
    pub dev_turns: usize,
    pub mthief_turns: usize,
    pub monopoly_turns: usize,
    pub winner: usize,
}

#[typetag::serde]
pub trait GameHistoryTrait {
    fn set_metadata(&mut self, metadata: GameMetadata);
    fn set_start_info(&mut self, state: &State, players_order: &Vec<usize>);
    fn push_snapshot(&mut self, state: &State);
    fn push_turn_record(&mut self, record: TurnRecord);
    fn display_test(&self);
    fn reconstruct_turn(&self, turn_index: usize) -> Option<(State, Phase, PlayerId, Action)>;
    fn player_count(&self) -> u8;
    fn create_new(&self) -> GameHistory;
    fn set_winner(&mut self, winner: PlayerId);
    fn get_winner(&self) -> Option<PlayerId>;
    fn history_len(&self) -> usize {
        0
    }
    fn get_turns_of_player(&self, _player: PlayerId) -> Vec<usize> {
        Vec::new()
    }
    fn get_summary(&self) -> GameSummary {
        GameSummary {
            total_turns: self.history_len(),
            total_dice_rolls: 0,
            dev_turns: 0,
            mthief_turns: 0,
            monopoly_turns: 0,
            winner: match self.get_winner() {
                Some(p) => p.to_usize(),
                None => usize::MAX,
            },
        }
    }
    fn get_development_turn(&self, _development_index: usize) -> Option<usize> {
        None
    }
    fn get_thief_turn(&self, _thief_index: usize) -> Option<usize> {
        None
    }
    fn get_monopole_turn(&self, _monopole_index: usize) -> Option<usize> {
        None
    }
}

pub struct TurnRecord {
    pub player_id: PlayerId,
    pub action: Option<Action>,
    pub phase: Phase,
    //pub action_id: usize,
    pub dice_roll: Option<u8>,
    pub knight_pos: Option<Coord>,
    pub build_pos: Option<Coord>,
    pub resource_changes: Vec<Resources>,
    pub development_card: Option<DevelopmentCard>,
    pub new_longest_road: Option<PlayerId>,
    pub new_largest_army: Option<PlayerId>,
    //pub vp_change: Vec<i8>,
}
impl TurnRecord {
    pub fn new(player_count: usize) -> Self {
        Self {
            player_id: PlayerId::NONE,
            action: None,
            phase: Phase::START_GAME,
            //action_id: 0,
            dice_roll: None,
            knight_pos: None,
            build_pos: None,
            resource_changes: vec![Resources::ZERO; player_count],
            development_card: None,
            new_longest_road: None,
            new_largest_army: None,
            //vp_change: vec![0; player_count],
        }
    }
    pub fn set_player_id(&mut self, player_id: PlayerId) {
        self.player_id = player_id;
    }
    pub fn set_action(&mut self, action: Action) {
        self.action = Some(action);
    }
    pub fn set_phase(&mut self, phase: Phase) {
        self.phase = phase;
    }
    //pub fn set_action_id(&mut self, action_id: usize) {
    //     self.action_id = action_id;
    // }
    pub fn set_dice_roll(&mut self, roll: u8) {
        self.dice_roll = Some(roll);
    }
    pub fn set_knight_pos(&mut self, pos: Coord) {
        self.knight_pos = Some(pos);
    }
    pub fn set_build_pos(&mut self, pos: Coord) {
        self.build_pos = Some(pos);
    }
    pub fn set_resources_change(&mut self, player_id: PlayerId, change: Resources) {
        self.resource_changes[player_id.to_usize()] = change;
    }
    pub fn set_resources_change_all(&mut self, change: Vec<Resources>) {
        self.resource_changes = change;
    }
    pub fn set_development_card(&mut self, card: DevelopmentCard) { 
        self.development_card = Some(card);
    }
    pub fn set_new_longest_road(&mut self, player_id: PlayerId) {
        self.new_longest_road = Some(player_id);
    }
    pub fn set_new_largest_army(&mut self, player_id: PlayerId) {
        self.new_largest_army = Some(player_id);
    }
    pub fn set_discards_change(&mut self, discards: &Vec<(PlayerId, Option<Resources>)>) {
        for (p, r) in discards {
            if let Some(res) = r {
                self.set_resources_change(*p, -*res);
            }
        }
    }
    // pub fn set_vp_change(&mut self, player_id: PlayerId, change: i8) {
    //     self.vp_change[player_id.to_usize()] = change;
    // }
}
