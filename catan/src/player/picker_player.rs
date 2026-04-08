use crate::state::{State, PlayerId};
use crate::game::{Action, Notification, Error, Phase, legal};
use crate::utils::{Resource, Resources};
use crate::board::Layout;
use super::CatanPlayer;
use once_cell::sync::OnceCell;
use std::sync::Once;

pub trait PickerPlayerTrait {
    type ACTIONS;
    type PICKED;

    fn new_game(&mut self, position: PlayerId, state: &State, possible_actions: &Vec<Action>);
    fn pick_action(&mut self, phase: &Phase, state: &State, legal_actions: &Self::ACTIONS) -> Self::PICKED;
    fn bad_action(&mut self, error: Error);
    fn is_trade_activated(&self) -> bool;
    fn notify(&mut self, notification: &Notification);
    fn results(&mut self, state: &State, winner: PlayerId);
}

pub fn generate_possible_actions(possible_actions: &mut Vec<Action>, player: PlayerId, state: &State, include_trade: bool, for_actor: bool) {
    generate_possible_actions_without_state(possible_actions, player, state.player_count(), state.get_layout(), include_trade, for_actor);
}

pub fn generate_possible_actions_without_state(possible_actions: &mut Vec<Action>, player: PlayerId, player_count: u8, layout: &Layout, include_trade: bool, for_actor: bool) {
    possible_actions.clear();
    // # BOARD
    // ## Hexes: MoveThief
    for hex in layout.hexes.iter() {
        for p in 0..player_count {
            let p = p + player.to_u8();
            let p = if p >= player_count { PlayerId::from(p - player_count) } else { PlayerId::from(p) };
            possible_actions.push(Action::MoveThief { hex: *hex, victim: p });
        }
    }
    // ## Paths: BuildRoad
    for path in layout.paths.iter() {
        possible_actions.push(Action::BuildRoad { path: *path });
    }
    // ## Intersections: BuildSettlement and BuildCity
    for intersection in layout.intersections.iter() {
        possible_actions.push(Action::BuildSettlement { intersection: *intersection });
        possible_actions.push(Action::BuildCity { intersection: *intersection });
    }
    // # FLAT
    // ## TurnPhase
    possible_actions.push(Action::RollDice);
    possible_actions.push(Action::EndTurn);
    // ## Trade
    for given in Resource::ALL.iter() {
        for asked in Resource::ALL.iter() {
            if given != asked {
                possible_actions.push(Action::TradeBank { given: *given , asked: *asked });
            }
        }
    }

    // ## Development
    possible_actions.push(Action::BuyDevelopment);
    possible_actions.push(Action::DevelopmentKnight);
    possible_actions.push(Action::DevelopmentRoadBuilding);
    for b in 0..2 {
        for l in 0..2-b {
            for o in 0..2-(b+l) {
                for g in 0..2-(b+l+o) {
                    let w = 2-(b+l+o+g);
                    let resources = Resources::new(b,l,o,g,w);
                    possible_actions.push(Action::DevelopmentYearOfPlenty { resources })
                }
            }
        }
    }
    for resource in Resource::ALL.iter() {
        possible_actions.push(Action::DevelopmentMonopole { resource: *resource });
    }
    // ## Discards
    // TODO: Cleaner
    for b in 0..5 {
        for l in 0..5-b {
            for o in 0..5-(b+l) {
                for g in 0..5-(b+l+o) {
                    let w = 4-(b+l+o+g);
                    let resources = Resources::new(b,l,o,g,w);
                    possible_actions.push(Action::Keep { resources })
                }
            }
        }
    }

    possible_actions.push(Action::TradePlayersAccept);
    possible_actions.push(Action::TradePlayersDecline);
    // ## TradePlayers (only 1-on-1 trade)
    if include_trade {
        let mut add_trade_actions = |offer: Resources, want: Resources| {
            for p in 0..player_count {
                if for_actor && p == 0 { continue; }
                let p_idx = p + player.to_u8();
                let p_id = if p_idx >= player_count { 
                    PlayerId::from(p_idx - player_count) 
                } else { 
                    PlayerId::from(p_idx)
                };

                possible_actions.push(Action::TradePlayers { 
                    offer: offer.clone(), 
                    want: want.clone(), 
                    partner: p_id 
                });
            }
        };
        // 1r on 1r trade
        for given in Resource::ALL.iter() {
            for asked in Resource::ALL.iter() {
                if given != asked {
                    add_trade_actions(
                        Resources::new_one(*given, 1), 
                        Resources::new_one(*asked, 1)
                    );
                }
            }
        }
        // 2r on 1r trade
        for given1 in 0..Resource::COUNT {
            for given2 in given1..Resource::COUNT {
                for asked in 0..Resource::COUNT {
                    if given1 != asked && given2 != asked {
                        let mut offer = Resources::ZERO;
                        offer[given1] += 1;
                        offer[given2] += 1;
                        let mut want = Resources::ZERO;
                        want[asked] += 1;
                        add_trade_actions(offer, want);
                    }
                }
            }
        }
        // 1r on 2r trade
        for given in 0..Resource::COUNT {
            for asked1 in 0..Resource::COUNT {
                for asked2 in asked1..Resource::COUNT {
                    if given != asked1 && given != asked2 {
                        let mut offer = Resources::ZERO;
                        offer[given] += 1;
                        let mut want = Resources::ZERO;
                        want[asked1] += 1;
                        want[asked2] += 1;
                        add_trade_actions(offer, want);
                    }
                }
            }
        }
    }
}

pub struct ActionPickerPlayer<T : PickerPlayerTrait<ACTIONS = Vec<Action>, PICKED = Action>> {
    position: PlayerId,
    possible_actions: Vec<Action>,
    action_length: usize,
    player: T,
}

pub struct IndexPickerPlayer<T : PickerPlayerTrait<ACTIONS = Vec<bool>, PICKED = u8>> {
    position: PlayerId,
    possible_actions: Vec<Action>,
    action_length: usize,
    player: T,
}

impl<T : PickerPlayerTrait<ACTIONS = Vec<Action>, PICKED = Action>> ActionPickerPlayer<T> {
    pub fn new(player: T) -> ActionPickerPlayer<T> {
        ActionPickerPlayer {
            position: PlayerId::NONE,
            possible_actions: Vec::new(),
            action_length: 0,
            player,
        }
    }

    fn init_possible_actions(&mut self, state: &State) {
        generate_possible_actions(&mut self.possible_actions, self.position, state, self.player.is_trade_activated(), true);
        self.action_length = self.possible_actions.len();
    }

    fn legal_actions(&mut self, phase: &Phase, state: &State) -> Vec<Action> {
        let mut legal_actions = Vec::new();
        for action in self.possible_actions.iter() {
            // TODO: More optimized
            // for example, don't check if every road is legal if you can't even afford a road
            if legal::legal(phase, state, *action, true).is_ok() {
                legal_actions.push(*action);
            }
        }
        legal_actions
    }
}

impl<T : PickerPlayerTrait<ACTIONS = Vec<bool>, PICKED = u8>> IndexPickerPlayer<T> {
    pub fn new(player: T) -> IndexPickerPlayer<T> {
        IndexPickerPlayer {
            position: PlayerId::NONE,
            possible_actions: Vec::new(),
            action_length: 0,
            player,
        }
    }

    fn init_possible_actions(&mut self, state: &State) {
        generate_possible_actions(&mut self.possible_actions, self.position, state, self.player.is_trade_activated(), true);
        self.action_length = self.possible_actions.len();
    }

    fn legal_actions(&mut self, phase: &Phase, state: &State) -> Vec<bool> {
        let mut legal_actions = Vec::new();
        for action in self.possible_actions.iter() {
            // TODO: More optimized
            // for example, don't check if every road is legal if you can't even afford a road
            legal_actions.push(legal::legal(phase, state, *action, true).is_ok());
        }
        legal_actions
    }
}

impl<T : PickerPlayerTrait<ACTIONS = Vec<Action>, PICKED = Action>> CatanPlayer for ActionPickerPlayer<T> {
    fn new_game(&mut self, position: PlayerId, state: &State) {
        self.position = position;
        self.init_possible_actions(state);
        self.player.new_game(position, state, &self.possible_actions)
    }

    fn pick_action(&mut self, phase: &Phase, state: &State) -> Action {
        let legal_actions = self.legal_actions(phase, &*state);
        self.player.pick_action(phase, state, &legal_actions)
    }

    fn bad_action(&mut self, error: Error) {
        self.player.bad_action(error)
    }

    fn notify(&mut self, notification: &Notification) {
        self.player.notify(notification)
    }

    fn results(&mut self, state: &State, winner: PlayerId) {
        self.player.results(state, winner)
    }
}

impl<T : PickerPlayerTrait<ACTIONS = Vec<bool>, PICKED = u8>> CatanPlayer for IndexPickerPlayer<T> {
    fn new_game(&mut self, position: PlayerId, state: &State) {
        self.position = position;
        self.init_possible_actions(state);
        self.player.new_game(position, state, &self.possible_actions);
    }

    fn pick_action(&mut self, phase: &Phase, state: &State) -> Action {
        let legal_actions = self.legal_actions(phase, state);
        loop {
            let action = self.player.pick_action(phase, state, &legal_actions) as usize;
            if action < self.possible_actions.len() {
                return self.possible_actions[action as usize];
            }
            self.player.bad_action(Error::ActionNotUnderstood)
        }
    }

    fn bad_action(&mut self, error: Error) {
        self.player.bad_action(error);
    }

    fn notify(&mut self, notification: &Notification) {
        self.player.notify(notification);
    }

    fn results(&mut self, state: &State, winner: PlayerId) {
        self.player.results(state, winner)
    }
}

static GLOBAL_POSSIBLE_ACTIONS_WITH_TRADE: OnceCell<Vec<Action>> = OnceCell::new();
static GLOBAL_POSSIBLE_ACTIONS_WITHOUT_TRADE: OnceCell<Vec<Action>> = OnceCell::new();
static INIT: Once = Once::new();

pub fn init_global_possible_actions(num_players: u8) {
    INIT.call_once(|| {
        let mut possible_actions_with_trade = Vec::new();
        let mut possible_actions_without_trade = Vec::new();
        
        generate_possible_actions_without_state(&mut possible_actions_with_trade, PlayerId::from(0u8), num_players, &crate::board::layout::DEFAULT, true, false);
        generate_possible_actions_without_state(&mut possible_actions_without_trade, PlayerId::from(0u8), num_players, &crate::board::layout::DEFAULT, false, false);
        
        GLOBAL_POSSIBLE_ACTIONS_WITH_TRADE.set(possible_actions_with_trade).unwrap();
        GLOBAL_POSSIBLE_ACTIONS_WITHOUT_TRADE.set(possible_actions_without_trade).unwrap();
    });
}

pub fn get_global_possible_actions(trade_activated: bool) -> &'static Vec<Action> {
    if trade_activated {
        GLOBAL_POSSIBLE_ACTIONS_WITH_TRADE
            .get()
            .expect("GLOBAL_POSSIBLE_ACTIONS_WITH_TRADE not initialized; call init_global_possible_actions(...) first")
    } else {
        GLOBAL_POSSIBLE_ACTIONS_WITHOUT_TRADE
            .get()
            .expect("GLOBAL_POSSIBLE_ACTIONS_WITHOUT_TRADE not initialized; call init_global_possible_actions(...) first")
    }
}
