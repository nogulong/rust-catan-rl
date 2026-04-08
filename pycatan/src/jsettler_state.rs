use ndarray::Array1;
use rand::Rng;

use catan::utils::{Coord, DevelopmentCard, DevelopmentCards, Harbor, Hex, LandHex, Resource, Resources};
use catan::board::utils::topology::Topology;

use catan::player::generate_possible_actions;
use catan::state::{State, StateMaker, PlayerId};
use catan::game::{Action, DevelopmentPhase, Phase, TurnPhase, legal};
use catan::state::TricellState;
use pyo3::prelude::*;
use numpy::convert::IntoPyArray;

use crate::{PyObservationFormat, PyCatanObservation};

pub fn js_hex_to_rust_coord(js_coord: i32) -> Coord {
    let rust_coord = match js_coord {
        51 => (-8, 0),
        53 => (-6, 2),
        55 => (-4, 4),

        83 => (-6, -2),
        85 => (-4, 0),
        87 => (-2, 2),
        89 => (0, 4),

        115 => (-4, -4),
        117 => (-2, -2),
        119 => (0, 0),
        121 => (2, 2),
        123 => (4, 4),

        149 => (0, -4),
        151 => (2, -2),
        153 => (4, 0),
        155 => (6, 2),

        183 => (4, -4),
        185 => (6, -2),
        187 => (8, 0),
        _ => panic!("Invalid JS hex coord: {}", js_coord),
    };
    Coord {
        x: rust_coord.0,
        y: rust_coord.1,
    }
}

pub fn rust_coord_to_js_hex(rust_coord: &Coord) -> i32 {
    match (rust_coord.x, rust_coord.y) {
        (-8, 0) => 51,
        (-6, 2) => 53,
        (-4, 4) => 55,

        (-6, -2) => 83,
        (-4, 0) => 85,
        (-2, 2) => 87,
        (0, 4) => 89,

        (-4, -4) => 115,
        (-2, -2) => 117,
        (0, 0) => 119,
        (2, 2) => 121,
        (4, 4) => 123,

        (0, -4) => 149,
        (2, -2) => 151,
        (4, 0) => 153,
        (6, 2) => 155,

        (4, -4) => 183,
        (6, -2) => 185,
        (8, 0) => 187,
        _ => panic!("Invalid Rust hex coord: ({}, {})", rust_coord.x, rust_coord.y),
    }
}

pub fn js_intersection_to_rust_coord(js_coord: i32) -> Coord {
    let rust_coord = match js_coord {
        35 =>(-10, 1),
        37 =>(-8, 3),
        39 =>(-6, 5),

        50 =>(-10, -1),
        52 =>(-8, 1),
        54 =>(-6, 3),
        56 =>(-4, 5),

        67 =>(-8, -1),
        69 =>(-6, 1),
        71 =>(-4, 3),
        73 =>(-2, 5),

        82 =>(-8, -3),
        84 =>(-6, -1),
        86 =>(-4, 1),
        88 =>(-2, 3),
        90 =>(0, 5),

        99 =>(-6, -3),
        101 =>(-4, -1),
        103 =>(-2, 1),
        105 =>(0, 3),
        107 =>(2, 5),

        114 =>(-6, -5),
        116 =>(-4, -3),
        118 =>(-2, -1),
        120 =>(0, 1),
        122 =>(2, 3),
        124 =>(4, 5),

        131 =>(-4, -5),
        133 =>(-2, -3),
        135 =>(0, -1),
        137 =>(2, 1),
        139 =>(4, 3),
        141 =>(6, 5),

        148 =>(-2, -5),
        150 =>(0, -3),
        152 =>(2, -1),
        154 =>(4, 1),
        156 =>(6, 3),

        165 =>(0, -5),
        167 =>(2, -3),
        169 =>(4, -1),
        171 =>(6, 1),
        173 =>(8, 3),

        182 =>(2, -5),
        184 =>(4, -3),
        186 =>(6, -1),
        188 =>(8, 1),

        199 =>(4, -5),
        201 =>(6, -3),
        203 =>(8, -1),
        205 =>(10, 1),

        216 =>(6, -5),
        218 =>(8, -3),
        220 =>(10, -1),
        _ => panic!("Invalid JS intersection coord: {}", js_coord),
    };
    Coord {
        x: rust_coord.0,
        y: rust_coord.1,
    }
}

fn rust_coord_to_js_intersection(rust_coord: &Coord) -> i32 {
    match (rust_coord.x, rust_coord.y) {
        (-10, 1) => 35,
        (-8, 3) => 37,
        (-6, 5) => 39,

        (-10, -1) => 50,
        (-8, 1) => 52,
        (-6, 3) => 54,
        (-4, 5) => 56,

        (-8, -1) => 67,
        (-6, 1) => 69,
        (-4, 3) => 71,
        (-2, 5) => 73,

        (-8, -3) => 82,
        (-6, -1) => 84,
        (-4, 1) => 86,
        (-2, 3) => 88,
        (0, 5) => 90,

        (-6, -3) => 99,
        (-4, -1) => 101,
        (-2, 1) => 103,
        (0, 3) => 105,
        (2, 5) => 107,

        (-6, -5) => 114,
        (-4, -3) => 116,
        (-2, -1) => 118,
        (0, 1) => 120,
        (2, 3) => 122,
        (4, 5) => 124,

        (-4, -5) => 131,
        (-2, -3) => 133,
        (0, -1) => 135,
        (2, 1) => 137,
        (4, 3) => 139,
        (6, 5) => 141,

        (-2, -5) => 148,
        (0, -3) => 150,
        (2, -1) => 152,
        (4, 1) => 154,
        (6, 3) => 156,

        (0, -5) => 165,
        (2, -3) => 167,
        (4, -1) => 169,
        (6, 1) => 171,
        (8, 3) =>173,

        (2, -5) => 182,
        (4, -3) => 184,
        (6, -1) => 186,
        (8, 1) => 188,

        (4, -5) => 199,
        (6, -3) => 201,
        (8, -1) => 203,
        (10, 1) => 205,

        (6, -5) => 216,
        (8, -3) => 218,
        (10, -1) => 220,
    _ => panic!("Invalid Rust intersection coord: ({}, {})", rust_coord.x, rust_coord.y),
    }
}

fn js_edge_to_rust_coord(js_coord: i32) -> Coord {
    let rust_coord = match js_coord {
        34 => (-10, 0),
        35 => (-9, 1),
        36 => (-8, 2),
        37 => (-7, 3),
        38 => (-6, 4),
        39 => (-5, 5),

        50 => (-9, -1),
        52 => (-7, 1),
        54 => (-5, 3),
        56 => (-3, 5),

        66 => (-8, -2),
        67 => (-7, -1),
        68 => (-6, 0),
        69 => (-5, 1),
        70 => (-4, 2),
        71 => (-3, 3),
        72 => (-2, 4),
        73 => (-1, 5),

        82 => (-7, -3),
        84 => (-5, -1),
        86 => (-3, 1),
        88 => (-1, 3),
        90 => (1, 5),

        98 => (-6, -4),
        99 => (-5, -3),
        100 => (-4, -2),
        101 => (-3, -1),
        102 => (-2, 0),
        103 => (-1, 1),
        104 => (0, 2),
        105 => (1, 3),
        106 => (2, 4),
        107 => (3, 5),

        114 => (-5, -5),
        116 => (-3, -3),
        118 => (-1, -1),
        120 => (1, 1),
        122 => (3, 3),
        124 => (5, 5),

        131 => (-3, -5),
        132 => (-2, -4),
        133 => (-1, -3),
        134 => (0, -2),
        135 => (1, -1),
        136 => (2, 0),
        137 => (3, 1),
        138 => (4, 2),
        139 => (5, 3),
        140 => (6, 4),

        148 => (-1, -5),
        150 => (1, -3),
        152 => (3, -1),
        154 => (5, 1),
        156 => (7, 3),

        165 => (1, -5),
        166 => (2, -4),
        167 => (3, -3),
        168 => (4, -2),
        169 => (5, -1),
        170 => (6, 0),
        171 => (7, 1),
        172 => (8, 2),

        182 => (3, -5),
        184 => (5, -3),
        186 => (7, -1),
        188 => (9, 1),

        199 => (5, -5),
        200 => (6, -4),
        201 => (7, -3),
        202 => (8, -2),
        203 => (9, -1),
        204 => (10, 0),
    _ => panic!("Invalid JS edge coord: {}", js_coord),
    };
    Coord {
        x: rust_coord.0,
        y: rust_coord.1,
    }
}

fn rust_coord_to_js_edge(rust_coord: &Coord) -> i32 {
    match (rust_coord.x, rust_coord.y) {
        (-10, 0) => 34,
        (-9, 1) => 35,
        (-8, 2) => 36,
        (-7, 3) => 37,
        (-6, 4) => 38,
        (-5, 5) => 39,

        (-9, -1) => 50,
        (-7, 1) => 52,
        (-5, 3) => 54,
        (-3, 5) => 56,

        (-8, -2) => 66,
        (-7, -1) => 67,
        (-6, 0) => 68,
        (-5, 1) => 69,
        (-4, 2) => 70,
        (-3, 3) => 71,
        (-2, 4) => 72,
        (-1, 5) => 73,

        (-7, -3) => 82,
        (-5, -1) => 84,
        (-3, 1) => 86,
        (-1, 3) => 88,
        (1, 5) => 90,

        (-6, -4) => 98,
        (-5, -3) => 99,
        (-4, -2) => 100,
        (-3, -1) => 101,
        (-2, 0) => 102,
        (-1, 1) =>103,
        (0, 2) =>104,
        (1, 3) =>105,
        (2, 4) =>106,
        (3, 5) =>107,

        (-5, -5) =>114,
        (-3, -3) =>116,
        (-1, -1) =>118,
        (1, 1) =>120,
        (3, 3) =>122,
        (5, 5) =>124,

        (-3, -5) =>131,
        (-2, -4) =>132,
        (-1, -3) =>133,
        (0, -2) =>134,
        (1, -1) =>135,
        (2, 0) =>136,
        (3, 1) =>137,
        (4, 2) =>138,
        (5, 3) =>139,
        (6, 4) =>140,

        (-1, -5) =>148,
        (1, -3) =>150,
        (3, -1) =>152,
        (5, 1) =>154,
        (7, 3) =>156,

        (1, -5) =>165,
        (2, -4) =>166,
        (3, -3) =>167,
        (4, -2) =>168,
        (5, -1) =>169,
        (6, 0) =>170,
        (7, 1) =>171,
        (8, 2) =>172,

        (3, -5) =>182,
        (5, -3) =>184,
        (7, -1) =>186,
        (9, 1) =>188,

        (5, -5) =>199,
        (6, -4) =>200,
        (7, -3) =>201,
        (8, -2) =>202,
        (9, -1) =>203,
        (10, 0) =>204,
    _ => panic!("Invalid Rust edge coord: ({}, {})", rust_coord.x, rust_coord.y),
    }
}

pub fn js_res_to_rust_res(js_res: i32) -> Resource {
    match js_res {
        1 => Resource::Brick,
        2 => Resource::Ore,
        3 => Resource::Wool,
        4 => Resource::Grain,
        5 => Resource::Lumber,
        _ => panic!("Invalid JS resource: {}", js_res),
    }
}

pub fn rust_res_to_js_res(rust_res: Resource) -> i32 {
    match rust_res {
        Resource::Brick => 1,
        Resource::Ore => 2,
        Resource::Wool => 3,
        Resource::Grain => 4,
        Resource::Lumber => 5,
    }
}

pub fn js_index_to_direction(js_index: i32) -> (i8, i8) {
    match js_index {
        1 => (1, 1), // 右上
        2 => (2, 0), // 右
        3 => (1, -1),// 右下
        4 => (-1, -1),// 左下
        5 => (-2, 0),// 左
        6 => (-1, 1),// 左上 
        _ => panic!("Invalid JS direction index: {}", js_index),
    }
}

const JS_INDEX_TO_RUST_HEX: [(i8, i8); 37] = [
                (-6, 6), (-2, 6), (2, 6), (6, 6),
        (-8, 4), (-4, 4), (0, 4), (4, 4), (8, 4),
    (-10, 2), (-6, 2), (-2, 2), (2, 2), (6, 2), (10, 2),
(-12, 0), (-8, 0), (-4, 0), (0, 0), (4, 0), (8, 0), (12, 0),
    (-10, -2), (-6, -2), (-2, -2), (2, -2), (6, -2), (10, -2),
        (-8, -4), (-4, -4), (0, -4), (4, -4), (8, -4),
            (-6, -6), (-2, -6), (2, -6), (6, -6),
];

#[pyclass]
pub struct PyTricellState {
    state: State,
    phase: Phase,
    player_id: PlayerId,
    possible_actions: Array1<Action>,
    normal_action_length: usize,
    trade_length: usize,
    trade_activated: bool,
    trade_limit: usize,
    trade_count: usize,
}
#[pymethods]
impl PyTricellState {
    #[staticmethod]
    #[pyo3(signature = (players=3, trade_activated=true, trade_limit=1))]
    pub fn new(players: usize, trade_activated: bool, trade_limit: usize) -> Self {
        PyTricellState {
            state: TricellState::new_empty(&catan::board::layout::DEFAULT, players as u8),
            phase: Phase::START_GAME,
            player_id: PlayerId::from(0 as u8),//仮の措置
            possible_actions: vec![Action::EndTurn;0].into_iter().collect(),
            normal_action_length: 0,
            trade_length: 0,
            trade_activated,
            trade_limit,
            trade_count: 0,
        }
    }

    pub fn set_player_id(&mut self, player: i32) {
        self.player_id = PlayerId::from(player as u8);
        let mut possible_action_vec = Vec::new();
        generate_possible_actions(&mut possible_action_vec, self.player_id, &self.state, true, true);
        self.possible_actions = possible_action_vec.into_iter().collect();

        let mut nontrade_actions = Vec::new();
        generate_possible_actions(&mut nontrade_actions, self.player_id, &self.state, false, true);
        self.normal_action_length = nontrade_actions.len();
        self.trade_length = self.possible_actions.len() - self.normal_action_length;
        // development cards の初期化をしておく
        *self.state.get_development_cards_mut() = DevelopmentCards {
            knight: 14,
            road_building: 2,
            year_of_plenty: 2,
            monopole: 2,
            victory_point: 5,
        }
    }

    fn is_my_turn(&self) -> bool {
        self.phase.player() == self.player_id
    }

    pub fn put_piece(&mut self, player: i32, piece_type: i32, coord: i32) {
        let player_id = PlayerId::from(player as u8);
        match piece_type {
            0 => { // road
                let coord = js_edge_to_rust_coord(coord);
                self.state.set_dynamic_path(coord, player_id).unwrap();

                let hand = self.state.get_player_hand_mut(player_id);
                hand.road_pieces -= 1;

                self.state.update_longest_road(player_id, coord);
            }
            1 => { // settlement
                let coord = js_intersection_to_rust_coord(coord);
                self.state.set_dynamic_intersection(coord, player_id, false).unwrap();

                let harbor = self.state.get_static_harbor(coord).expect("Harbor error");
                let hand = self.state.get_player_hand_mut(player_id);
                hand.settlement_pieces -= 1;
                hand.building_vp += 1;
                hand.harbor.add(harbor);
                // break checkも後で追加
                if let Phase::InitialPlacement { player: _, placing_second, placing_road: _ } = &self.phase {
                    if *placing_second {
                        // 2回目の配置なら資源を配る
                        for hex in self.state.intersection_hex_neighbours(coord).expect("Hex neighbour error") {
                            if let Hex::Land(LandHex::Prod(res, _)) = self.state.get_static_hex(hex).expect("Static hex error") {
                                self.state.get_player_hand_mut(player_id).resources[res] += 1;
                                // self.state.get_bank_resources_mut()[res] -= 1;
                            }
                        }
                    }
                }
            }
            2 => { // city
                let coord = js_intersection_to_rust_coord(coord);
                self.state.set_dynamic_intersection(coord, player_id, true).unwrap();
                let hand = self.state.get_player_hand_mut(player_id);
                hand.settlement_pieces += 1;
                hand.city_pieces -= 1;
                hand.building_vp += 1;
            }
            3 => { // thief
                let coord = js_hex_to_rust_coord(coord);
                self.state.set_thief_hex(coord);
            }
            _ => {
                panic!("Invalid piece type: {}", piece_type);
            }
        }
    }

    pub fn update_player_elements(&mut self, player: i32, action: i32, element: i32, amount: i32) {
        let player_id = PlayerId::from(player as u8);
        let player_hand = self.state.get_player_hand_mut(player_id);
        if element > 0 && element <= 5 {
            let resource = js_res_to_rust_res(element);
            match action {
                100 => { // set
                    player_hand.resources[resource] = amount as i8;
                }
                101 => { // add
                    player_hand.resources[resource] += amount as i8;
                }
                102 => { // remove
                    player_hand.resources[resource] -= amount as i8;
                }
                _ => {
                    panic!("Invalid action for resource: {}", action);
                }
            }             
        }
        else if element == 15 { //騎士力
            player_hand.knights += 1;
            self.state.update_largest_army(player_id);
        }
        else if element == 17 {//手持ち資源総数の通知
            // 後で考える
        }
        else if element == 19 {//発展カードの使用
            if let Phase::Turn { player: _, turn_phase: _, development_phase } = &mut self.phase {
                *development_phase = DevelopmentPhase::DevelopmentPlayed;
            }
        }
    }

    pub fn get_hand(&self, player: i32) -> Vec<i32> {
        let player_id = PlayerId::from(player as u8);
        let hand = self.state.get_player_hand(player_id);
        let mut res_map = vec![0; 5];
        for r in Resource::ALL.iter() {
            let idx = rust_res_to_js_res(*r) - 1;
            res_map[idx as usize] = hand.resources[*r] as i32;
        }
        res_map
    }

    pub fn update_phase(&mut self, phase: i32) {
        let player = if let Phase::Turn { player, turn_phase: _, development_phase: _ } = &self.phase {
            *player
        } else {
            self.phase.player()
        };
        let current_turn_phase = if let Phase::Turn { player: _, turn_phase, development_phase: _ } = &self.phase {
            turn_phase.clone()
        } else {
            TurnPhase::PreRoll
        };
        let current_development_phase = if let Phase::Turn { player: _, turn_phase: _, development_phase } = &self.phase {
            development_phase.clone()
        } else {
            DevelopmentPhase::Ready
        };
        self.phase = match phase {
            5 => Phase::InitialPlacement { player, placing_second: false, placing_road: false },
            6 => Phase::InitialPlacement { player, placing_second: false, placing_road: true },
            10 => Phase::InitialPlacement { player, placing_second: true, placing_road: false },
            11 => Phase::InitialPlacement { player, placing_second: true, placing_road: true },
            15 => Phase::Turn { player, turn_phase: TurnPhase::PreRoll, development_phase: current_development_phase },
            20 => Phase::Turn { player, turn_phase: TurnPhase::Free, development_phase: current_development_phase },
            33 => Phase::Turn { player, turn_phase: TurnPhase::MoveThief, development_phase: current_development_phase },
            40 => Phase::Turn { player, turn_phase: current_turn_phase, development_phase: DevelopmentPhase::RoadBuildingActive { two_left: true } },
            41 => Phase::Turn { player, turn_phase: current_turn_phase, development_phase: DevelopmentPhase::RoadBuildingActive { two_left: false } },
            _ => Phase::Turn { player, turn_phase: current_turn_phase, development_phase: current_development_phase },//仮の措置
        };
    }

    pub fn set_development_played(&mut self) {
        if let Phase::Turn { player: _, turn_phase: _, development_phase } = &mut self.phase {
            if let DevelopmentPhase::RoadBuildingActive { two_left } = development_phase {
                if *two_left {
                    *development_phase = DevelopmentPhase::RoadBuildingActive { two_left: false };
                } else {
                    *development_phase = DevelopmentPhase::DevelopmentPlayed;
                }
            }
        }
    }

    pub fn update_player_turn(&mut self, player: i32) {
        self.trade_count = 0;

        // development card の更新
        let prev_player = self.phase.player();
        let hand = self.state.get_player_hand_mut(prev_player);
        hand.development_cards += hand.new_development_cards;
        hand.new_development_cards.clear();

        self.phase = Phase::Turn { player: PlayerId::from(player as u8), turn_phase: TurnPhase::PreRoll, development_phase: DevelopmentPhase::Ready };
    }

    pub fn set_discard_phase(&mut self) { // 自分のdiscardフェーズにする
        if let Phase::Turn { player: _, turn_phase, development_phase: _ } = &mut self.phase {
            *turn_phase = TurnPhase::Discard(self.player_id);
        }
    }

    pub fn execute_discard(&mut self, player: i32, res_changes: Vec<i32>) {
        let player = if player >= 0 { player } else { self.player_id.to_u8() as i32 };
        let player_id = PlayerId::from(player as u8);
        let mut discard_res = Resources::ZERO;
        for res in 1..=5 {
            let resource = js_res_to_rust_res(res);
            discard_res[resource] = res_changes[(res - 1) as usize] as i8;
        }
        let resource = Resource::Brick; // ダミー
        discard_res[resource] += res_changes[5] as i8; // ジョーカー分
        let player_hand = self.state.get_player_hand_mut(player_id);
        player_hand.resources -= discard_res;
        self.resolve_hand(player_id);
    }

    pub fn execute_trade(&mut self, partner: i32, player: i32, offer: Vec<i32>, request: Vec<i32>, is_bank: bool) {
        if !is_bank{
            let partner_id = PlayerId::from(partner as u8);
            let player_id = PlayerId::from(player as u8);
            let mut offer_res = Resources::ZERO;
            let mut request_res = Resources::ZERO;
            for res in 1..=5 {
                let resource = js_res_to_rust_res(res);
                offer_res[resource] = offer[(res - 1) as usize] as i8;
                request_res[resource] = request[(res - 1) as usize] as i8;
            }
            let diff = request_res - offer_res;
            let player_hand = self.state.get_player_hand_mut(player_id);
            player_hand.resources += diff;
            let partner_hand = self.state.get_player_hand_mut(partner_id);
            partner_hand.resources -= diff;
            self.resolve_hand(player_id);
            self.resolve_hand(partner_id);
        }
        else {
            let player_id = PlayerId::from(player as u8);
            let mut offer_res = Resources::ZERO;
            let mut request_res = Resources::ZERO;
            for res in 1..=5 {
                let resource = js_res_to_rust_res(res);
                offer_res[resource] = offer[(res - 1) as usize] as i8;
                request_res[resource] = request[(res - 1) as usize] as i8;
            }
            let diff = request_res - offer_res;
            let player_hand = self.state.get_player_hand_mut(player_id);
            player_hand.resources += diff;
            // jsettlerではbankの資源管理をしていないのでここでは変更しない
            // let bank_resources = self.state.get_bank_resources_mut();
            // *bank_resources -= diff;
            self.resolve_hand(player_id);
        }
    }

    pub fn set_trade_offer(&mut self, player: i32, offer: Vec<i32>, request: Vec<i32>) {// playerはofferを出したプレイヤー
        let player_id = PlayerId::from(player as u8);
        let mut offer_res = Resources::ZERO;
        let mut request_res = Resources::ZERO;
        for res in 1..=5 {
            let resource = js_res_to_rust_res(res);
            offer_res[resource] = offer[(res - 1) as usize] as i8;
            request_res[resource] = request[(res - 1) as usize] as i8;
        }
        self.state.set_trade_info(offer_res, request_res, player_id, self.player_id);
        if let Phase::Turn { player: _, turn_phase, development_phase: _ } = &mut self.phase {
            *turn_phase = TurnPhase::TradeSupposed(self.player_id)
        }
    }

    pub fn update_develop_card(&mut self, player: i32, action: i32, card: i32) {
        let player_id = PlayerId::from(player as u8);
        let player_hand = self.state.get_player_hand_mut(player_id);
        let dev_card = match card {
            1 => Some(DevelopmentCard::RoadBuilding),
            2 => Some(DevelopmentCard::YearOfPlenty),
            3 => Some(DevelopmentCard::Monopole),
            4 | 5 | 6 | 7 | 8 => Some(DevelopmentCard::VictoryPoint), // VPはまとめて扱う
            9 => Some(DevelopmentCard::Knight),
            0 => Some(DevelopmentCard::Knight), // Unknownの場合、便宜上Knightとして扱う
            _ => {
                eprintln!("Unknown card type: {}", card);
                None
            }
        };

        match action {
            0 => { // DRAW (購入)
                if let Some(c) = dev_card {
                    // 種類がわかっている場合（自分）
                    player_hand.new_development_cards[c] += 1;
                    // state側の在庫を減らす
                    let dev_cards = self.state.get_development_cards_mut();
                    dev_cards[c] -= 1; // resolveでマイナスはチェック
                }
            }
            1 => { // PLAY (使用)
                if let Some(c) = dev_card {
                    // 古いカードから減らす
                    player_hand.development_cards[c] -= 1;
                    // 騎士の使用回数カウントは, 1024のときに行う
                }
            }
            2 => { // ADD_NEW
                if let Some(c) = dev_card {
                    player_hand.new_development_cards[c] += 1;
                }
            }
            3 => { // ADD_OLD
                if let Some(c) = dev_card {
                    player_hand.development_cards[c] += 1;
                }
            }
            _ => {
                eprintln!("Invalid action for development card: {}", action);
            }
        }
        self.resolve_development_cards(player_id);
    }

    pub fn update_board_layout(&mut self, hl: Vec<i32>, nl: Vec<i32>, rh :i32) {
        for i in 0..hl.len() {
            let coord_tuple = JS_INDEX_TO_RUST_HEX[i];
            let coord = Coord { x: coord_tuple.0, y: coord_tuple.1 };
            match hl[i] {
                0 => {//desert
                    self.state.set_static_hex(coord, Hex::Land(LandHex::Desert)).unwrap();
                }
                1 => {
                    self.state.set_static_hex(coord, Hex::Land(LandHex::Prod(Resource::Brick, nl[i] as u8))).unwrap();
                }
                2 => {
                    self.state.set_static_hex(coord, Hex::Land(LandHex::Prod(Resource::Ore, nl[i] as u8))).unwrap();
                }
                3 => {
                    self.state.set_static_hex(coord, Hex::Land(LandHex::Prod(Resource::Wool, nl[i] as u8))).unwrap();
                }
                4 => {
                    self.state.set_static_hex(coord, Hex::Land(LandHex::Prod(Resource::Grain, nl[i] as u8))).unwrap();
                }
                5 => {
                    self.state.set_static_hex(coord, Hex::Land(LandHex::Prod(Resource::Lumber, nl[i] as u8))).unwrap();
                }
                6 => (), //water
                _ => {//port
                    if hl[i] >= 7 && hl[i] <= 12 {
                        let direction = js_index_to_direction(hl[i] - 6);
                        //harbor edgeの座標
                        let harbor_coord = Coord { x: coord.x + direction.0, y: coord.y + direction.1 };
                        for intersection_coord in self.state.path_intersection_neighbours(harbor_coord).expect("Wrong path").iter() {
                            self.state.set_static_harbor(*intersection_coord, Harbor::Generic)
                            .expect("Failed setting harbor");
                        }
                    }
                    else {//下位4bitが資源, 上位bitがdirection
                        let resource_id = hl[i] & 0x0F;
                        let direction_id = hl[i] >> 4;
                        let resource = js_res_to_rust_res(resource_id);
                        let direction = js_index_to_direction(direction_id);
                        let harbor_coord = Coord { x: coord.x + direction.0, y: coord.y + direction.1 };
                        for intersection_coord in self.state.path_intersection_neighbours(harbor_coord).expect("Wrong path").iter() {
                            self.state.set_static_harbor(*intersection_coord, Harbor::Special(resource))
                            .expect("Failed setting harbor");
                        }
                    }
                }
            }
        }
        if rh != -1 {
            let thief_coord = js_hex_to_rust_coord(rh);
            self.state.set_thief_hex(thief_coord);
        }
    }

    pub fn update_resources(&mut self, player: i32, action: i32, res_changes: Vec<i32>) {
        let player_id = PlayerId::from(player as u8);
        let player_hand = self.state.get_player_hand_mut(player_id);
        // let mut bank_change = Resources::ZERO;
        for res in 1..=5 {
            let resource = js_res_to_rust_res(res);
            match action {
                100 => { // set
                    player_hand.resources[resource] = res_changes[(res - 1) as usize] as i8;
                }
                101 => { // add
                    player_hand.resources[resource] += res_changes[(res - 1) as usize] as i8;
                    // bank_change[resource] -= res_changes[(res - 1) as usize] as i8;
                }
                102 => { // remove
                    player_hand.resources[resource] -= res_changes[(res - 1) as usize] as i8;
                    // bank_change[resource] += res_changes[(res - 1) as usize] as i8;
                }
                _ => {
                    panic!("Invalid action for resource: {}", action);
                }
            }          
        }
        self.resolve_hand(player_id);
        // let bank_resources = self.state.get_bank_resources_mut();
        // *bank_resources += bank_change;
    }

    pub fn execute_robbery(&mut self, player: i32, victim: i32, res_type: i32) {
        let player_id = PlayerId::from(player as u8);
        let victim_id = PlayerId::from(victim as u8);
        if res_type != 6 {
            let resource = js_res_to_rust_res(res_type);
            let victim_hand = self.state.get_player_hand_mut(victim_id);
            victim_hand.resources[resource] -= 1;
            let player_hand = self.state.get_player_hand_mut(player_id);
            player_hand.resources[resource] += 1;
        } else {
            // とりあえずbrickを奪うことにする
            let resource = Resource::Brick;
            let victim_hand = self.state.get_player_hand_mut(victim_id);
            victim_hand.resources[resource] -= 1;
            let player_hand = self.state.get_player_hand_mut(player_id);
            player_hand.resources[resource] += 1;
        }
        self.resolve_hand(victim_id);
    }

    pub fn increment_trade_count(&mut self) {
        self.trade_count += 1;
    }
        
    pub fn to_observation(&self, py: Python) -> PyResult<PyObject> {
        let mut normal_legal_actions = Array1::from_elem(self.normal_action_length + 1, false); // +1 for trade actions
        let mut trade_legal_actions = Array1::from_elem(self.trade_length, false);

        let trade_allowed = self.trade_activated && self.trade_count < self.trade_limit;
        for (i, action) in self.possible_actions.iter().enumerate() {
            if legal::legal(&self.phase, &self.state, *action, trade_allowed).is_ok() {
                if i < self.normal_action_length {
                    normal_legal_actions[i] = true;
                } else {
                    trade_legal_actions[i - self.normal_action_length] = true;
                }
            }
        }
        if trade_allowed && self.state.get_player_hand(self.player_id).resources.total() > 0 {
            if let Phase::Turn { player: _, turn_phase, development_phase} = &self.phase {
                if *turn_phase == TurnPhase::Free && (*development_phase == DevelopmentPhase::Ready || *development_phase == DevelopmentPhase::DevelopmentPlayed) {
                    normal_legal_actions[self.normal_action_length] = true; // Trade action
                }
            }
        }
        let observation = PyCatanObservation::new_array(PyObservationFormat::default(), self.player_id, &self.state, &self.phase, normal_legal_actions, trade_legal_actions, trade_allowed);
        let elements: Vec<PyObject> = vec![observation.board.into_pyarray(py).into(), observation.flat.into_pyarray(py).into(), observation.normal_actions.into_pyarray(py).into(), observation.trade_actions.into_pyarray(py).into()];
        Ok(elements.into_pyobject(py).unwrap().unbind().into_any())
    }

    pub fn decode_action(&self, py: Python, action_index: usize) -> PyResult<PyObject> {
        let action = self.possible_actions[action_index];
        let action_type = action.category() as i32;
        let mut param0 = -1;
        let mut param1 = -1;
        let mut res0 = Array1::from_elem(5, 0i32);
        let mut res1 = Array1::from_elem(5, 0i32);

        match action {
            Action::EndTurn => {}
            Action::RollDice => {}
            Action::MoveThief { hex, victim } => {
                param0 = rust_coord_to_js_hex(&hex);
                if victim != self.player_id {
                    param1 = victim.to_u8() as i32;
                }
                // 資源を持っていない相手を対象に指定するとjsettlerで受理されない
                // ので、その場合は他の候補を探す (もし他にいなければ-1のまま)
                let victim_hand = self.state.get_player_hand(victim);
                if victim_hand.resources.total() == 0 {
                    param1 = -1;
                    // 他にvictimの候補がいないか調べる
                    for intersection in self.state.hex_intersection_neighbours(hex).expect("Hex neighbour error").iter() {
                        if let Some((owner, _)) = self.state.get_dynamic_intersection(*intersection).unwrap() {
                            if owner != self.player_id {
                                let other_victim_hand = self.state.get_player_hand(owner);
                                if other_victim_hand.resources.total() > 0 {
                                    param1 = owner.to_u8() as i32;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            Action::BuildRoad { path } => {
                param0 = rust_coord_to_js_edge(&path);
            }
            Action::BuildSettlement { intersection } => {
                param0 = rust_coord_to_js_intersection(&intersection);
            }
            Action::BuildCity { intersection } => {
                param0 = rust_coord_to_js_intersection(&intersection);
            }
            Action::TradeBank { given, asked } => {
                let hand = self.state.get_player_hand(self.player_id);
                let given_count = hand.harbor.rate(given) as i8;
                res0[(rust_res_to_js_res(given) - 1) as usize] = given_count as i32;
                res1[(rust_res_to_js_res(asked) - 1) as usize] = 1;
            }
            Action::TradePlayers { offer, want, partner } => {
                for res in Resource::ALL.iter() {
                    let js_res_index = rust_res_to_js_res(*res);
                    res0[(js_res_index - 1) as usize] = offer[*res] as i32;
                    res1[(js_res_index - 1) as usize] = want[*res] as i32;
                }
                param0 = partner.to_u8() as i32;
            }
            Action::TradePlayersAccept => {
                if let Phase::Turn { player, turn_phase: _, development_phase: _ } = &self.phase {
                    param0 = player.to_u8() as i32;
                }
            }
            Action::TradePlayersDecline => {}
            Action::BuyDevelopment => {}
            Action::DevelopmentKnight => {}
            Action::DevelopmentRoadBuilding => {}
            Action::DevelopmentYearOfPlenty { resources } => {
                for res in Resource::ALL.iter() {
                    let js_res_index = rust_res_to_js_res(*res);
                    res0[(js_res_index - 1) as usize] = resources[*res] as i32;
                }
            }
            Action::DevelopmentMonopole { resource } => {
                let js_res_index = rust_res_to_js_res(resource);
                param0 = js_res_index;
            }
            Action::Keep {resources: kept} => {//足りていない場合があるので, その場合はrandomに選ぶ
                let current = self.state.get_player_hand(self.player_id).resources;
                let should_discard = current.total() / 2;
                let mut discarded = current - kept;
                let mut total_discards = discarded.total();
                let mut rng = rand::rng();
                while total_discards > should_discard {
                    // Randomly keep cards
                    let mut picked = rng.random_range(0..total_discards);
                    for res in Resource::ALL.iter() {
                        if picked < discarded[*res] {
                            discarded[*res] -= 1;
                            break;
                        } else {
                            picked -= discarded[*res];
                        }
                    }
                    total_discards -= 1;
                }
                for res in Resource::ALL.iter() {
                    let js_res_index = rust_res_to_js_res(*res);
                    res0[(js_res_index - 1) as usize] = discarded[*res] as i32;
                }
            }
            _ => {}
        }
        let elements: Vec<PyObject> = vec![
            action_type.into_pyobject(py).unwrap().unbind().into_any(),
            param0.into_pyobject(py).unwrap().unbind().into_any(),
            param1.into_pyobject(py).unwrap().unbind().into_any(),
            res0.into_pyarray(py).into(),
            res1.into_pyarray(py).into(),
        ];
        Ok(elements.into_pyobject(py).unwrap().unbind().into_any())
    }
}

impl PyTricellState {
    fn resolve_hand(&mut self, player_id: PlayerId) { 
        //手札は常に総数は合っているが, マイナスのものが生まれることがある
        //その場合, ゼロにしてプラスの資源から引く
        let mut negative_sum = 0;
        let player_hand = self.state.get_player_hand_mut(player_id);
        for res in Resource::ALL.iter() {
            if player_hand.resources[*res] < 0 {
                negative_sum += -player_hand.resources[*res] as i32;
                player_hand.resources[*res] = 0;
            }
        }
        if negative_sum > 0 {
            for res in Resource::ALL.iter() {
                if player_hand.resources[*res] > 0 {
                    let take_amount = std::cmp::min(player_hand.resources[*res] as i32, negative_sum);
                    player_hand.resources[*res] -= take_amount as i8;
                    negative_sum -= take_amount;
                    if negative_sum == 0 {
                        break;
                    }
                }
            }
        }
    }

    fn resolve_development_cards(&mut self, player_id: PlayerId) {
        let mut negative_sum = 0;
        let player_hand = self.state.get_player_hand_mut(player_id);
        for card in DevelopmentCard::ALL.iter() {
            if player_hand.development_cards[*card] < 0 {
                negative_sum += -(player_hand.development_cards[*card] as i32);
                player_hand.development_cards[*card] = 0;
            }
        }
        if negative_sum > 0 {
            for card in DevelopmentCard::ALL.iter() {
                if player_hand.development_cards[*card] > 0 {
                    let take_amount = std::cmp::min(player_hand.development_cards[*card] as i32, negative_sum);
                    player_hand.development_cards[*card] -= take_amount as i8;
                    negative_sum -= take_amount;
                    if negative_sum == 0 {
                        break;
                    }
                }
            }
        }

        // 在庫も同様に確認, 隠れ情報なので適当でいいが, 数は合うようにする
        let mut negative_sum = 0;
        let development_cards = self.state.get_development_cards_mut();
        for card in DevelopmentCard::ALL.iter() {
            if development_cards[*card] < 0 {
                negative_sum += -(development_cards[*card] as i32);
                development_cards[*card] = 0;
            }
        }
        if negative_sum > 0 {
            for card in DevelopmentCard::ALL.iter() {
                if development_cards[*card] > 0 {
                    let take_amount = std::cmp::min(development_cards[*card] as i32, negative_sum);
                    development_cards[*card] -= take_amount as i8;
                    negative_sum -= take_amount;
                    if negative_sum == 0 {
                        break;
                    }
                }
            }
        }
    }
}