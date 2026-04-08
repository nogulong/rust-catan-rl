use ndarray::Array1;
use ndarray::Array2;
use ndarray::Array3;
use ndarray::Array4;
use numpy::{PyArray2};
use pyo3::prelude::*;
 use pyo3::types::{PyBytes, PyDict, PyTuple, PyList};
 use pyo3::Bound;
use pyo3::IntoPyObjectExt;
use numpy::convert::IntoPyArray;
use std::thread;
use std::sync::mpsc::{channel, Sender, Receiver};
use rayon::prelude::*;

use catan::game::{Game, GameResult};
use catan::game::Action;
use catan::player::{init_global_possible_actions, get_global_possible_actions};
use catan::history::{GameHistory, GameSummary};
use catan::state::PlayerId;
use catan::player::Randomy;
use catan::board::layout;
use catan::utils::{Resource, Resources};
use super::{PyCatanObservation, PyObservationFormat, PythonPlayer};

use std::sync::Mutex;

pub fn to_py_tuple(py: Python, hidden_state: bool, observation: Option<(u8, PyCatanObservation)>) -> PyObject {
    let elements: Vec<PyObject> = if let Some((id, observation)) = observation {
        if hidden_state {
            vec![
                // u8 -> PyInt -> PyObject
                id.into_py_any(py).unwrap(),

                // ndarray -> PyArray -> PyObject
                observation.board.into_pyarray(py).into(),
                observation.flat.into_pyarray(py).into(),
                observation.hidden.unwrap().into_pyarray(py).into(),
                observation.normal_actions.into_pyarray(py).into(),
                observation.trade_actions.into_pyarray(py).into(),
                (observation.position as i8).into_py_any(py).unwrap(),
                false.into_py_any(py).unwrap(), 
            ]
        } else {
            vec![
                id.into_py_any(py).unwrap(),
                observation.board.into_pyarray(py).into(),
                observation.flat.into_pyarray(py).into(),
                observation.normal_actions.into_pyarray(py).into(),
                observation.trade_actions.into_pyarray(py).into(),
                (observation.position as i8).into_py_any(py).unwrap(),
                false.into_py_any(py).unwrap(),
            ]
        }
    } else {
        if hidden_state {
            vec![
                0i8.into_py_any(py).unwrap(),
                py.None(),
                py.None(),
                py.None(),
                py.None(),
                py.None(),
                (-1i8).into_py_any(py).unwrap(),
                true.into_py_any(py).unwrap(),
            ]
        } else {
            vec![
                0i8.into_py_any(py).unwrap(),
                py.None(),
                py.None(),
                py.None(),
                py.None(),
                (-1i8).into_py_any(py).unwrap(),
                true.into_py_any(py).unwrap(),
            ]
        }
    };

    elements.into_pyobject(py).unwrap().unbind().into_any()
}

#[pyclass]
pub struct SingleEnvironment {
    action_sender: Sender<u16>,
    observation_receiver: Mutex<Receiver<Option<(u8, PyCatanObservation)>>>,
    result_receiver: Mutex<Receiver<(u8,bool)>>,
    history_receiver: Mutex<Receiver<(Vec<u8>, GameSummary)>>,
    game_thread: Option<thread::JoinHandle<()>>,
    include_hidden: bool,
}

#[pymethods]
impl SingleEnvironment {

    #[staticmethod]
    #[pyo3(signature = (format, opponents=2, trade_activated=false, trade_limit=1))]
    pub(crate) fn new(format: &PyObservationFormat, opponents: usize, trade_activated: bool, trade_limit: u8) -> SingleEnvironment {
        let format = *format;
        let (action_sender, action_receiver) = channel();
        let (observation_sender, observation_receiver) = channel();
        let (result_sender, result_receiver) = channel();
        let (history_sender, history_receiver) = channel();
        
        init_global_possible_actions((opponents + 1) as u8);

        let game_thread = thread::spawn(move || {
            let mut game = Game::new();
            for _ in 0..opponents {
                game.add_player(Box::new(Randomy::new_player(trade_activated)));
            };
            game.add_player(Box::new(PythonPlayer::new(0, format, action_receiver, observation_sender, result_sender, trade_activated, trade_limit)));
            loop {
                let (history, game_result) = game.setup_and_play();
                match game_result {
                    GameResult::Interrupted => break,
                    _ => (),
                }
                // 履歴をシリアライズして送信 + python側に必要な履歴の詳細を伝える
                let summary = history.get_summary();
                let serialized_record = rmp_serde::to_vec(&history).unwrap();
                let _ = history_sender.send((serialized_record, summary));
            }
        });
        SingleEnvironment {
            action_sender,
            observation_receiver: Mutex::new(observation_receiver),
            result_receiver: Mutex::new(result_receiver),
            history_receiver: Mutex::new(history_receiver),
            game_thread: Some(game_thread),
            include_hidden: format.include_hidden,
        }
    }

    pub(crate) fn start(&mut self, py: Python) -> PyResult<PyObject> {
        Ok(to_py_tuple(py, self.include_hidden, self.observation_receiver.lock().unwrap().recv().expect("Failed to read start observation")))
    }

    fn play(&mut self, py: Python, action: u16) -> PyResult<PyObject> {
        self.action_sender.send(action).expect("Failed to send action");
        if let Some(ref handle) = self.game_thread {
            handle.thread().unpark();
        }
        Ok(to_py_tuple(py, self.include_hidden, self.observation_receiver.lock().unwrap().recv().expect("Failed to read play observation")))
    }

    pub(crate) fn result(&mut self, _py: Python) -> PyResult<(u8,bool)> {
        Ok(self.result_receiver.lock().unwrap().recv().expect("Failed to read results"))
    }

    pub(crate) fn reset(&mut self, py: Python) -> PyResult<PyObject>{
        self.action_sender.send(PythonPlayer::ACTION_RESET_SIGNAL).expect("Failed to send reset action");
        if let Some(ref handle) = self.game_thread {
            handle.thread().unpark();
        }
        Ok(to_py_tuple(py, self.include_hidden, self.observation_receiver.lock().unwrap().recv().expect("Failed to read play observation")))
    }

    pub(crate) fn close(&mut self) {
        self.action_sender.send(PythonPlayer::ACTION_EXIT_SIGNAL).expect("Failed to send exit action");
        if let Some(ref handle) = self.game_thread {
            handle.thread().unpark();
        }
        let _ = self.game_thread.take().expect("Failed to take game thread").join();
    }

    pub(crate) fn export_history(&mut self, py: Python) -> PyResult<PyObject> {
        let (serialized_record, summary) = self.history_receiver.lock().unwrap()
            .recv()
            .expect("Failed to receive history");

        let py_bytes = pyo3::types::PyBytes::new(py, &serialized_record);
        let summary_dict = PyDict::new(py);
        summary_dict.set_item("total_turns", summary.total_turns)?;
        summary_dict.set_item("total_dice_rolls", summary.total_dice_rolls)?;
        summary_dict.set_item("dev_turns", summary.dev_turns)?;
        summary_dict.set_item("mthief_turns", summary.mthief_turns)?;
        summary_dict.set_item("monopoly_turns", summary.monopoly_turns)?;
        summary_dict.set_item("winner", summary.winner)?;

       let py_bytes_obj: PyObject = py_bytes.into_any().unbind();
       let summary_dict_obj: PyObject = summary_dict.into_any().unbind();
       let tup = PyTuple::new(py, &[py_bytes_obj, summary_dict_obj])?;
       Ok(tup.into())
   }
}

impl SingleEnvironment {
    pub(crate) fn play_internal(&mut self, action: u16) -> (bool, Option<(u8, PyCatanObservation)>) {
        self.action_sender.send(action).expect("Failed to send action");
        if let Some(ref handle) = self.game_thread {
            handle.thread().unpark();
        }
        (self.include_hidden, self.observation_receiver.lock().unwrap().recv().expect("Failed to read play observation"))
    }
}


#[pyclass]
pub struct MultiEnvironment {
    players: usize,
    action_senders: Vec<Sender<u16>>,
    observation_receiver: Mutex<Receiver<Option<(u8, PyCatanObservation)>>>,
    result_receivers: Vec<Mutex<Receiver<(u8,bool)>>>,
    history_receiver: Mutex<Receiver<(Vec<u8>, GameSummary)>>,
    game_thread: Option<thread::JoinHandle<()>>,
    include_hidden: bool,
}

#[pymethods]
impl MultiEnvironment {

    #[staticmethod]
    #[pyo3(signature = (format, players=3, trade_activated=false, trade_limit=1))]
    pub(crate)fn new(format: &PyObservationFormat, players: usize, trade_activated: bool, trade_limit: u8) -> MultiEnvironment {
        let format = *format;
        let mut action_senders = Vec::new();
        let mut action_receivers = Vec::new();
        let mut result_senders = Vec::new();
        let mut result_receivers = Vec::new();
        for _ in 0..players {
            let (action_sender, action_receiver) = channel();
            let (result_sender, result_receiver) = channel();
            action_senders.push(action_sender);
            action_receivers.push(action_receiver);
            result_senders.push(result_sender);
            result_receivers.push(result_receiver);
        }
        let (observation_sender, observation_receiver) = channel();
        let (history_sender, history_receiver) = channel();
        init_global_possible_actions(players as u8);
        let game_thread = thread::spawn(move || {
            let mut game = Game::new();
            for (id, (action_receiver, result_sender)) in action_receivers.into_iter().zip(result_senders.into_iter()).enumerate() {
                game.add_player(Box::new(
                    PythonPlayer::new(id as u8, format, action_receiver, observation_sender.clone(), result_sender, trade_activated, trade_limit))
                );
            };
            loop {
                let (history, game_result) = game.setup_and_play();
                match game_result {
                    GameResult::Interrupted => break,
                    _ => (),
                }
                // 履歴をシリアライズして送信 + python側に必要な履歴の詳細を伝える
                let summary = history.get_summary();
                let serialized_record = rmp_serde::to_vec(&history).unwrap();
                let _ = history_sender.send((serialized_record, summary));
            }
        });
        MultiEnvironment {
            players,
            action_senders,
            observation_receiver: Mutex::new(observation_receiver),
            result_receivers: result_receivers.into_iter().map(Mutex::new).collect(),
            history_receiver: Mutex::new(history_receiver),
            game_thread: Some(game_thread),
            include_hidden: format.include_hidden,
        }
    }

    pub(crate) fn start(&mut self, py: Python) -> PyResult<PyObject> {
        Ok(to_py_tuple(py, self.include_hidden, self.observation_receiver.lock().unwrap().recv().expect("Failed to read start observation")))
    }

    fn play(&mut self, py: Python, player: u8, action: u16) -> PyResult<PyObject> {
        self.action_senders[player as usize].send(action).expect("Failed to send action");
        if let Some(ref handle) = self.game_thread {
            handle.thread().unpark();
        }
        Ok(to_py_tuple(py, self.include_hidden, self.observation_receiver.lock().unwrap().recv().expect("Failed to read play observation")))
    }

    pub(crate) fn result(&mut self, py: Python) -> PyResult<(PyObject, u8)> {
        let mut winner = 0;
        let mut vps = Array1::<u8>::zeros(self.players);
        for player in 0..self.players {
            let result = self.result_receivers[player].lock().unwrap().recv().expect("Failed to read results");
            vps[player] = result.0;
            if result.1 {
                winner = player;
            }
        }
        Ok((vps.into_pyarray(py).into_py_any(py).unwrap(), winner as u8))
    }

    pub(crate) fn reset(&mut self, py: Python, player: u8) -> PyResult<PyObject>{
        self.action_senders[player as usize].send(PythonPlayer::ACTION_RESET_SIGNAL).expect("Failed to send reset action");
        if let Some(ref handle) = self.game_thread {
            handle.thread().unpark();
        }
        Ok(to_py_tuple(py, self.include_hidden, self.observation_receiver.lock().unwrap().recv().expect("Failed to read play observation")))
    }

    pub(crate) fn close(&mut self) {
        for sender in &self.action_senders {
            let _ = sender.send(u16::MAX); // Exit action
        }
        if let Some(ref handle) = self.game_thread {
            handle.thread().unpark();
        }
        let _ = self.game_thread.take().expect("Failed to join game thread").join();
    }

    pub(crate) fn export_history(&mut self, py: Python) -> PyResult<PyObject> {
        let (serialized_record, summary) = self.history_receiver.lock().unwrap()
            .recv()
            .expect("Failed to receive history");

        let py_bytes = pyo3::types::PyBytes::new(py, &serialized_record);

        let summary_dict = PyDict::new(py);
        summary_dict.set_item("total_turns", summary.total_turns)?;
        summary_dict.set_item("total_dice_rolls", summary.total_dice_rolls)?;
        summary_dict.set_item("dev_turns", summary.dev_turns)?;
        summary_dict.set_item("mthief_turns", summary.mthief_turns)?;
        summary_dict.set_item("monopoly_turns", summary.monopoly_turns)?;
        summary_dict.set_item("winner", summary.winner)?;

       let py_bytes_obj: PyObject = py_bytes.into_any().unbind();
       let summary_dict_obj: PyObject = summary_dict.into_any().unbind();
       let tup = PyTuple::new(py, &[py_bytes_obj, summary_dict_obj])?;
       Ok(tup.into())
    }
}

impl MultiEnvironment {
    pub(crate) fn play_internal(&mut self, player: u8, action: u16) -> (bool, Option<(u8, PyCatanObservation)>) {
        self.action_senders[player as usize].send(action).expect("Failed to send action");
        if let Some(ref handle) = self.game_thread {
            handle.thread().unpark();
        }
        (self.include_hidden, self.observation_receiver.lock().unwrap().recv().expect("Failed to read play observation"))
    }
}

#[pyfunction(name="display_history")]
pub fn display_history(_py: Python, history_bytes: &Bound<PyBytes>) -> PyResult<()> {
    let slice: &[u8] = history_bytes.as_bytes();
    let history: GameHistory = rmp_serde::from_slice(slice)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("msgpack decode failed: {e}")))?;
    init_global_possible_actions(history.player_count() as u8);
    history.display_test();
    Ok(())
}

pub struct ObservationData {
    pub actions: Array1<bool>,
    pub board: Array3<i8>,
    pub flat: Array1<i8>,
    pub acting_player_id: i8,
    pub selected_action_id: i32,
    pub winner: i8,
    pub turn_index: i32,
    pub action_type: i8,
    pub action_detail: i8,
}
impl ObservationData {
    pub fn to_pyobject(self, py: Python<'_>) -> PyResult<PyObject> {
        let board_obj: PyObject = self.board.into_pyarray(py).into();
        let flat_obj: PyObject = self.flat.into_pyarray(py).into();
        let acting_obj = self.acting_player_id.into_py_any(py)?;
        let selected_obj = self.selected_action_id.into_py_any(py)?;
        let actions_obj = self.actions.into_pyarray(py).into();
        let winner_obj = self.winner.into_py_any(py)?;
        let turn_index_obj = self.turn_index.into_py_any(py)?;
        let action_type_obj = self.action_type.into_py_any(py)?;
        let action_detail_obj = self.action_detail.into_py_any(py)?;

        let tup = PyTuple::new(py, &[board_obj, flat_obj, actions_obj, acting_obj, selected_obj, winner_obj, turn_index_obj, action_type_obj, action_detail_obj])?;
        Ok(tup.into_any().unbind())
    }
}
fn reconstruct_turn_observation_internal(
    history: &GameHistory,
    turn_index: usize,
) -> ObservationData {
    init_global_possible_actions(history.player_count());
    let (state, phase, acting_player_id, selected_action) = history
        .reconstruct_turn(turn_index)
        .expect("turn_index out of range");

    let fmt = PyObservationFormat::default();

    // ボード / フラット生成
    let (obs, selected_action_id) = PyCatanObservation::new_array_and_legal_actions(fmt, acting_player_id, &state, &phase, &selected_action);

    let acting_player_id: i8 = acting_player_id.to_u8() as i8;

    let winner_id: i8 = if let Some(w) = history.get_winner() {
        w.to_u8() as i8
    } else {
        -1
    };
    let board = obs.board;
    let flat = obs.flat;
    let actions = obs.normal_actions;

    let mut action_detail = 0;
    let action_type = match selected_action {
        Action::BuyDevelopment => 1,
        Action::MoveThief{hex:_, victim} => {
            action_detail = victim.to_u8() as i8;
            2
        },
        Action::DevelopmentMonopole{resource} => {
            action_detail = match resource {
                Resource::Brick => 0,
                Resource::Lumber => 1,
                Resource::Ore => 2,
                Resource::Grain => 3,
                Resource::Wool => 4,
            };
            3
        },
        _ => 0,
    };

    ObservationData {
        actions,
        board,
        flat,
        acting_player_id,
        selected_action_id,
        winner: winner_id,
        turn_index: turn_index as i32,
        action_type,
        action_detail,
    }
}

#[pyfunction(name="get_turn_observation")]
pub fn get_turn_observation(// return (board, flat, legal_mask, acting_player_id, selected_action_id, winner)
    py: Python<'_>,
    history_bytes: &Bound<PyBytes>,
    turn_index: usize,
) -> PyResult<PyObject> {
    let slice: &[u8] = history_bytes.as_bytes();
    let history: GameHistory = rmp_serde::from_slice(slice)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("msgpack decode failed: {e}")))?;

    reconstruct_turn_observation_internal(&history, turn_index).to_pyobject(py)
}

#[pyfunction(name="get_turn_observations_parallel")]
pub fn get_turn_observations_parallel(// return list of (board, flat, legal_mask, acting_player_id, selected_action_id, winner)
    py: Python<'_>,
    datas: &Bound<PyList>
) -> PyResult<PyObject> {
    // Python -> Rust の型変換
    let data_items: Result<Vec<(Vec<u8>, usize)>, _> = datas.iter()
        .map(|item| {
            let tuple = item.downcast::<PyTuple>()?;
            let history_bytes = tuple.get_item(0)?
                .downcast::<PyBytes>()?
                .as_bytes()
                .to_vec();  // Send可能なVec<u8>にコピー
            let turn_index = tuple.get_item(1)?.extract::<usize>()?;
            Ok((history_bytes, turn_index))
        })
        .collect::<Result<Vec<(Vec<u8>, usize)>, pyo3::PyErr>>();
    
    let data_items = data_items?;
    let batch_size = data_items.len();
    
    // Rayonを使って並列処理
    let results: Result<Vec<ObservationData>, String> = data_items.par_iter()
        .map(|(history_bytes, turn_index)| {
            let history: GameHistory = rmp_serde::from_slice(history_bytes)
                .map_err(|e| format!("msgpack decode failed: {}", e))?;
            let obs_data = reconstruct_turn_observation_internal(&history, *turn_index);
            Ok(obs_data)
        })
        .collect();
    let observation_datas = results.map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    let sample = &observation_datas[0];
    let board_shape = sample.board.dim(); // (C, H, W)
    let flat_len = sample.flat.len();
    let actions_len = sample.actions.len();

    let mut out_boards = Array4::<i8>::zeros((batch_size, board_shape.0, board_shape.1, board_shape.2));
    let mut out_flats = Array2::<i8>::zeros((batch_size, flat_len));
    let mut out_actions = Array2::<bool>::default((batch_size, actions_len));
    
    let mut out_meta = Array2::<i8>::zeros((batch_size, 4));
    let mut out_meta_i32 = Array2::<i32>::zeros((batch_size, 2));
    let obs_array = Array1::from(observation_datas);

    ndarray::Zip::from(out_boards.outer_iter_mut())
        .and(out_flats.outer_iter_mut())
        .and(out_actions.outer_iter_mut())
        .and(out_meta.outer_iter_mut())
        .and(out_meta_i32.outer_iter_mut())
        .and(&obs_array)
        .par_for_each(|mut board, mut flat, mut action, mut meta, mut meta_i32, obs| {
            board.assign(&obs.board);
            flat.assign(&obs.flat);
            action.assign(&obs.actions);
            meta[0] = obs.acting_player_id;
            meta[1] = obs.winner;
            meta[2] = obs.action_type;
            meta[3] = obs.action_detail;

            meta_i32[0] = obs.selected_action_id as i32;
            meta_i32[1] = obs.turn_index;
        });

    let tuple = PyTuple::new(py, &[
        out_boards.into_pyarray(py).into_any(),
        out_flats.into_pyarray(py).into_any(),
        out_actions.into_pyarray(py).into_any(),
        out_meta.column(0).to_owned().into_pyarray(py).into_any(), // acting
        out_meta_i32.column(0).to_owned().into_pyarray(py).into_any(), // selected
        out_meta.column(1).to_owned().into_pyarray(py).into_any(), // winner
        out_meta_i32.column(1).to_owned().into_pyarray(py).into_any(), // turn
        out_meta.column(2).to_owned().into_pyarray(py).into_any(), // type
        out_meta.column(3).to_owned().into_pyarray(py).into_any(), // detail
    ])?;

    Ok(tuple.into())
}

#[pyfunction(name="generate_board_mask")]//hexcoordinateでのアクションマスクを作る, 無効アクションは-1, 有効アクションはアクションインデックスを入れる, 基本一度しか使わない
pub fn generate_board_mask(
    py: Python<'_>,
    player: usize,
) -> PyResult<PyObject> {
    let format = PyObservationFormat::default();
    let mut board_mask = Array3::<i16>::from_elem((format.width, format.height, player + 3), -1); //MTがPlayer次元ある
    // global_possible_actionsは初期化されていることを仮定
    //layoutはデフォルトを使う

    //MTアクションを埋める
    for coord in layout::DEFAULT.hexes.iter() {
        let (x,y) = format.map(*coord);
        for player_id in 0..player {
            let action = Action::MoveThief {hex: *coord, victim: PlayerId::from(player_id)};
            let action_index = get_global_possible_actions(true).iter().position(|a| *a == action);
            if let Some(index) = action_index {
                board_mask[(x, y, player_id)] = index as i16;
            }
        }
    };

    // Build Pathアクションを埋める
    for coord in layout::DEFAULT.paths.iter() {
        let (x,y) = format.map(*coord);
        let action = Action::BuildRoad {path: *coord};
        let action_index = get_global_possible_actions(true).iter().position(|a| *a == action);
        if let Some(index) = action_index {
            board_mask[(x, y, player)] = index as i16;
        }
    };

    // Build Settlement と Build Cityアクションを埋める
    for coord in layout::DEFAULT.intersections.iter() {
        let (x,y) = format.map(*coord);
        let action = Action::BuildSettlement {intersection: *coord};
        let action_index = get_global_possible_actions(true).iter().position(|a| *a == action);
        if let Some(index) = action_index {
            board_mask[(x, y, player + 1)] = index as i16;
        }
        let action = Action::BuildCity {intersection: *coord};
        let action_index = get_global_possible_actions(true).iter().position(|a| *a == action);
        if let Some(index) = action_index {
            board_mask[(x, y, player + 2)] = index as i16;
        }
    };
    Ok(board_mask.into_pyarray(py).into())
}

#[pyfunction(name = "get_trade_table")]
pub fn get_trade_table<'py>(
    py: Python<'py>,
    player_count: usize,
) -> PyResult<Bound<'py, PyArray2<i8>>> {
    let mut feats_vec:Vec<Vec<i8>> = Vec::new();
    // let mut trade_id = 0;
    let num_opponents = player_count - 1;

    let mut add_recipe_group = |diff: &Resources| {
        let diff_total = -diff.total(); // 相手の資源増減量 (自分が-1なら相手は+1)

        for p_rel in 1..player_count { 
            // 5次元(資源) + 相手OneHot(num_opponents) + 相手増減(num_opponents) 
            let mut feat: Vec<i8> = vec![0; Resource::COUNT + num_opponents * 2];

            // 1. 資源の増減 (自分の視点)
            for r in 0..Resource::COUNT {
                feat[r] = diff[r];
            }

            // 2. 相手のOne-Hot (Offer Player)
            // p_rel = 1 -> index 0
            feat[Resource::COUNT + (p_rel - 1)] = 1;

            // 3. 相手の資源増減量 (Want Player Total Change)
            feat[Resource::COUNT + num_opponents + (p_rel - 1)] = diff_total;

            feats_vec.push(feat);
        }
    };

    // 1 on 1 trade
    for given in Resource::ALL.iter() {
        for asked in Resource::ALL.iter() {
            if given != asked {
                let mut diff = Resources::ZERO;
                diff[*given] -= 1;
                diff[*asked] += 1;
                
                add_recipe_group(&diff);
            }
        }
    }
    // 2 on 1 trade
    for given1 in 0..Resource::COUNT {
        for given2 in given1..Resource::COUNT {
            for asked in 0..Resource::COUNT {
                if given1 != asked && given2 != asked {
                    let mut diff = Resources::ZERO;
                    diff[given1] -= 1;
                    diff[given2] -= 1;
                    diff[asked] += 1;

                    add_recipe_group(&diff);
                }
            }
        }
    }
    // 1 on 2 trade
    for given in 0..Resource::COUNT {
        for asked1 in 0..Resource::COUNT {
            for asked2 in asked1..Resource::COUNT {
                if given != asked1 && given != asked2 {
                    let mut diff = Resources::ZERO;
                    diff[given] -= 1;
                    diff[asked1] += 1;
                    diff[asked2] += 1;

                    add_recipe_group(&diff);
                }
            }
        }
    }
    PyArray2::from_vec2(py, &feats_vec)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}
