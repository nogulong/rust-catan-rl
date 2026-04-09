// python player vs terminal player
use ndarray::Array1;
use pyo3::prelude::*;
 use pyo3::types::{PyDict, PyTuple};
use pyo3::IntoPyObjectExt;
use numpy::convert::IntoPyArray;
use std::thread;
use std::sync::mpsc::{channel, Sender, Receiver};

use catan::game::{Game, GameResult};
use catan::player::{init_global_possible_actions};
use catan::history::{GameSummary};
use catan_player::TerminalPlayer;
use super::{PyCatanObservation, PyObservationFormat, PythonPlayer};
use super::environment::to_py_tuple;

use std::sync::Mutex;

#[pyclass]
pub struct TerminalEnvironment {
    players: usize,
    action_senders: Vec<Sender<u16>>,
    observation_receiver: Mutex<Receiver<Option<(u8, PyCatanObservation)>>>,
    result_receivers: Vec<Mutex<Receiver<(u8,bool)>>>,
    history_receiver: Mutex<Receiver<(Vec<u8>, GameSummary)>>,
    game_thread: Option<thread::JoinHandle<()>>,
    include_hidden: bool,
}

#[pymethods]
impl TerminalEnvironment {

    #[staticmethod]
    #[pyo3(signature = (format, players=3, trade_activated=false, trade_limit=1))]
    pub(crate)fn new(format: &PyObservationFormat, players: usize, trade_activated: bool, trade_limit: u8) -> TerminalEnvironment {
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
            game.add_player(Box::new(TerminalPlayer::new()));
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
        TerminalEnvironment {
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