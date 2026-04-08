use ndarray::Array1;
use ndarray::Array2;
use ndarray::Array4;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use numpy::convert::IntoPyArray;
use rayon::prelude::*;

use catan::player::{generate_possible_actions_without_state};
use catan::state::PlayerId;
use catan::board::layout;
use super::{PyCatanObservation, PyObservationFormat};
use super::environment::{SingleEnvironment, MultiEnvironment};

#[pyclass]
pub struct SingleVecEnvironment {
    envs: Vec<SingleEnvironment>,
    format: PyObservationFormat,
    player_count: usize,
    normal_action_dim: usize,
    trade_action_dim: usize,
}

#[pymethods]
impl SingleVecEnvironment {
    #[staticmethod]
    #[pyo3(signature=(num_envs, format, opponents=3, trade_activated=false, trade_limit=1))]
    fn new(num_envs: usize, format: &PyObservationFormat, opponents: usize, trade_activated: bool, trade_limit: u8) -> Self {
        let mut possible_actions = Vec::new();
        let player_count = (opponents + 1) as u8;
        generate_possible_actions_without_state(&mut possible_actions, PlayerId::from(0u8), player_count, &layout::DEFAULT, true, true);
        let mut without_trade_actions = Vec::new();
        generate_possible_actions_without_state(&mut without_trade_actions, PlayerId::from(0u8), player_count, &layout::DEFAULT, false, true);

        let total_action_dim = possible_actions.len();
        let without_trade_action_dim = without_trade_actions.len();
        let envs = (0..num_envs).map(|_| {
            SingleEnvironment::new(format, opponents, trade_activated, trade_limit)
        }).collect();
        SingleVecEnvironment { 
            envs,
            format: format.clone(),
            player_count: player_count as usize,
            normal_action_dim: without_trade_action_dim + 1, // +1 for trade action
            trade_action_dim: total_action_dim - without_trade_action_dim,
        }
    }

    fn play(&mut self, py: Python, actions: Vec<u16>) -> PyResult<PyObject> {
        if actions.len() != self.envs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("actions length must match number of environments"));
        }

        // 並列処理,
        let results: Result<Vec<(bool, Option<(u8, PyCatanObservation)>)>, _> = 
            self.envs.par_iter_mut()
                .zip(&actions)
                .map(|(env, &action)| {
                    Ok(env.play_internal(action))
                })
                .collect::<Result<Vec<(bool, Option<(u8, PyCatanObservation)>)>, pyo3::PyErr>>();
        let results = results?;

        let batch_size = self.envs.len();

        let mut out_boards = Array4::<i8>::zeros((batch_size, self.format.width,self.format.height, 13 + 2 * self.player_count));
        let mut out_flats = Array2::<i8>::zeros((batch_size, 29+self.player_count*10));
        let mut out_hidden = Array2::<i8>::zeros((batch_size, (self.player_count - 1)*29));
        let mut out_normal_masks = Array2::<bool>::default((batch_size, self.normal_action_dim));
        let mut out_trade_masks = Array2::<bool>::default((batch_size, self.trade_action_dim));

        let mut out_meta = Array2::<i8>::zeros((batch_size, 3));

        let results_arr = Array1::from_vec(results);

        ndarray::Zip::from(out_boards.outer_iter_mut())
            .and(out_flats.outer_iter_mut())
            .and(out_hidden.outer_iter_mut())
            .and(out_normal_masks.outer_iter_mut())
            .and(out_trade_masks.outer_iter_mut())
            .and(&results_arr)
            .par_for_each(|mut board, mut flat, mut hidden, mut normal_mask, mut trade_mask, res| {
                let (_, obs_opt) = res;
                
                if let Some((_, obs)) = obs_opt {
                    board.assign(&obs.board);
                    flat.assign(&obs.flat);
                    normal_mask.assign(&obs.normal_actions);
                    trade_mask.assign(&obs.trade_actions);
                    
                    if self.format.include_hidden {
                        if let Some(h_src) = &obs.hidden {
                            hidden.assign(h_src);
                        }
                    }
                }
            });

        // --- ループ 2: メタデータの更新 ---
        ndarray::Zip::from(out_meta.outer_iter_mut())
            .and(&results_arr)
            .par_for_each(|mut meta, res| {
                let (_, obs_opt) = res;

                if let Some((p_id, obs)) = obs_opt {
                    meta[0] = *p_id as i8;
                    meta[1] = obs.position as i8;
                    meta[2] = 0; // Done = False
                } else {
                    // Doneの場合
                    meta[0] = 0;
                    meta[1] = -1;
                    meta[2] = 1; // Done = True
                }
            });

        let py_ids = out_meta.column(0).to_owned().into_pyarray(py).into();
        let py_pos = out_meta.column(1).to_owned().into_pyarray(py).into();
        
        // Doneフラグは bool に戻す (i32 -> bool)
        let py_dones = out_meta.column(2).mapv(|v| v != 0).into_pyarray(py).into();

        // Hiddenは有効なら配列、無効ならNone
        let py_hidden = if self.format.include_hidden{
            out_hidden.into_pyarray(py).into()
        } else {
            py.None()
        };

        let tup = if self.format.include_hidden {
            PyTuple::new(py, &[py_ids, out_boards.into_pyarray(py).into(), out_flats.into_pyarray(py).into(), py_hidden, out_normal_masks.into_pyarray(py).into(), out_trade_masks.into_pyarray(py).into(), py_pos, py_dones])?
        } else {
            PyTuple::new(py, &[py_ids, out_boards.into_pyarray(py).into(), out_flats.into_pyarray(py).into(), out_normal_masks.into_pyarray(py).into(), out_trade_masks.into_pyarray(py).into(), py_pos, py_dones])?
        };
        Ok(tup.into())
    }

    fn start(&mut self, py: Python, env_index: usize) -> PyResult<PyObject> {
        if env_index >= self.envs.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err("env_index out of range"));
        }
        self.envs[env_index].start(py)
    }

    fn close(&mut self) {
        for env in &mut self.envs {
            env.close();
        }
    }

    fn reset(&mut self, py: Python, env_index: usize) -> PyResult<PyObject> {
        if env_index >= self.envs.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err("env_index out of range"));
        }
        self.envs[env_index].reset(py)
    }

    fn result(&mut self, py: Python, env_index: usize) -> PyResult<(u8,bool)> {
        if env_index >= self.envs.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err("env_index out of range"));
        }
        self.envs[env_index].result(py)
    }

    fn export_history(&mut self, py: Python, env_index: usize) -> PyResult<PyObject> {
        if env_index >= self.envs.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err("env_index out of range"));
        }
        self.envs[env_index].export_history(py)
    }
}

#[pyclass]
pub struct MultiVecEnvironment {
    envs: Vec<MultiEnvironment>,
    format: PyObservationFormat,
    player_count: usize,
    normal_action_dim: usize,
    trade_action_dim: usize,
}

#[pymethods]
impl MultiVecEnvironment {
    #[staticmethod]
    #[pyo3(signature=(num_envs, format, players=4, trade_activated=false, trade_limit=1))]
    fn new(num_envs: usize, format: &PyObservationFormat, players: usize, trade_activated: bool, trade_limit: u8) -> Self {
        let mut possible_actions = Vec::new();
        let player_count = (players) as u8;
        generate_possible_actions_without_state(&mut possible_actions, PlayerId::from(0u8), player_count, &layout::DEFAULT, true, true);
        let mut without_trade_actions = Vec::new();
        generate_possible_actions_without_state(&mut without_trade_actions, PlayerId::from(0u8), player_count, &layout::DEFAULT, false, true);

        let total_action_dim = possible_actions.len();
        let without_trade_action_dim = without_trade_actions.len();
        let envs = (0..num_envs).map(|_| {
            MultiEnvironment::new(format, players, trade_activated, trade_limit)
        }).collect();
        MultiVecEnvironment { 
            envs,
            format: format.clone(),
            player_count: players,
            normal_action_dim: without_trade_action_dim + 1, // +1 for trade action
            trade_action_dim: total_action_dim - without_trade_action_dim,
        }
    }

    fn play(&mut self, py: Python, players: Vec<u8>, actions: Vec<u16>) -> PyResult<PyObject> {
        if actions.len() != self.envs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("actions length must match number of environments"));
        }

        // 並列処理,
        let results: Result<Vec<(bool, Option<(u8, PyCatanObservation)>)>, _> = 
            self.envs.par_iter_mut()
                .zip(&players)
                .zip(&actions)
                .map(|((env, &player), &action)| {
                    Ok(env.play_internal(player, action))
                })
                .collect::<Result<Vec<(bool, Option<(u8, PyCatanObservation)>)>, pyo3::PyErr>>();
        let results = results?;

        let batch_size = self.envs.len();

        let mut out_boards = Array4::<i8>::zeros((batch_size, self.format.width,self.format.height, 13 + 2 * self.player_count));
        let mut out_flats = Array2::<i8>::zeros((batch_size, 29+self.player_count*10));
        let mut out_hidden = Array2::<i8>::zeros((batch_size, (self.player_count - 1)*29));
        let mut out_normal_masks = Array2::<bool>::default((batch_size, self.normal_action_dim));
        let mut out_trade_masks = Array2::<bool>::default((batch_size, self.trade_action_dim));

        let mut out_meta = Array2::<i8>::zeros((batch_size, 3));

        let results_arr = Array1::from_vec(results);

        ndarray::Zip::from(out_boards.outer_iter_mut())
            .and(out_flats.outer_iter_mut())
            .and(out_hidden.outer_iter_mut())
            .and(out_normal_masks.outer_iter_mut())
            .and(out_trade_masks.outer_iter_mut())
            .and(&results_arr)
            .par_for_each(|mut board, mut flat, mut hidden, mut normal_mask, mut trade_mask, res| {
                let (_, obs_opt) = res;
                
                if let Some((_, obs)) = obs_opt {
                    board.assign(&obs.board);
                    flat.assign(&obs.flat);
                    normal_mask.assign(&obs.normal_actions);
                    trade_mask.assign(&obs.trade_actions);
                    
                    if self.format.include_hidden {
                        if let Some(h_src) = &obs.hidden {
                            hidden.assign(h_src);
                        }
                    }
                }
            });

        // --- ループ 2: メタデータの更新 ---
        ndarray::Zip::from(out_meta.outer_iter_mut())
            .and(&results_arr)
            .par_for_each(|mut meta, res| {
                let (_, obs_opt) = res;

                if let Some((p_id, obs)) = obs_opt {
                    meta[0] = *p_id as i8;
                    meta[1] = obs.position as i8;
                    meta[2] = 0; // Done = False
                } else {
                    // Doneの場合
                    meta[0] = 0;
                    meta[1] = -1;
                    meta[2] = 1; // Done = True
                }
            });
        let py_ids = out_meta.column(0).to_owned().into_pyarray(py).into();
        let py_pos = out_meta.column(1).to_owned().into_pyarray(py).into();
        
        // Doneフラグは bool に戻す (i32 -> bool)
        let py_dones = out_meta.column(2).mapv(|v| v != 0).into_pyarray(py).into();

        // Hiddenは有効なら配列、無効ならNone
        let py_hidden = if self.format.include_hidden{
            out_hidden.into_pyarray(py).into()
        } else {
            py.None()
        };

        let tup = if self.format.include_hidden {
            PyTuple::new(py, &[py_ids, out_boards.into_pyarray(py).into(), out_flats.into_pyarray(py).into(), py_hidden, out_normal_masks.into_pyarray(py).into(), out_trade_masks.into_pyarray(py).into(), py_pos, py_dones])?
        } else {
            PyTuple::new(py, &[py_ids, out_boards.into_pyarray(py).into(), out_flats.into_pyarray(py).into(), out_normal_masks.into_pyarray(py).into(), out_trade_masks.into_pyarray(py).into(), py_pos, py_dones])?
        };
        Ok(tup.into())
    }

    fn start(&mut self, py: Python, env_index: usize) -> PyResult<PyObject> {
        if env_index >= self.envs.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err("env_index out of range"));
        }
        self.envs[env_index].start(py)
    }

    fn close(&mut self) {
        for env in &mut self.envs {
            env.close();
        }
    }

    fn reset(&mut self, py: Python, env_index: usize, player: u8) -> PyResult<PyObject> {
        if env_index >= self.envs.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err("env_index out of range"));
        }
        self.envs[env_index].reset(py, player)
    }

    fn result(&mut self, py: Python, env_index: usize) -> PyResult<(PyObject, u8)>  {
        if env_index >= self.envs.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err("env_index out of range"));
        }
        self.envs[env_index].result(py)
    }

    fn export_history(&mut self, py: Python, env_index: usize) -> PyResult<PyObject> {
        if env_index >= self.envs.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err("env_index out of range"));
        }
        self.envs[env_index].export_history(py)
    }
}
