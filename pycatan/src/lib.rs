mod environment;
mod vec_env;
mod python_state;
mod python_player;
mod py_catan_observation;
mod py_observation_format;
mod jsettler_state;
mod terminal_game;

use pyo3::prelude::*;

use environment::{SingleEnvironment, MultiEnvironment};
use vec_env::{SingleVecEnvironment, MultiVecEnvironment};
use python_state::PythonState;
use python_player::PythonPlayer;
use py_catan_observation::PyCatanObservation;
pub use py_observation_format::PyObservationFormat;
use jsettler_state::PyTricellState;
use terminal_game::TerminalEnvironment;

#[pymodule]
fn _pycatan(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SingleEnvironment>()?;
    m.add_class::<MultiEnvironment>()?;
    m.add_class::<PyObservationFormat>()?;
    m.add_class::<SingleVecEnvironment>()?;
    m.add_class::<MultiVecEnvironment>()?;
    m.add_class::<PyTricellState>()?;
    m.add_function(wrap_pyfunction!(environment::display_history, m)?)?;
    m.add_function(wrap_pyfunction!(environment::get_turn_observation, m)?)?;
    m.add_function(wrap_pyfunction!(environment::get_turn_observations_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(environment::generate_board_mask, m)?)?;
    m.add_function(wrap_pyfunction!(environment::get_trade_table, m)?)?;
    m.add_class::<TerminalEnvironment>()?;
    Ok(())
}
