# rust-catan-rl
A Rust implementation for the board game Catan

This is a fork of [Swynfel's rust-catan](https://github.com/Swynfel/rust-catan), enhanced with custom utilities based on our research. Most notably, you can play Catan against our powerful AI agents, which were trained entirely through end-to-end Reinforcement Learning (RL). A paper detailing our research and methodology is currently in preparation.

## Prerequisites

Before running the project, ensure you have the following installed:
- **Rust**: stable toolchain 1.91+ (verified with Rust/Cargo 1.91.1)
- **Python**: 3.12 (managed via `pyproject.toml`, `uv` is highly recommended)
- **Java JRE/JDK**: 11+ (verified with OpenJDK 11.0.29)

## Quick Start

Getting started is very simple. Just execute the following command in your terminal to launch the JSettlers GUI and start playing:

```bash
maturin develop --release
play-catan
```

After the JSettlers GUI opens, fill in your nickname, join the game, and sit down in an empty seat.

![JSettlers GUI Gameplay against RL Agent](assets/gameplay.png)

## Training

python/train.py is our RL code. 

```bash
cd python
python train.py --use_archive
```

## Evaluation

To evaluate against JSettlers agents, execute

```bash
cd python/jsettlers
python run_parallel_experiment.py --workers 50 --gpus 0,1 --model [model_path] --games 10000 --result_dir [optional] --id default --trade
```
and the result json file is generated.

## License and Acknowledgments

The core code of this project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

This project integrates and communicates with the following open-source software. We are deeply grateful to their developers:

**Rust-catan**: Originally created by Swynfel, licensed under the [MIT License](LICENSE.rust_catan).

**JSettlers2**: Used as an independent graphical frontend and client-server component. Licensed under the [GNU General Public License (GPL)](external/jsettlers/COPYING-GPLv3.txt).

Please note that while this main repository is provided under the MIT License, the JSettlers component retains its original GPL license to comply with its terms.
