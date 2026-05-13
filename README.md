# Connect4 ML

Machine learning project for predicting strong Connect 4 moves from board positions

The project uses a UCI Connect 4 midgame dataset. Each position is represented by the move sequence that produced the board, and the target label is the solver recommended `best_move`.

The project also includes an AlphaZero-style self-play pipeline that fine-tunes a convolutional neural network using MCTS-generated data, giving the model coverage over early-game positions not present in the solver dataset. The trained model can play live games autonomously on papergames.io via a browser bridge.

## Requirements

* Python 3.10 or newer
* `pip`
* Optional: GNU Make, if you want to use the `make` commands
* Optional: CUDA-capable GPU for self-play and training (CPU works but is slow)

Python dependencies are listed in `requirements.txt`. The browser bridge additionally requires `playwright` and a Firefox installation via `playwright install firefox`.

## Setup

With Make:

```bash
make setup
```

Without Make:

```bash
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
```

## Windows

Open **PowerShell** or **Command Prompt** in the `483-Connect4-ML/` directory.

**Setup:**

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
```

**Inspect dataset:**

```powershell
.venv\Scripts\python src/check_data.py
```

**Train baseline (logistic regression):**

```powershell
.venv\Scripts\python src/train_baseline.py
```

**Train phase-aware policy (SGD):**

```powershell
.venv\Scripts\python src/train_phase_policy.py ^
  --phase mid ^
  --train-csv data/UCI-Midgame-d30.train.csv ^
  --val-csv data/UCI-Midgame-d30.val.csv
```

Replace `mid` with `early`, `late`, or `all` for other phase windows.

**Train CNN + MCTS (AlphaZero-style):**

```powershell
.venv\Scripts\python src/train_network.py
```

Run one self-play iteration (CPU — slow; use `--device cuda` if GPU available):

```powershell
.venv\Scripts\python src/run_iteration.py ^
  --start-model artifacts/models/connect4_net.pth ^
  --iter 0 --n-iters 1 ^
  --filters 64 --n-residuals 6 ^
  --games-per-iter 300 --simulations 100 ^
  --supervised-csv data/UCI-Midgame-d30.train.csv ^
  --device cpu
```

**Browser bridge (optional):**

```powershell
.venv\Scripts\python -m playwright install firefox
.venv\Scripts\python browser_bridge.py --our-username "Your Username"
```

Outputs (models, metrics, reports) go to `artifacts/` regardless of platform.

## Feature Representation

`src/preprocess.py` converts each move sequence into a `6 x 7` board:

* `1` means player 1
* Negative one means player 2
* `0` means empty
* Row `0` is the bottom row

For training, `src/dataset.py` converts every board into the current player's perspective and flattens it into 42 numeric features. Labels are converted from 1 based columns to 0 based classes in `[0, 6]`.

The dataset loader can also produce legal move masks with shape `(n_samples, 7)`, where `1` means the column is legal and `0` means it is full.

## Common Commands

Inspect the dataset:

```bash
make check-data
```

Or directly:

```bash
python src/check_data.py
```

Run the feature building smoke test:

```bash
make dataset
```

Or directly:

```bash
python src/dataset.py
```

Train and evaluate the baseline logistic regression model:

```bash
make train
```

Or directly:

```bash
python src/train_baseline.py
```

Train and evaluate the random forest experiment:

```bash
python src/train_improved1.py
```

Clean generated local files:

```bash
make clean
```

## Models

### Baseline

`src/train_baseline.py` trains a logistic regression model with:

* `max_iter=200`
* `solver="lbfgs"`
* `random_state=0`

### Improved Experiment

`src/train_improved1.py` trains a random forest classifier with:

* `criterion="entropy"`
* `max_depth=16`
* `random_state=1`

### CNN + MCTS (AlphaZero-style)

`src/train_network.py` trains a convolutional neural network (64 filters, 6 residual blocks) on the solver dataset. The network has a policy head (7-dim move distribution) and a value head (win probability).

`src/self_play.py` uses MCTS with the trained network to generate self-play games covering all plies from move 0. `src/train_selfplay.py` fine-tunes the network on this data mixed with the original solver data to prevent forgetting.

`src/run_iteration.py` orchestrates one full iteration: generate games → fine-tune → log stats.

Run one self-play iteration:

```bash
python src/run_iteration.py \
  --start-model artifacts/models/connect4_net.pth \
  --iter 0 --n-iters 1 \
  --filters 64 --n-residuals 6 \
  --games-per-iter 300 --simulations 100 \
  --supervised-csv data/UCI-Midgame-d30.train.csv \
  --device cuda
```

Check progress across iterations:

```bash
python src/show_progress.py
```

Evaluate a model against random, a previous model, and the C++ solver:

```bash
python src/eval_winrate.py \
  --model artifacts/models/selfplay_iter1.pth \
  --prev-model artifacts/models/connect4_net.pth \
  --solver /path/to/solver \
  --games 20
```

## Evaluation

`src/evaluate.py` reports:

* Top 1 accuracy
* Top 2 accuracy
* Macro F1
* Illegal move rate
* Confusion matrix
* Per class classification report

The illegal move rate checks whether predicted columns are playable in the corresponding board position.

## Browser Bridge

The bridge automates live play on papergames.io using the CNN+MCTS model.

Setup (first time):

```bash
.venv/bin/playwright install firefox
```

Launch:

```bash
bash launch_bridge.sh --our-username "Your Username"
```

Override model:

```bash
BRIDGE_MODEL=artifacts/models/selfplay_iter1.pth bash launch_bridge.sh --our-username "Your Username"
```