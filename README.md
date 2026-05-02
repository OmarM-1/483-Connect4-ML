# Connect4 ML

Machine learning project for predicting strong Connect 4 moves from board positions

The project uses a UCI Connect 4 midgame dataset. Each position is represented by the move sequence that produced the board, and the target label is the solver recommended `best_move`.

## Requirements

* Python 3.10 or newer
* `pip`
* Optional: GNU Make, if you want to use the `make` commands

Python dependencies are listed in `requirements.txt`.

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

## Evaluation

`src/evaluate.py` reports:

* Top 1 accuracy
* Top 2 accuracy
* Macro F1
* Illegal move rate
* Confusion matrix
* Per class classification report

The illegal move rate checks whether predicted columns are playable in the corresponding board position.

## Future

In the process of having the model be playable against on https://papergames.io/en/connect4 via the play with a friend feature