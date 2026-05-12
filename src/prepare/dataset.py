import numpy as np
import pandas as pd
from preprocess import (
    load_dataset,
    board_from_current_player_perspective,
    flatten_board,
    get_legal_moves,
)


def build_features(df: pd.DataFrame):
    """
    Converts dataframe into (X, y) for baseline:
    X: flattened current-player board (n_samples, 42)
    y: best_move class label in [0, 6]
    """
    boards = df["sequence"].astype(str).apply(
        board_from_current_player_perspective
    )
    X = np.stack(boards.apply(flatten_board).values)
    y = df["best_move"].values.astype(int) - 1
    return X, y


def build_features_with_mask(df: pd.DataFrame):
    """
 Converts dataframe into:
    X: flattened board features
    y: best move labels in [0, 6]
    legal_masks: shape (n_samples, 7), 1 = legal, 0 = illegal
    """
    boards = df["sequence"].astype(str).apply(
        board_from_current_player_perspective
    )
    X = np.stack(boards.apply(flatten_board).values)
    y = df["best_move"].values.astype(int) - 1
    legal_masks = np.stack(
        boards.apply(get_legal_moves).values
    ).astype(np.int8)
    return X, y, legal_masks


def load_train_val():
    train_df = load_dataset("UCI-Midgame-d30.train.csv")
    val_df = load_dataset("UCI-Midgame-d30.val.csv")

    X_train, y_train = build_features(train_df)
    X_val, y_val = build_features(val_df)

    return X_train, y_train, X_val, y_val


def load_train_val_with_mask():
    train_df = load_dataset("UCI-Midgame-d30.train.csv")
    val_df = load_dataset("UCI-Midgame-d30.val.csv")

    X_train, y_train, train_masks = build_features_with_mask(train_df)
    X_val, y_val, val_masks = build_features_with_mask(val_df)

    return X_train, y_train, train_masks, X_val, y_val, val_masks


if __name__ == "__main__":
    X_train, y_train, train_masks, X_val, y_val, val_masks = load_train_val_with_mask()

    print("Train shape:", X_train.shape)
    print("Val shape:", X_val.shape)
    print("Train legal mask shape:", train_masks.shape)
    print("Val legal mask shape:", val_masks.shape)
    print("Example label:", y_train[0])
    print("Example legal mask:", train_masks[0])