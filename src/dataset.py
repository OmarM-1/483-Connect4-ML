import numpy as np
import pandas as pd
from preprocess import (
    load_dataset,
    board_from_current_player_perspective,
    flatten_board,
)

def build_features(df: pd.DataFrame):
    """
    Converts dataframe into (X, y)
    """
    boards = df["sequence"].astype(str).apply(
        board_from_current_player_perspective
    )

    X = np.stack(boards.apply(flatten_board).values)

    y = df["best_move"].values.astype(int) - 1

    return X, y


def load_train_val():
    train_df = load_dataset("UCI-Midgame-d30.train.csv")
    val_df = load_dataset("UCI-Midgame-d30.val.csv")

    X_train, y_train = build_features(train_df)
    X_val, y_val = build_features(val_df)

    return X_train, y_train, X_val, y_val


if __name__ == "__main__":
    X_train, y_train, X_val, y_val = load_train_val()

    print("Train shape:", X_train.shape)
    print("Val shape:", X_val.shape)

    print("Example feature vector (first row):")
    print(X_train[0])

    print("Example label:", y_train[0])