from pathlib import Path
import numpy as np
import pandas as pd

ROWS = 6
COLS = 7
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def sequence_to_board(sequence: str) -> np.ndarray:
    """
    Convert a move sequence like '112347' into a 6x7 Connect 4 board.

    Board convention:
    - shape = (6, 7)
    - row 0 is the bottom row
    - 1 = player 1
    - -1 = player 2
    - 0 = empty
    """
    seq = str(sequence).strip()
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    heights = np.zeros(COLS, dtype=np.int8)

    player = 1
    for move_index, ch in enumerate(seq):
        if ch < "1" or ch > "7":
            raise ValueError(f"Invalid column '{ch}' at position {move_index}.")

        col = int(ch) - 1
        row = heights[col]

        if row >= ROWS:
            raise ValueError(f"Column {col + 1} is full at move index {move_index}.")

        board[row, col] = player
        heights[col] += 1
        player *= -1

    return board


def current_player_from_sequence(sequence: str) -> int:
    seq_len = len(str(sequence).strip())
    return 1 if seq_len % 2 == 0 else -1


def board_from_current_player_perspective(sequence: str) -> np.ndarray:
    board = sequence_to_board(sequence)
    to_move = current_player_from_sequence(sequence)
    return board * to_move


def board_to_pretty_string(board: np.ndarray) -> str:
    symbol = {1: "X", -1: "O", 0: "."}
    lines = []
    for r in range(ROWS - 1, -1, -1):
        lines.append(" ".join(symbol[int(board[r, c])] for c in range(COLS)))
    lines.append("1 2 3 4 5 6 7")
    return "\n".join(lines)


def load_dataset(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    return pd.read_csv(path, low_memory=False)


def get_legal_moves(board: np.ndarray) -> np.ndarray:
    legal = np.zeros(COLS, dtype=np.int8)
    for c in range(COLS):
        legal[c] = 1 if board[ROWS - 1, c] == 0 else 0
    return legal


def legal_move_indices(board: np.ndarray) -> np.ndarray:
    return np.where(get_legal_moves(board) == 1)[0]


def flatten_board(board: np.ndarray) -> np.ndarray:
    return board.astype(np.int8).reshape(-1)


def add_board_columns(df: pd.DataFrame, perspective: str = "raw") -> pd.DataFrame:
    df = df.copy()

    if perspective == "raw":
        df["board"] = df["sequence"].astype(str).apply(sequence_to_board)
    elif perspective == "current":
        df["board"] = df["sequence"].astype(str).apply(board_from_current_player_perspective)
    else:
        raise ValueError("perspective must be 'raw' or 'current'")

    return df


if __name__ == "__main__":
    df = load_dataset("UCI-Midgame-d30.val.csv")

    print("Loaded rows:", len(df))
    print("Columns:", list(df.columns))

    sample = df.iloc[0]
    seq = str(sample["sequence"])

    print("\nSample sequence:", seq)
    print("Move count column:", sample["move_count"])
    print("Computed move count:", len(seq))

    raw_board = sequence_to_board(seq)
    current_board = board_from_current_player_perspective(seq)

    print("\nRaw board:")
    print(board_to_pretty_string(raw_board))

    print("\nCurrent-player perspective board:")
    print(board_to_pretty_string(current_board))

    print("\nLegal moves:", get_legal_moves(raw_board))
    print("Best move label:", sample["best_move"])