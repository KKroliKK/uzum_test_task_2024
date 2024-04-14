import pandas as pd
from sklearn.model_selection import train_test_split

from solution.constants import SEED


def train_valid_test_split(
    df: pd.DataFrame, val_size=0.1, test_size=0.1, stratify_col: str = None
) -> pd.DataFrame:

    df = df.copy()

    train_idxs, val_idxs = train_test_split(
        df.index, test_size=val_size, stratify=df[stratify_col], random_state=SEED
    )

    scaled_test_size = test_size / (1 - val_size)

    train_idxs, test_idxs = train_test_split(
        train_idxs,
        test_size=scaled_test_size,
        stratify=df[stratify_col].loc[train_idxs],
        random_state=SEED,
    )

    df["train"] = df.index.isin(train_idxs)
    df["valid"] = df.index.isin(val_idxs)
    df["test"] = df.index.isin(test_idxs)

    return df
