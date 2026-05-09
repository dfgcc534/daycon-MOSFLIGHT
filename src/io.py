"""IO utilities for DACON 236716 muflight data.

Per plan-001 §3.1, §4.1.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

TIMESTEPS_MS = np.arange(-400, 1, 40, dtype=np.int64)
N_TIMESTEPS = 11
N_AXES = 3
T_TARGET_MS = 80


def _data_root(data_root: Path | None) -> Path:
    return Path(data_root) if data_root is not None else DATA_ROOT


def load_sample(
    sample_id: str,
    split: Literal["train", "test"] = "train",
    data_root: Path | None = None,
) -> np.ndarray:
    path = _data_root(data_root) / split / f"{sample_id}.csv"
    arr = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(1, 2, 3), dtype=np.float64)
    if arr.shape != (N_TIMESTEPS, N_AXES):
        raise ValueError(f"{sample_id}: shape {arr.shape} != ({N_TIMESTEPS}, {N_AXES})")
    return arr


def load_all_samples(
    split: Literal["train", "test"] = "train",
    data_root: Path | None = None,
) -> tuple[list[str], np.ndarray]:
    root = _data_root(data_root) / split
    files = sorted(root.glob("*.csv"))
    ids = [f.stem for f in files]
    X = np.empty((len(files), N_TIMESTEPS, N_AXES), dtype=np.float64)
    for i, f in enumerate(files):
        arr = np.loadtxt(f, delimiter=",", skiprows=1, usecols=(1, 2, 3), dtype=np.float64)
        if arr.shape != (N_TIMESTEPS, N_AXES):
            raise ValueError(f"{f.stem}: shape {arr.shape} != ({N_TIMESTEPS}, {N_AXES})")
        X[i] = arr
    return ids, X


def load_labels(data_root: Path | None = None) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(_data_root(data_root) / "train_labels.csv")
    df = df.sort_values("id").reset_index(drop=True)
    return df["id"].tolist(), df[["x", "y", "z"]].to_numpy(dtype=np.float64)


def kfold_split(
    ids: list[str], k: int = 5, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic stride-k split over alphabetically-sorted ids.

    fold f gets samples whose alphabetical rank i satisfies i % k == f.
    `seed` accepted for API stability; the split is fully deterministic.
    """
    n = len(ids)
    rank = np.argsort(np.argsort(ids))
    fold_of = rank % k
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for f in range(k):
        val_idx = np.where(fold_of == f)[0]
        train_idx = np.where(fold_of != f)[0]
        folds.append((train_idx, val_idx))
    _ = seed
    return folds
