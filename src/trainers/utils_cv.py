# src/trainers/utils_cv.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterator, Tuple, Sequence

class PurgedKFold:
    """
    Time-aware CV on a shared date axis.
    Splits are made on UNIQUE DATES, then mapped back to sample indices.
    Embargo removes samples around validation dates from the train set.

    Parameters
    ----------
    n_splits : int
        Number of folds on the unique date index.
    embargo : int
        Embargo size in trading days applied on both sides of the validation window.
    """
    def __init__(self, n_splits: int = 5, embargo: int = 5):
        assert n_splits >= 2
        self.n_splits = n_splits
        self.embargo = embargo

    def split(self, dates: Sequence[pd.Timestamp]) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        # dates: per-sample datetime-like array
        dates = pd.to_datetime(dates)
        unique_dates = pd.Index(pd.Series(dates).dt.normalize().unique()).sort_values()
        n = len(unique_dates)
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1

        current = 0
        date_slices = []
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            date_slices.append((start, stop))
            current = stop

        # Precompute sample index per unique date
        date_to_positions = {}
        for i, d in enumerate(dates):
            dn = pd.Timestamp(d).normalize()
            date_to_positions.setdefault(dn, []).append(i)

        for (start, stop) in date_slices:
            val_dates = unique_dates[start:stop]

            # Validation indices
            val_idx = np.array([idx for d in val_dates for idx in date_to_positions.get(d, [])], dtype=int)

            # Embargoed train dates
            embargo_start = max(0, start - self.embargo)
            embargo_stop  = min(len(unique_dates), stop + self.embargo)
            train_dates = unique_dates[:embargo_start].append(unique_dates[embargo_stop:])

            train_idx = np.array([idx for d in train_dates for idx in date_to_positions.get(d, [])], dtype=int)

            yield train_idx, val_idx
