from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader, Dataset

RANDOM_SEED = 4213

if TYPE_CHECKING:
    import pandas as pd


class TabularDataset(Dataset):
    def __init__(
        self,
        *,
        datasets: list[tuple[torch.Tensor, torch.Tensor]],
        max_steps: int,
        is_classification: bool,
        cross_val_splits: int = 10,
    ):
        self.datasets = datasets
        self.max_steps = max_steps
        self.cross_val_splits = cross_val_splits
        self.is_classification = is_classification
        self._rng = np.random.RandomState(RANDOM_SEED)

        self._split_generators = [
            self.splits_generator(
                X_train=X,
                y_train=y,
                cross_val_splits=cross_val_splits,
                stratify_split=is_classification,
                seed=RANDOM_SEED + i,  # ensure different seeds per dataset
            )
            for i, (X, y) in enumerate(datasets)
        ]

    @staticmethod
    def splits_generator(
        *,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        cross_val_splits: int,
        stratify_split: bool,
        seed: int,
    ):
        rng = np.random.RandomState(seed)
        splitter = StratifiedKFold if stratify_split else KFold

        while True:
            splits = splitter(
                n_splits=cross_val_splits,
                random_state=rng.randint(0, 1 << 30),
                shuffle=True,
            ).split(
                X=X_train,
                y=y_train.cpu().numpy() if stratify_split else None,
            )
            yield from splits

    def __len__(self):
        return self.max_steps

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        dataset_idx = self._rng.randint(0, len(self.datasets))
        X_train, y_train = self.datasets[dataset_idx]
        train_idx, test_idx = next(self._split_generators[dataset_idx])

        expected_test_size = len(X_train) // self.cross_val_splits
        if len(test_idx) != expected_test_size:
            train_idx = np.concatenate([train_idx, test_idx[: len(test_idx) - expected_test_size]])
            test_idx = test_idx[len(test_idx) - expected_test_size :]

        return dict(
            X_train=X_train[train_idx],
            X_test=X_train[test_idx],
            y_train=y_train[train_idx],
            y_test=y_train[test_idx],
        )



def get_data_loader(
    *,
    Xs: list[pd.DataFrame],
    ys: list[pd.Series],
    max_steps: int,
    torch_rng: torch.Generator,
    batch_size: int,
    is_classification: bool,
    num_workers: int,
) -> DataLoader:
    datasets = [
        (
            torch.tensor(X.copy().values).float(),
            torch.tensor(y.copy().values).reshape(-1, 1).float(),
        )
        for X, y in zip(Xs, ys)
    ]

    dataset = TabularDataset(
        datasets=datasets,
        max_steps=max_steps * batch_size,
        is_classification=is_classification,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
        generator=torch_rng,
        persistent_workers=False,
    )
