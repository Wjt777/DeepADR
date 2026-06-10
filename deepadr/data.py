from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset, random_split


@dataclass(frozen=True)
class DeepADRData:
    drug_molformer: np.ndarray
    drug_target: np.ndarray
    adr_semantic: np.ndarray
    adr_matrix: pd.DataFrame

    @property
    def adr_features(self) -> np.ndarray:
        return self.adr_semantic.T


def load_data(data_dir: Union[str, Path]) -> DeepADRData:
    data_dir = Path(data_dir)
    return DeepADRData(
        drug_molformer=np.load(data_dir / "drug_molformer.npy"),
        drug_target=np.load(data_dir / "drug_target.npy"),
        adr_semantic=np.load(data_dir / "adr_semantic.npy"),
        adr_matrix=pd.read_csv(data_dir / "drug_adr_mtx.csv", index_col=0),
    )


class DrugADRDataset(Dataset):
    def __init__(
        self,
        data: DeepADRData,
        drug_indices: np.ndarray,
        adr_indices: np.ndarray,
        labels: np.ndarray,
    ):
        self.drug_molformer = torch.tensor(
            data.drug_molformer[drug_indices], dtype=torch.float32
        )
        self.drug_target = torch.tensor(data.drug_target[drug_indices], dtype=torch.float32)
        self.adr_semantic = torch.tensor(data.adr_features[adr_indices], dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).view(-1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return (
            self.drug_molformer[index],
            self.drug_target[index],
            self.adr_semantic[index],
            self.labels[index],
        )


def make_regression_dataset(data: DeepADRData) -> DrugADRDataset:
    values = data.adr_matrix.values
    drug_indices, adr_indices = np.nonzero(values)
    labels = values[drug_indices, adr_indices]
    return DrugADRDataset(data, drug_indices, adr_indices, labels)


def make_classification_dataset(
    data: DeepADRData,
    negative_ratio: float = 1.0,
    seed: int = 42,
) -> DrugADRDataset:
    """Build a binary occurrence dataset.

    Labels follow the inclusive occurrence definition used in this release:
    any non-zero frequency level, including 1 and 2, is positive; 0 is negative.
    Negative samples are randomly drawn from zero entries.
    """
    values = data.adr_matrix.values
    flat = values.reshape(-1)
    positive_flat_indices = np.flatnonzero(flat > 0)
    negative_flat_indices = np.flatnonzero(flat == 0)

    rng = np.random.default_rng(seed)
    n_negative = min(len(negative_flat_indices), int(len(positive_flat_indices) * negative_ratio))
    sampled_negative_indices = rng.choice(
        negative_flat_indices, size=n_negative, replace=False
    )

    selected = np.concatenate([positive_flat_indices, sampled_negative_indices])
    rng.shuffle(selected)

    n_adrs = values.shape[1]
    drug_indices = selected // n_adrs
    adr_indices = selected % n_adrs
    labels = (flat[selected] > 0).astype(np.float32)
    return DrugADRDataset(data, drug_indices, adr_indices, labels)


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[Subset, Subset, Subset]:
    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val, n_test], generator=generator)


def describe_matrix(data: DeepADRData) -> Dict[str, int]:
    values = data.adr_matrix.values
    unique, counts = np.unique(values, return_counts=True)
    stats = {f"level_{int(level)}": int(count) for level, count in zip(unique, counts)}
    stats["positive_nonzero"] = int(np.count_nonzero(values))
    stats["total_pairs"] = int(values.size)
    return stats
