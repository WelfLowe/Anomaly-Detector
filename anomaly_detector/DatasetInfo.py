import numpy as np
from dataclasses import dataclass

@dataclass
class DatasetInfo:
    data: np.ndarray
    labels: np.ndarray
    val_data: np.ndarray
    val_labels: np.ndarray
    k: int
    n: int
    data_min: float
    data_max: float

