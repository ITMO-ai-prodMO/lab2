from __future__ import annotations

import random
from pathlib import Path

import numpy as np


DEFAULT_SEED = 42


def seed_everything(seed: int = DEFAULT_SEED) -> int:
    random.seed(seed)
    np.random.seed(seed)
    return seed


def project_root_from_file(file_path: str) -> Path:
    return Path(file_path).resolve().parents[1]
