import numpy as np

def get_rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)