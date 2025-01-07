import pandas as pd
import numpy as np
import pyarrow as pa
from time import perf_counter
from pathlib import Path
import os
import pickle


def test():
    rng = np.random.default_rng(seed=42)
    base_array = rng.random((100, 100, 1500))
    b_path = Path()
    f_paths = []

    print(base_array.shape)

    # numpy
    f_paths.append(b_path / "base_array.npy")
    tic = perf_counter()
    np.save(f_paths[-1], base_array, allow_pickle=True)
    elapsed = perf_counter() - tic
    size = os.stat(f_paths[-1]).st_size / (1024 * 1024)
    print(f"Numpy save time: {elapsed:.4f}s, size: {size:.2f}MB")
    tic = perf_counter()
    np.load(f_paths[-1])
    print(f"Numpy load time: {perf_counter()-tic:.4f}s")

    # pickle
    f_paths.append(b_path / "base_array.pkl")
    tic = perf_counter()
    with open(f_paths[-1], "wb") as f:
        pickle.dump(base_array, f)
    elapsed = perf_counter() - tic
    size = os.stat(f_paths[-1]).st_size / (1024 * 1024)
    print(f"Pickle save time: {elapsed:.4f}s, size: {size:.2f}MB")
    tic = perf_counter()
    with open(f_paths[-1], "rb") as f:
        pickle.load(f)
    print(f"Pickle load time: {perf_counter()-tic:.4f}s")

    # clean
    for file in f_paths:
        if file.is_file():
            file.unlink()


if __name__ == "__main__":
    test()
