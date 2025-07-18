#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "polars",
#     "zarr",
# ]
# ///
import time
from pathlib import Path

import numpy as np
import polars as pl
import zarr


def _load_leaves(z: zarr.Group | zarr.Array, mask: np.ndarray | None) -> None:
    if isinstance(z, zarr.Array):
        if mask is not None:
            # z[mask][...]  # does not work
            z.get_orthogonal_selection(mask)
        else:
            z[...]
        return
    for key in z.keys():
        return _load_leaves(z[key], mask)


def _uniform(n_nodes: int, frac: float) -> np.ndarray:
    size = n_nodes * frac
    step = int(max(n_nodes / size, 1))
    mask = np.zeros(n_nodes, dtype=bool)
    mask[::step] = True
    return mask


def _ordered(n_nodes: int, frac: float) -> np.ndarray:
    size = int(n_nodes * frac)
    mask = np.zeros(n_nodes, dtype=bool)
    mask[:size] = True
    return mask


def main() -> None:
    DATA_PATH = Path("/Users/jordao.bragantini/Data/geff/Fluo-N3DL-DRO.zarr/01/tracks")

    z = zarr.open(DATA_PATH, mode="r")

    # size = z["nodes"]["ids"].shape[0]
    # only evaluating nodes
    # z = z["nodes"]
    size = z["edges"]["ids"].shape[0]
    z = z["edges"]

    data = []

    for frac in [0.5, 0.25, 0.1, 0.05]:
        for name, func in [
            ("uniform", _uniform),
            ("ordered", _ordered),
        ]:
            mask = func(size, frac)
            start_time = time.time()
            _load_leaves(z, mask)
            end_time = time.time()
            data.append(
                {
                    "name": name,
                    "frac": frac,
                    "time": end_time - start_time,
                }
            )

    start_time = time.time()
    _load_leaves(z, None)
    end_time = time.time()
    full_time = end_time - start_time

    data.append(
        {
            "name": "full",
            "frac": 1.0,
            "time": full_time,
        }
    )
    df = (
        pl.DataFrame(data)
        .with_columns(
            time_frac=pl.col("time") / full_time,
        )
        .sort(["name", "frac"])
    )

    print(df)


if __name__ == "__main__":
    main()
