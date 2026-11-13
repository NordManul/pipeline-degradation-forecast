"""Provisional local 10-metre grid, with explicit interval boundaries."""

import numpy as np
import pandas as pd


def build_grid(passport, segment_length_m=10.0):
    if not np.isfinite(segment_length_m) or segment_length_m <= 0:
        raise ValueError("Segment length must be positive")
    frames = []
    for row in passport.to_dict("records"):
        count = int(np.ceil(row["length_m"] / segment_length_m))
        frame = pd.DataFrame({"segment_id": np.arange(count, dtype=int)})
        for key, value in row.items():
            frame[key] = value
        frame["start_m"] = frame["segment_id"] * segment_length_m
        frame["end_m"] = np.minimum(frame["start_m"] + segment_length_m, row["length_m"])
        frame["center_m"] = (frame["start_m"] + frame["end_m"]) / 2
        frame["segment_length_m"] = frame["end_m"] - frame["start_m"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(["id", "segment_id"]).reset_index(drop=True)


def assign_segments(rows, passport, segment_length_m):
    result = rows.merge(passport[["id", "length_m"]], on="id", how="left", validate="many_to_one")
    result["in_range"] = (
        result["distance_m"].ge(0) & result["distance_m"].lt(result["length_m"])
    )
    result["segment_id"] = np.floor(result["distance_m"] / segment_length_m).astype("Int64")
    return result
