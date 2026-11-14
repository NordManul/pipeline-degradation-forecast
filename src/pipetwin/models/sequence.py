"""Encoders for the pre-inspection pipe profile and TECH history."""

import numpy as np
import pandas as pd
import torch
from torch import nn


SPATIAL_RADIUS_CELLS = 4
TECH_BINS = 24
TECH_BIN_DAYS = 30
TECH_CHANNELS = ["water_fraction", "temperature_c", "velocity_m_s", "fill_coefficient"]
COMMON_PROFILE = ["altitude_relative_m", "slope_50m", "lowpoint_depth_m"]
BETWEEN_PROFILE = ["anchor_mm", "anchor_observed", "edge_min_mm", "local_contrast_mm", "individual_min_mm"]


def spatial_windows(grid, columns):
    """Offsets are real segment IDs, within one section and one forecast window."""
    width = 2 * SPATIAL_RADIUS_CELLS + 1
    windows = np.full((len(grid), width, len(columns)), np.nan)
    for _, section in grid.reset_index(drop=True).groupby(["exposure_id", "id", "target_date"], sort=False):
        cells = section.segment_id.to_numpy()
        if len(np.unique(cells)) != len(cells):
            raise ValueError("Duplicate cells in a profile window")
        values = section[columns].to_numpy(dtype=float).copy()
        if "anchor_mm" in columns:
            values[~section.anchor_observed.to_numpy(), columns.index("anchor_mm")] = np.nan
        search = pd.Index(cells)
        for step in range(-SPATIAL_RADIUS_CELLS, SPATIAL_RADIUS_CELLS + 1):
            indices = search.get_indexer(cells + step)
            keep = indices >= 0
            windows[section.index[keep], step + SPATIAL_RADIUS_CELLS] = values[indices[keep]]
    return windows


def tech_windows(history, grid):
    """Twenty-four 30-day windows ending immediately before the example date."""
    histories = {}
    grouped = history.groupby("exposure_id", sort=False)
    for sample in grid[["exposure_id", "window_start", "window_end"]].drop_duplicates().itertuples(index=False):
        period = grouped.get_group(sample.exposure_id)
        known = period.observed
        if (not period.day.lt(sample.window_end).all() or not period.day.ge(sample.window_start).all()
                or not period.loc[known, "available_at"].lt(sample.window_end).all()):
            raise ValueError("TECH token includes history outside the causal window")
        lag = (sample.window_end - period.day).dt.days - 1
        keep = lag < TECH_BINS * TECH_BIN_DAYS
        block = TECH_BINS - 1 - lag.loc[keep] // TECH_BIN_DAYS
        values = period.loc[keep, TECH_CHANNELS].where(known.loc[keep], np.nan)
        time = values.groupby(block).mean().reindex(range(TECH_BINS))
        observed_count = known.loc[keep].groupby(block).sum()
        calendar_count = block.value_counts()
        time["coverage"] = observed_count.reindex(time.index, fill_value=0) / TECH_BIN_DAYS
        time["inside_window"] = calendar_count.reindex(time.index, fill_value=0) / TECH_BIN_DAYS
        histories[sample.exposure_id] = time.to_numpy(dtype=float)
    return np.stack([histories[key] for key in grid.exposure_id])


class NumericInputs:
    """Fit channel statistics on train. Emit finite values and a missing mask."""

    def fit(self, values):
        flat = values.reshape(-1, values.shape[-1])
        valid = np.isfinite(flat)
        count = np.maximum(1, valid.sum(axis=0))
        self.mean_ = np.where(valid, flat, 0.).sum(axis=0) / count
        variance = np.where(valid, (flat - self.mean_) ** 2, 0.).sum(axis=0) / count
        self.scale_ = np.maximum(np.sqrt(variance), 1e-6)
        return self

    def transform(self, values):
        missing = ~np.isfinite(values)
        scaled = np.clip((values - self.mean_) / self.scale_, -4, 4)
        return torch.from_numpy(np.concatenate([np.where(missing, 0., scaled), missing], axis=-1).astype(np.float32))


class SequenceEncoder(nn.Module):
    def __init__(self, architecture, channels, length, hidden=16):
        super().__init__()
        self.architecture = architecture
        if architecture == "lstm":
            self.encoder = nn.LSTM(channels, hidden, batch_first=True, bidirectional=True)
            self.output_width = 2 * hidden
        elif architecture == "cnn":
            self.encoder = nn.Sequential(nn.Conv1d(channels, hidden, 3, padding=1), nn.GELU(),
                                         nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU())
            self.output_width = hidden
        elif architecture == "transformer":
            self.project = nn.Linear(channels, hidden)
            self.position = nn.Parameter(torch.empty(1, length, hidden))
            nn.init.normal_(self.position, std=.02)
            layer = nn.TransformerEncoderLayer(hidden, nhead=2, dim_feedforward=32,
                                               dropout=.10, batch_first=True, activation="gelu")
            self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
            self.output_width = hidden
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def forward(self, x, center_token):
        if self.architecture == "cnn":
            values = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        elif self.architecture == "lstm":
            values, _ = self.encoder(x)
        else:
            values = self.encoder(self.project(x) + self.position)
        if center_token:
            return values[:, values.shape[1] // 2]
        return values.mean(dim=1)


class ProfileRegressor(nn.Module):
    def __init__(self, architecture, dimensions):
        super().__init__()
        static, space, sequence = dimensions
        self.local = nn.Sequential(nn.Linear(static[-1], 24), nn.LayerNorm(24), nn.GELU())
        self.space = SequenceEncoder(architecture, space[-1], space[-2])
        self.time = SequenceEncoder(architecture, sequence[-1], sequence[-2])
        width = 24 + self.space.output_width + self.time.output_width
        self.regression = nn.Sequential(nn.Linear(width, 24), nn.GELU(), nn.Dropout(.10), nn.Linear(24, 1))

    def forward(self, static, space, sequence):
        features = torch.cat([self.local(static), self.space(space, True), self.time(sequence, False)], dim=1)
        return self.regression(features).squeeze(-1)


def fit_predict(architecture, fit, query, target, weights, *, seed, epochs):
    """Equal training schedule for the encoders, with no held-out early stopping."""
    torch.manual_seed(seed)
    train, test = [], []
    for x, q in zip(fit, query):
        transform = NumericInputs().fit(x)
        train.append(transform.transform(x))
        test.append(transform.transform(q))
    model = ProfileRegressor(architecture, [x.shape for x in train])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
    mean = float(np.average(target, weights=weights))
    scale = max(1e-8, np.sqrt(np.average((target - mean) ** 2, weights=weights)))
    y = torch.tensor((target - mean) / scale, dtype=torch.float32)
    w = torch.tensor(weights, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for indices in torch.randperm(len(y), generator=generator).split(64):
            optimizer.zero_grad()
            prediction = model(*(x[indices] for x in train))
            loss = torch.mean(w[indices] * (prediction - y[indices]) ** 2)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        batches = zip(*(x.split(512) for x in test))
        result = torch.cat([model(*x) for x in batches]).numpy().astype(float)
    return result * scale + mean, sum(p.numel() for p in model.parameters())
