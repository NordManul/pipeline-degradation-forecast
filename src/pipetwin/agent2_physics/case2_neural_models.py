"""Small CPU models for the height sequence around a cold-case segment."""

import torch
from torch import nn
import numpy as np
import pandas as pd

from .broad_case2 import decode, encode, training_rows


POINTS = 11
STATIC = ["position_fraction", "altitude_relative_m"]


def local_windows(frame, radius):
    """Build windows by section/state, keyed to original rows."""
    if not frame.index.is_unique:
        raise ValueError("A profile frame requires a unique row index")
    result = np.zeros((len(frame), POINTS, 2), dtype=np.float32)
    offsets = np.linspace(-radius, radius, POINTS)
    for indices in frame.groupby("exposure_id", sort=False).indices.values():
        group = frame.iloc[indices].sort_values("center_m")
        x = group.center_m.to_numpy()
        z = group.altitude_relative_m.to_numpy()
        known = np.isfinite(z)
        if not known.any():
            continue
        if np.any(np.diff(x) <= 0):
            raise ValueError("Profile positions must be strictly increasing")
        samples = x[:, None] + offsets
        z_known = z[known]
        x_known = x[known]
        center = np.interp(x, x_known, z_known)
        heights = np.interp(samples, x_known, z_known) - center[:, None]
        available = (samples >= x_known[0]) & (samples <= x_known[-1])
        order = frame.index.get_indexer(group.index)
        result[order, :, 0] = np.where(available, heights, 0)
        result[order, :, 1] = available
    return result


def attach_windows(frame, radii):
    result = frame.copy()
    for radius in radii:
        window = local_windows(frame, radius)
        columns = window_columns(radius)
        result = result.join(pd.DataFrame(window.reshape(len(frame), -1), columns=columns, index=frame.index))
    return result


def window_columns(radius):
    return [f"profile_{radius}_{i}_{channel}" for i in range(POINTS) for channel in ["dz", "available"]]


class ProfileNet(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.architecture = architecture
        if architecture == "mlp":
            self.profile = nn.Sequential(nn.Flatten(), nn.Linear(POINTS * 2, 12), nn.GELU())
        elif architecture == "cnn":
            self.profile = nn.Sequential(nn.Conv1d(2, 8, kernel_size=3, padding=1), nn.GELU(),
                                         nn.Conv1d(8, 12, kernel_size=3, padding=1), nn.GELU())
        elif architecture == "transformer":
            self.project = nn.Linear(2, 12)
            self.position = nn.Parameter(torch.zeros(1, POINTS, 12))
            self.profile = nn.TransformerEncoderLayer(d_model=12, nhead=2, dim_feedforward=24,
                                                      dropout=0.15, activation="gelu", batch_first=True)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        width = 12 if architecture == "mlp" else 24
        self.head = nn.Sequential(nn.Linear(width + len(STATIC), 12), nn.GELU(),
                                  nn.Dropout(0.15), nn.Linear(12, 1))

    def forward(self, profile, static):
        if self.architecture == "cnn":
            tokens = self.profile(profile.transpose(1, 2)).transpose(1, 2)
        elif self.architecture == "transformer":
            tokens = self.profile(self.project(profile) + self.position)
        else:
            return self.head(torch.cat([self.profile(profile), static], dim=1)).squeeze(1)
        mask = profile[:, :, 1:]
        pooled = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.head(torch.cat([tokens[:, POINTS // 2], pooled, static], dim=1)).squeeze(1)


class NeuralProfile:
    def __init__(self, architecture, radius, seed):
        self.architecture, self.radius, self.seed = architecture, radius, seed

    def inputs(self, frame):
        profile = frame[window_columns(self.radius)].to_numpy(dtype=np.float32, copy=True)
        profile = profile.reshape(len(frame), POINTS, 2)
        profile[:, :, 0] = np.clip(profile[:, :, 0] / self.profile_scale, -4, 4)
        values = frame[STATIC].fillna(self.median).to_numpy()
        static = np.clip((values - self.mean) / self.scale, -4, 4)
        return torch.from_numpy(profile), torch.from_numpy(static.astype(np.float32))

    def fit(self, frame):
        train, weights = training_rows(frame, "balanced")
        values = train[STATIC]
        self.median = values.median().fillna(0)
        imputed = values.fillna(self.median).to_numpy()
        self.mean = imputed.mean(axis=0)
        self.scale = np.maximum(imputed.std(axis=0), 1e-3)
        raw_windows = train[window_columns(self.radius)].to_numpy().reshape(len(train), POINTS, 2)
        known = raw_windows[:, :, 1] > 0
        relief = raw_windows[:, :, 0][known]
        self.profile_scale = max(float(np.sqrt(np.mean(relief ** 2))), 0.1) if len(relief) else 1.0
        response = encode(train, "log_rate")
        self.response_mean = float(np.average(response, weights=weights))
        self.response_scale = max(float(np.sqrt(np.average(
            (response - self.response_mean) ** 2, weights=weights))), 1e-6)
        tensors = self.inputs(train)
        target = torch.tensor((response - self.response_mean) / self.response_scale, dtype=torch.float32)
        weight = torch.tensor(weights, dtype=torch.float32)
        # Isolate initialization, dropout and sample ordering from other fits.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.net = ProfileNet(self.architecture)
            optimizer = torch.optim.AdamW(self.net.parameters(), lr=0.003, weight_decay=0.02)
            for _ in range(50):
                for batch in torch.randperm(len(train)).split(64):
                    prediction = self.net(*(value[batch] for value in tensors))
                    loss = (weight[batch] * (prediction - target[batch]).square()).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        self.net.eval()
        return self

    def predict(self, frame, *, development=False):
        with torch.inference_mode():
            profile, static = self.inputs(frame)
            values = [self.net(x, z).numpy() for x, z in zip(profile.split(256), static.split(256))]
        response = np.concatenate(values).astype(float) * self.response_scale + self.response_mean
        return decode(response, frame, "log_rate")
