#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Channel Estimator
=================
Reusable bandwidth prediction API for the Jetson pipeline.

Supported model types:
- linear_regression
- mlp
- 1d_cnn
- mean_factor

Default behavior matches the pipeline requirement:
- default predictor model: mean_factor
- collect bandwidth observations during warmup
- train once after warmup
- optionally switch to online mode and retrain after each task
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn


SUPPORTED_CHANNEL_MODEL_TYPES = ['linear_regression', 'mlp', '1d_cnn', 'mean_factor']
DEFAULT_CHANNEL_MODEL_TYPE = 'mean_factor'
DEFAULT_CHANNEL_UPDATE_MODE = 'warmup'
DEFAULT_BANDWIDTH_BYTES_PER_SEC = 10e6
DEFAULT_MEAN_FACTOR = 1.0
DEFAULT_TORCH_EPOCHS = 100
DEFAULT_TORCH_LR = 0.001
DEFAULT_TORCH_BATCH_SIZE = 64


def _normalize_model_type(model_type):
    normalized = model_type.strip().lower().replace('-', '_')
    aliases = {
        'linear': 'linear_regression',
        'linreg': 'linear_regression',
        'lr': 'linear_regression',
        'mlp': 'mlp',
        'cnn': '1d_cnn',
        'conv1d': '1d_cnn',
        '1dcnn': '1d_cnn',
        '1d_cnn': '1d_cnn',
        'mean': 'mean_factor',
        'avg': 'mean_factor',
        'average': 'mean_factor',
        'mean_factor': 'mean_factor',
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_CHANNEL_MODEL_TYPES:
        raise ValueError(
            'Unsupported channel model type: {}. Choose from {}'.format(
                model_type, SUPPORTED_CHANNEL_MODEL_TYPES,
            )
        )
    return normalized


class BandwidthMLP(nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(window_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


class BandwidthCNN(nn.Module):
    def __init__(self, window_size):
        super().__init__()
        pooled_width = max(1, window_size // 2)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(16 * pooled_width, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x.unsqueeze(1))


def _create_sliding_window_dataset(series, window_size):
    data_x = []
    data_y = []
    for index in range(len(series) - window_size):
        data_x.append(series[index:index + window_size])
        data_y.append(series[index + window_size])
    return np.asarray(data_x, dtype=float), np.asarray(data_y, dtype=float)


def _train_torch_model(model, x_train, y_train, device,
                       epochs=DEFAULT_TORCH_EPOCHS,
                       lr=DEFAULT_TORCH_LR,
                       batch_size=DEFAULT_TORCH_BATCH_SIZE):
    model = model.to(device)
    model.train()

    x_tensor = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    y_tensor = torch.as_tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)

    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    return model


def _predict_torch_model(model, x_input, device):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.as_tensor(x_input, dtype=torch.float32, device=device)
        return model(x_tensor).detach().cpu().numpy().reshape(-1)


class _SingleLinkChannelModel:
    def __init__(self, model_type, window_size, default_bandwidth, device,
                 mean_factor=DEFAULT_MEAN_FACTOR,
                 epochs=DEFAULT_TORCH_EPOCHS,
                 lr=DEFAULT_TORCH_LR,
                 batch_size=DEFAULT_TORCH_BATCH_SIZE):
        self.model_type = _normalize_model_type(model_type)
        self.window_size = max(2, int(window_size))
        self.default_bandwidth = float(default_bandwidth)
        self.device = device
        self.mean_factor = float(mean_factor)
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

        self.history = []
        self.scaler = StandardScaler()
        self.model = None
        self.trained = False
        self.target_mean = 0.0
        self.target_std = 1.0
        self.fixed_prediction = None

    def add_observation(self, throughput):
        self.history.append(float(throughput))

    def fit(self):
        if len(self.history) < self.window_size + 1:
            if self.model_type == 'mean_factor' and self.history:
                self.trained = True
                self.model = None
                self.fixed_prediction = max(float(np.mean(self.history)) * self.mean_factor, 1e-6)
                return True
            self.trained = False
            self.model = None
            self.fixed_prediction = None
            return False

        if self.model_type == 'mean_factor':
            self.trained = True
            self.model = None
            self.fixed_prediction = max(float(np.mean(self.history)) * self.mean_factor, 1e-6)
            return True

        x_train, y_train = _create_sliding_window_dataset(self.history, self.window_size)
        x_train_scaled = self.scaler.fit_transform(x_train)

        if self.model_type == 'linear_regression':
            self.target_mean = 0.0
            self.target_std = 1.0
            self.model = LinearRegression()
            self.model.fit(x_train_scaled, y_train)
        else:
            self.target_mean = float(np.mean(y_train))
            self.target_std = float(np.std(y_train))
            if self.target_std < 1e-8:
                self.target_std = 1.0
            y_train_scaled = (y_train - self.target_mean) / self.target_std

        if self.model_type == 'mlp':
            self.model = _train_torch_model(
                BandwidthMLP(self.window_size),
                x_train_scaled,
                y_train_scaled,
                self.device,
                epochs=self.epochs,
                lr=self.lr,
                batch_size=self.batch_size,
            )
        elif self.model_type == '1d_cnn':
            self.model = _train_torch_model(
                BandwidthCNN(self.window_size),
                x_train_scaled,
                y_train_scaled,
                self.device,
                epochs=self.epochs,
                lr=self.lr,
                batch_size=self.batch_size,
            )

        self.trained = True
        return True

    def predict(self):
        if not self.history:
            return max(self.default_bandwidth * self.mean_factor, 1e-6)

        if self.model_type == 'mean_factor':
            if self.fixed_prediction is not None:
                return self.fixed_prediction
            return max(float(np.mean(self.history)) * self.mean_factor, 1e-6)

        if len(self.history) < self.window_size:
            return max(self.history[-1], 1e-6)

        if not self.trained or self.model is None:
            window = self.history[-self.window_size:]
            return max(float(np.mean(window)), 1e-6)

        x_input = np.asarray(self.history[-self.window_size:], dtype=float).reshape(1, -1)
        x_input_scaled = self.scaler.transform(x_input)

        if self.model_type == 'linear_regression':
            prediction = float(self.model.predict(x_input_scaled)[0])
        else:
            prediction_scaled = float(_predict_torch_model(self.model, x_input_scaled, self.device)[0])
            prediction = prediction_scaled * self.target_std + self.target_mean

        return max(prediction, 1e-6)


class ChannelEstimator:
    """Multi-link bandwidth estimator with a simple pipeline-friendly API."""

    def __init__(self,
                 n_links,
                 model_type=DEFAULT_CHANNEL_MODEL_TYPE,
                 window_size=5,
                 update_mode=DEFAULT_CHANNEL_UPDATE_MODE,
                 default_bandwidth=DEFAULT_BANDWIDTH_BYTES_PER_SEC,
                 mean_factor=DEFAULT_MEAN_FACTOR,
                 device=None,
                 epochs=DEFAULT_TORCH_EPOCHS,
                 lr=DEFAULT_TORCH_LR,
                 batch_size=DEFAULT_TORCH_BATCH_SIZE):
        if update_mode not in ('warmup', 'online'):
            raise ValueError("update_mode must be 'warmup' or 'online'")

        torch_device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_links = int(n_links)
        self.model_type = _normalize_model_type(model_type)
        self.window_size = max(2, int(window_size))
        self.update_mode = update_mode
        self.default_bandwidth = float(default_bandwidth)
        self.mean_factor = float(mean_factor)
        self.device = torch_device
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.links = [
            _SingleLinkChannelModel(
                model_type=self.model_type,
                window_size=self.window_size,
                default_bandwidth=self.default_bandwidth,
                device=self.device,
                mean_factor=self.mean_factor,
                epochs=self.epochs,
                lr=self.lr,
                batch_size=self.batch_size,
            )
            for _ in range(self.n_links)
        ]

    def observe(self, link_idx, bytes_sent, elapsed_sec, refit=None):
        if elapsed_sec <= 0 or bytes_sent <= 0:
            return None

        throughput = float(bytes_sent) / float(elapsed_sec)
        self.links[link_idx].add_observation(throughput)

        should_refit = self.update_mode == 'online' if refit is None else refit
        if should_refit:
            self.links[link_idx].fit()
        return throughput

    def observe_task(self, tp_stats, refit=None):
        observations = []
        touched_links = []
        for link_idx, (bytes_sent, elapsed_sec) in enumerate(tp_stats):
            throughput = self.observe(link_idx, bytes_sent, elapsed_sec, refit=False)
            observations.append(throughput)
            if throughput is not None:
                touched_links.append(link_idx)

        should_refit = self.update_mode == 'online' if refit is None else refit
        if should_refit:
            for link_idx in touched_links:
                self.links[link_idx].fit()
        return observations

    def fit(self):
        return [link.fit() for link in self.links]

    def predict(self, link_idx):
        return self.links[link_idx].predict()

    def predict_all(self):
        return [self.predict(link_idx) for link_idx in range(self.n_links)]

    def summary(self):
        trained_links = sum(1 for link in self.links if link.trained)
        history_sizes = [len(link.history) for link in self.links]
        return (
            'ChannelEstimator(model_type={}, update_mode={}, window_size={}, '
            'mean_factor={}, trained_links={}/{}, history={})'
        ).format(
            self.model_type,
            self.update_mode,
            self.window_size,
            self.mean_factor,
            trained_links,
            self.n_links,
            history_sizes,
        )


__all__ = [
    'SUPPORTED_CHANNEL_MODEL_TYPES',
    'DEFAULT_CHANNEL_MODEL_TYPE',
    'DEFAULT_CHANNEL_UPDATE_MODE',
    'DEFAULT_MEAN_FACTOR',
    'ChannelEstimator',
    'BandwidthMLP',
    'BandwidthCNN',
]
