#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone 4-node Jetson ResNet56 pipeline with single-task style policies.

This script intentionally does not import jetson_resnet_pipeline.py.
"""

import argparse
import copy
import datetime
import json
import logging
import math
import os
import pickle
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import torch
import torch.nn as nn

from channel_estimator import (
    ChannelEstimator,
    SUPPORTED_CHANNEL_MODEL_TYPES,
)
from compressors import get_compressor
from http_transport import MessageType, create_transport
from llmint8_eta_mapping import resolve_codec_execution_plan, validate_llmint8_mapping_entries
from single_task_offline_plots import render_trial_plots
import resnet20


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

NODE_ORDER = ["A", "B", "C", "D"]
NODE_IPS = {
    "A": os.environ.get("JETSON_NODE_A_IP", "192.168.1.20"),
    "B": os.environ.get("JETSON_NODE_B_IP", "192.168.1.21"),
    "C": os.environ.get("JETSON_NODE_C_IP", "192.168.1.22"),
    "D": os.environ.get("JETSON_NODE_D_IP", "192.168.1.23"),
}
NODE_PORTS = {
    "A": int(os.environ.get("JETSON_NODE_A_PORT", "52000")),
    "B": int(os.environ.get("JETSON_NODE_B_PORT", "52001")),
    "C": int(os.environ.get("JETSON_NODE_C_PORT", "52002")),
    "D": int(os.environ.get("JETSON_NODE_D_PORT", "52003")),
}

MODEL_CHECKPOINT = os.path.join(PROJECT_ROOT, "models", "resnet56-4bfd9763.th")
MODEL_TAG = "resnet56_cifar10"
NUM_PARTITIONS = 4
NUM_TRANSFER_POINTS = 3
K_LEVELS = [0.125, 0.25, 0.50, 0.75, 0.9, 1.0]
TRANSPORT_BACKEND = "http"
TRANSPORT_READY_TIMEOUT_SEC = 180.0

DEFAULT_BATCH_SIZE = 100
DEFAULT_MAX_BATCHES = None
DEFAULT_DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "cifar10")
DEFAULT_DOWNLOAD_DATA = True
DEFAULT_WARMUP_STEPS = 50
DEFAULT_T0_RATIO = 0.5
T0_RATIOS = [0.5]
DEFAULT_MU_VALUES = [0.01, 0.1, 1.0]
DEFAULT_EPSILON = 1e-7
DEFAULT_CHANNEL_WINDOW_SIZE = 5
DEFAULT_CHANNEL_MODEL_TYPE = "mean_factor"
DEFAULT_CHANNEL_UPDATE_MODE = "warmup"
ASYNC_MAX_INFLIGHT_TASKS = 4
ASYNC_QUEUE_POLL_SEC = 0.01
DEFAULT_TX_LIMIT_MBPS = 15.0
DEFAULT_TX_BUCKET_CAPACITY_BYTES = 64 * 1024
DEFAULT_TX_BUCKET_CAPACITY_KB = DEFAULT_TX_BUCKET_CAPACITY_BYTES / 1024.0
DEFAULT_ACCURACY_ESTIMATOR_MODES = "fitting_model,stein_estimator"
DEFAULT_STEIN_SIGMA = 0.05
DEFAULT_STEIN_N = 5
DEFAULT_STEIN_FAST_MAX_BATCHES = 1
DEFAULT_DYNAMIC_TIMESLOT_SIZES = [5, 10, 25]
DEFAULT_MAX_DYNAMIC_TIMESLOT_COUNT = None
DEFAULT_MAX_LAMBDA = 1e6

ESTIMATOR_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "accuracy_estimators")
TOPK_ESTIMATOR_PATH = os.path.join(
    ESTIMATOR_MODEL_DIR,
    "jetson_resnet_3tp_poly3_flex.pkl",
)
QUANTIZATION_ESTIMATOR_PATH = os.path.join(
    ESTIMATOR_MODEL_DIR,
    "jetson_resnet_3tp_quantization_poly3_flex.pkl",
)
LLMINT8_POLICY_NAME = "fp16_int4"
LLMINT8_OUTLIER_PRECISION = "fp16"
LLMINT8_REGULAR_PRECISION = "int4"
LLMINT8_ESTIMATOR_PATH = os.path.join(
    ESTIMATOR_MODEL_DIR,
    "jetson_resnet_3tp_llmint8_{}_poly3_flex.pkl".format(LLMINT8_POLICY_NAME),
)
LLMINT8_MAPPING_PATH = os.path.join(
    ESTIMATOR_MODEL_DIR,
    "raw_accuracy_resnet56_llmint8_eta_mapping.json",
)
BASE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "resnet_single_task")


def _current_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _format_tx_limit_tag(tx_limit_mbps=None):
    if tx_limit_mbps is None:
        return "tx_unlimited"
    rate = float(tx_limit_mbps)
    if rate.is_integer():
        rate_str = str(int(rate))
    else:
        rate_str = ("{:.3f}".format(rate)).rstrip("0").rstrip(".")
    return "tx_{}mbps".format(rate_str.replace(".", "p"))


def _sanitize_tag(text):
    chars = []
    for ch in str(text).strip().lower():
        chars.append(ch if ch.isalnum() else "_")
    return "".join(chars).strip("_")


def _format_float4(value):
    return "{:.4f}".format(float(value))


def _flatten_dict(data, prefix=""):
    flat = {}
    for key, value in data.items():
        new_key = "{}_{}".format(prefix, key) if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, new_key))
        elif isinstance(value, (list, tuple)):
            flat[new_key] = ",".join(str(item) for item in value)
        else:
            flat[new_key] = value
    return flat


def _make_run_output_dir(node_id, mode, device=None, tx_limit_mbps=None):
    device_tag = str(device or "na").replace(":", "_")
    tx_tag = _format_tx_limit_tag(tx_limit_mbps)
    run_name = "{}_{}_{}_inflight{}_{}_{}".format(
        node_id,
        mode,
        device_tag,
        ASYNC_MAX_INFLIGHT_TASKS,
        tx_tag,
        _current_timestamp(),
    )
    output_dir = os.path.join(BASE_OUTPUT_DIR, run_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _t0_output_dir(base_output_dir, t0_ratio):
    dirname = "t0_{:03d}".format(int(round(float(t0_ratio) * 100.0)))
    path = os.path.join(base_output_dir, dirname)
    os.makedirs(path, exist_ok=True)
    return path


def _save_rows(output_path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def _parse_mu_values(mu_values_arg):
    if mu_values_arg is None:
        return list(DEFAULT_MU_VALUES)
    if isinstance(mu_values_arg, (list, tuple)):
        values = [float(item) for item in mu_values_arg]
    else:
        values = [float(item.strip()) for item in str(mu_values_arg).split(",") if item.strip()]
    if not values:
        raise ValueError("At least one mu value must be provided")
    return sorted(list(OrderedDict((float(item), None) for item in values).keys()))


def _parse_accuracy_estimator_modes(modes_arg):
    if modes_arg is None:
        modes_arg = DEFAULT_ACCURACY_ESTIMATOR_MODES
    if isinstance(modes_arg, (list, tuple)):
        raw_items = [str(item).strip() for item in modes_arg if str(item).strip()]
    else:
        raw_items = [item.strip() for item in str(modes_arg).split(",") if item.strip()]
    alias_map = {
        "fitting": "fitting_model",
        "fitting_model": "fitting_model",
        "fit": "fitting_model",
        "stein": "stein_estimator",
        "steins": "stein_estimator",
        "steains": "stein_estimator",
        "stein_estimator": "stein_estimator",
    }
    modes = []
    for item in raw_items:
        normalized = alias_map.get(str(item).strip().lower().replace(" ", "_"))
        if normalized is None:
            raise ValueError(
                "Unsupported accuracy estimator mode '{}'. Choose from fitting_model, stein_estimator".format(item)
            )
        if normalized not in modes:
            modes.append(normalized)
    if not modes:
        raise ValueError("At least one accuracy estimator mode must be provided")
    return modes


def _parse_dynamic_timeslot_sizes(values_arg):
    if values_arg is None:
        return [int(item) for item in DEFAULT_DYNAMIC_TIMESLOT_SIZES]
    if isinstance(values_arg, (list, tuple)):
        raw_values = list(values_arg)
    else:
        raw_values = [item.strip() for item in str(values_arg).split(",") if item.strip()]
    sizes = []
    for item in raw_values:
        value = int(item)
        if value <= 0:
            raise ValueError("dynamic timeslot size must be positive, got {}".format(value))
        sizes.append(value)
    if not sizes:
        raise ValueError("At least one dynamic timeslot size must be provided")
    return list(OrderedDict((int(item), None) for item in sizes).keys())


class Poly3AccuracyAdapter:
    mode_name = "fitting_model"

    def __init__(self, model_path):
        with open(model_path, "rb") as handle:
            state = pickle.load(handle)
        self.model = state["model"]
        self.scaler = state["scaler"]
        self.poly = state.get("poly")
        self.model_type = str(state["model_type"])
        self.n_features = int(state["n_features"])
        self.feature_names = state.get("feature_names")
        if self.model_type not in {"linear_monotonic", "poly2", "poly3"}:
            raise ValueError(
                "Unsupported estimator model_type '{}' for differentiable runtime policy".format(
                    self.model_type
                )
            )

    def predict_raw(self, eta):
        x = np.asarray(eta, dtype=float).reshape(1, -1)
        if x.shape[1] != self.n_features:
            raise ValueError("Expected {} eta values, got {}".format(self.n_features, x.shape[1]))
        z = self.scaler.transform(x)
        if self.poly is not None:
            z = self.poly.transform(z)
        return float(self.model.predict(z)[0])

    def predict(self, eta):
        return float(np.clip(self.predict_raw(eta), 0.0, 1.0))

    def gradient(self, eta):
        x = np.asarray(eta, dtype=float).reshape(-1)
        if x.shape[0] != self.n_features:
            raise ValueError("Expected {} eta values, got {}".format(self.n_features, x.shape[0]))
        mean = np.asarray(getattr(self.scaler, "mean_", np.zeros(self.n_features)), dtype=float)
        scale = np.asarray(getattr(self.scaler, "scale_", np.ones(self.n_features)), dtype=float)
        scale = np.where(scale == 0.0, 1.0, scale)
        z = (x - mean) / scale

        if self.poly is None:
            coef = np.asarray(self.model.coef_, dtype=float).reshape(-1)
            grad = coef / scale
        else:
            coef = np.asarray(self.model.coef_, dtype=float).reshape(-1)
            powers = np.asarray(self.poly.powers_, dtype=int)
            grad_z = np.zeros(self.n_features, dtype=float)
            for feature_idx in range(self.n_features):
                partial = 0.0
                for term_idx, power_vec in enumerate(powers):
                    exponent = int(power_vec[feature_idx])
                    if exponent == 0:
                        continue
                    term = coef[term_idx] * exponent
                    for dim_idx, dim_power in enumerate(power_vec):
                        p = int(dim_power)
                        if dim_idx == feature_idx:
                            if p - 1 > 0:
                                term *= z[dim_idx] ** (p - 1)
                        else:
                            if p > 0:
                                term *= z[dim_idx] ** p
                    partial += term
                grad_z[feature_idx] = partial
            grad = grad_z / scale

        raw = self.predict_raw(x)
        if raw <= 0.0 or raw >= 1.0:
            return np.zeros_like(grad)
        return grad.astype(float)


class SteinAccuracyEstimatorAdapter:
    mode_name = "stein_estimator"

    def __init__(self, accuracy_callable, num_links, eta_min, sigma=DEFAULT_STEIN_SIGMA, N=DEFAULT_STEIN_N):
        self.accuracy_callable = accuracy_callable
        self.num_links = int(num_links)
        self.eta_min = np.asarray(eta_min, dtype=float).reshape(-1)
        if self.eta_min.shape[0] != self.num_links:
            raise ValueError("Expected {} eta values, got {}".format(self.num_links, self.eta_min.shape[0]))
        self.sigma = float(sigma)
        self.N = int(N)

    def _clamp_eta(self, eta):
        eta_arr = np.asarray(eta, dtype=float).reshape(-1)
        if eta_arr.shape[0] != self.num_links:
            raise ValueError("Expected {} eta values, got {}".format(self.num_links, eta_arr.shape[0]))
        return np.clip(eta_arr, self.eta_min, 1.0).astype(float)

    def predict(self, eta):
        return float(self.accuracy_callable(self._clamp_eta(eta)))

    def gradient(self, eta):
        eta_t = torch.tensor(self._clamp_eta(eta), dtype=torch.float32)

        def f(x):
            eta_np = self._clamp_eta(x.detach().cpu().numpy())
            return torch.tensor(float(self.accuracy_callable(eta_np)), dtype=torch.float32, device=x.device)

        g = torch.zeros_like(eta_t)
        for _ in range(self.N):
            z = torch.randn_like(eta_t)
            g += z * (f(eta_t + self.sigma * z) - f(eta_t - self.sigma * z))
        return (g / (2.0 * self.N * self.sigma)).detach().cpu().numpy().astype(float)


class ResNetHead(nn.Module):
    def __init__(self, avgpool, flatten, linear):
        super().__init__()
        self.avgpool = avgpool
        self.flatten = flatten
        self.linear = linear

    def forward(self, x):
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


class ResNetUnitPartition(nn.Module):
    def __init__(self, units, unit_names, partition_idx, is_last_partition):
        super().__init__()
        self.units = nn.ModuleList(list(units))
        self.unit_names = list(unit_names)
        self.partition_idx = int(partition_idx)
        self.is_last_partition = bool(is_last_partition)
        self.partition_info = {
            "partition_idx": int(partition_idx),
            "unit_names": list(unit_names),
            "is_last_partition": bool(is_last_partition),
        }

    def forward(self, x):
        for module in self.units:
            x = module(x)
        return x


def conv2d_macs(module, input_shape):
    batch, in_channels, in_h, in_w = input_shape
    out_channels = module.out_channels
    kernel_h, kernel_w = module.kernel_size
    stride_h, stride_w = module.stride
    pad_h, pad_w = module.padding
    dil_h, dil_w = module.dilation
    groups = module.groups
    out_h = math.floor((in_h + 2 * pad_h - dil_h * (kernel_h - 1) - 1) / stride_h + 1)
    out_w = math.floor((in_w + 2 * pad_w - dil_w * (kernel_w - 1) - 1) / stride_w + 1)
    macs = batch * out_h * out_w * out_channels * (in_channels // groups) * kernel_h * kernel_w
    return float(macs), (batch, out_channels, out_h, out_w)


def linear_macs(module, input_shape):
    batch = input_shape[0]
    macs = batch * module.in_features * module.out_features
    return float(macs), (batch, module.out_features)


def estimate_unit_cost(unit, input_shape):
    if isinstance(unit, nn.Sequential):
        total = 0.0
        shape = input_shape
        for child in unit.children():
            child_cost, shape = estimate_unit_cost(child, shape)
            total += child_cost
        return total, shape

    if isinstance(unit, resnet20.BasicBlock):
        total = 0.0
        shape = input_shape
        cost, shape = conv2d_macs(unit.conv1, shape)
        total += cost
        cost, shape = conv2d_macs(unit.conv2, shape)
        total += cost
        for child in unit.shortcut.children():
            if isinstance(child, nn.Conv2d):
                cost, _ = conv2d_macs(child, input_shape)
                total += cost
        return total, shape

    if isinstance(unit, nn.Conv2d):
        return conv2d_macs(unit, input_shape)

    if isinstance(unit, ResNetHead):
        batch, channels, _, _ = input_shape
        return linear_macs(unit.linear, (batch, channels))

    return 0.0, input_shape


def build_equal_cost_ranges(costs, num_partitions):
    if num_partitions <= 0:
        raise ValueError("num_partitions must be positive")
    if num_partitions > len(costs):
        raise ValueError("num_partitions cannot exceed number of units")
    total_cost = float(sum(costs))
    if total_cost <= 0:
        step = len(costs) // num_partitions
        remainder = len(costs) % num_partitions
        ranges = []
        start = 0
        for partition_idx in range(num_partitions):
            width = step + (1 if partition_idx < remainder else 0)
            end = start + width
            ranges.append((start, end))
            start = end
        return ranges

    prefix = [0.0]
    for cost in costs:
        prefix.append(prefix[-1] + float(cost))

    ranges = []
    start = 0
    for partition_idx in range(num_partitions - 1):
        target_cost = total_cost * float(partition_idx + 1) / float(num_partitions)
        min_end = start + 1
        max_end = len(costs) - (num_partitions - partition_idx - 1)
        best_end = min_end
        best_err = None
        for end in range(min_end, max_end + 1):
            err = abs(prefix[end] - target_cost)
            if best_err is None or err < best_err:
                best_err = err
                best_end = end
        ranges.append((start, best_end))
        start = best_end

    ranges.append((start, len(costs)))
    return ranges


class ResNet56PartitionFactory(object):
    def __init__(self, checkpoint_path=MODEL_CHECKPOINT, device="cuda", num_partitions=NUM_PARTITIONS):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.num_partitions = int(num_partitions)
        self.full_model = self._load_full_model()
        self.units, self.unit_names = self._build_units()
        self.unit_costs = self._estimate_unit_costs()
        self.partition_ranges = build_equal_cost_ranges(self.unit_costs, self.num_partitions)
        self.partition_plan = self._build_partition_plan()

    def _load_full_model(self):
        model = resnet20.resnet56()
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        normalized = OrderedDict()
        for key, value in state_dict.items():
            normalized[key[7:] if key.startswith("module.") else key] = value
        model.load_state_dict(normalized)
        model.to(self.device)
        model.eval()
        return model

    def _build_units(self):
        units = []
        names = []
        units.append(nn.Sequential(self.full_model.conv1, self.full_model.bn1, self.full_model.relu))
        names.append("stem")
        for idx, block in enumerate(self.full_model.layer1):
            units.append(block)
            names.append("layer1.{}".format(idx))
        for idx, block in enumerate(self.full_model.layer2):
            units.append(block)
            names.append("layer2.{}".format(idx))
        for idx, block in enumerate(self.full_model.layer3):
            units.append(block)
            names.append("layer3.{}".format(idx))
        units.append(ResNetHead(self.full_model.avgpool, self.full_model.flatten, self.full_model.linear))
        names.append("head")
        return units, names

    def _estimate_unit_costs(self):
        costs = []
        shape = (1, 3, 32, 32)
        for unit in self.units:
            cost, shape = estimate_unit_cost(unit, shape)
            costs.append(max(cost, 1.0))
        return costs

    def _build_partition_plan(self):
        plan = []
        for partition_idx, (start, end) in enumerate(self.partition_ranges):
            plan.append(
                {
                    "partition_idx": partition_idx,
                    "node_id": NODE_ORDER[partition_idx],
                    "unit_start": start,
                    "unit_end": end,
                    "unit_names": list(self.unit_names[start:end]),
                    "total_cost": float(sum(self.unit_costs[start:end])),
                    "is_last_partition": partition_idx == self.num_partitions - 1,
                }
            )
        return plan

    def build_partition(self, partition_idx):
        start, end = self.partition_ranges[partition_idx]
        node = ResNetUnitPartition(
            units=self.units[start:end],
            unit_names=self.unit_names[start:end],
            partition_idx=partition_idx,
            is_last_partition=(partition_idx == self.num_partitions - 1),
        ).to(self.device)
        node.eval()
        node.partition_info["node_id"] = NODE_ORDER[partition_idx]
        node.partition_info["checkpoint_path"] = self.checkpoint_path
        return node

    def build_partition_for_node(self, node_id):
        return self.build_partition(NODE_ORDER.index(node_id))


def load_cifar10_batches(
    batch_size=DEFAULT_BATCH_SIZE,
    max_batches=DEFAULT_MAX_BATCHES,
    data_root=DEFAULT_DATA_ROOT,
    download=DEFAULT_DOWNLOAD_DATA,
):
    try:
        import torchvision
        import torchvision.transforms as transforms
    except ImportError as exc:
        raise RuntimeError("torchvision is required for CIFAR-10 loading") from exc

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    dataset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=download,
        transform=transform,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    batches = []
    for batch_idx, (images, labels) in enumerate(loader):
        batches.append({"batch_idx": batch_idx, "images": images, "labels": labels})
        if max_batches is not None and len(batches) >= int(max_batches):
            break
    return batches


def _identity_activation_payload(tensor_cpu):
    return {"mode": "identity", "tensor": tensor_cpu}


def build_activation_payload(tensor, compression_param, compressor_name, feature_k_value):
    tensor_cpu = tensor.detach().cpu()
    original_bytes = int(tensor_cpu.numel() * tensor_cpu.element_size())

    if compressor_name in ("identity", None) or compression_param is None:
        payload = _identity_activation_payload(tensor_cpu)
        compressed_bytes = original_bytes
    else:
        compressor = get_compressor(compressor_name)
        compressed = compressor.compress(tensor_cpu, compression_param)
        compressed_bytes = None
        if hasattr(compressor, "get_compressed_size"):
            try:
                compressed_bytes = int(compressor.get_compressed_size(compressed))
            except Exception:
                compressed_bytes = None
        if compressed_bytes is None:
            compressed_bytes = int(compressed.get("compressed_bytes", original_bytes))
        payload = {
            "mode": "compressed",
            "compressor_name": compressor_name,
            "payload": compressed,
        }

    ratio = float(compressed_bytes) / float(original_bytes) if original_bytes > 0 else 1.0
    return payload, {
        "original_bytes": original_bytes,
        "compressed_bytes": compressed_bytes,
        "compression_ratio": ratio,
        "k_value": float(feature_k_value),
    }


def restore_activation_payload(payload, device):
    mode = payload.get("mode", "compressed")
    if mode == "identity":
        restored = payload["tensor"]
    else:
        compressor = get_compressor(payload["compressor_name"])
        restored = compressor.decompress(payload["payload"])
    return restored.to(device)


class ResNetFastAccuracyEvaluator:
    def __init__(self, checkpoint_path, device, batches):
        self.device = device
        factory = ResNet56PartitionFactory(checkpoint_path=checkpoint_path, device=device)
        self.partitions = [factory.build_partition(partition_idx) for partition_idx in range(NUM_PARTITIONS)]
        self.batches = list(batches)

    def evaluate(self, execution_plan):
        if not self.batches:
            return 0.0
        compressor_name = str(execution_plan["codec_name"])
        compression_params_list = list(execution_plan["compression_params_list"])
        feature_k_values = list(execution_plan["execution_feature_k_values"])
        total_correct = 0
        total_seen = 0
        with torch.no_grad():
            for batch in self.batches:
                hidden = batch["images"].to(self.device)
                labels = batch["labels"].to(self.device)
                for partition_idx, partition in enumerate(self.partitions):
                    hidden = partition(hidden)
                    if partition_idx < NUM_TRANSFER_POINTS:
                        payload, _ = build_activation_payload(
                            hidden,
                            compression_param=compression_params_list[partition_idx],
                            compressor_name=compressor_name,
                            feature_k_value=feature_k_values[partition_idx],
                        )
                        hidden = restore_activation_payload(payload, self.device)
                logits = hidden
                predictions = torch.argmax(logits, dim=1)
                total_correct += int((predictions == labels).sum().item())
                total_seen += int(labels.numel())
        return (float(total_correct) / float(total_seen)) if total_seen > 0 else 0.0


def compute_pipeline_stage_delays(eta, a_t, tau_t, c_t):
    stage_delays = []
    for idx in range(NUM_TRANSFER_POINTS):
        comm = float(a_t[idx]) * float(eta[idx]) / float(c_t[idx]) if float(c_t[idx]) > 0 else float("inf")
        stage_delays.append(max(float(tau_t[idx]), comm))
    stage_delays.append(float(tau_t[NUM_TRANSFER_POINTS]))
    return stage_delays


def compute_pipeline_delay(eta, a_t, tau_t, c_t):
    return max(compute_pipeline_stage_delays(eta, a_t, tau_t, c_t))


def _sanitize_lambda_value(value, epsilon, upper=DEFAULT_MAX_LAMBDA):
    try:
        numeric = float(value)
    except Exception:
        logging.warning("[control] lambda sanitize: invalid value=%r, fallback to epsilon=%.6g", value, float(epsilon))
        numeric = float(epsilon)
    if not np.isfinite(numeric):
        logging.warning("[control] lambda sanitize: non-finite value=%r, clipping to upper=%.6g", value, float(upper))
        numeric = float(upper)
    clipped = float(min(max(float(epsilon), numeric), float(upper)))
    if clipped != numeric:
        logging.warning(
            "[control] lambda sanitize: clipping value=%.6g into [%.6g, %.6g] -> %.6g",
            float(numeric),
            float(epsilon),
            float(upper),
            float(clipped),
        )
    return clipped


def _finite_mean(values, fallback):
    raw = np.asarray(list(values), dtype=float)
    arr = raw[np.isfinite(raw)]
    if arr.size != raw.size:
        logging.warning(
            "[control] finite mean: filtered %d non-finite values before averaging",
            int(raw.size - arr.size),
        )
    if arr.size == 0:
        logging.warning("[control] finite mean: no finite values left, fallback=%.6g", float(fallback))
        return float(fallback)
    return float(np.mean(arr))


class BaseOnlinePolicy(object):
    def __init__(self, policy_key, display_name, eta_min):
        self.policy_key = str(policy_key)
        self.policy_name = str(display_name)
        self.eta_min = np.asarray(eta_min, dtype=float)
        self.lambda_t = 0.0
        self.mu = None

    def select_eta(self, a_t, tau_t, c_hat, deadline):
        raise NotImplementedError

    def update_dual(self, actual_delay, deadline):
        return None

    def initialize_from_channel_estimator(self, channel_estimator):
        return None

    def observe_channel(self, tp_stats):
        return None


class EstimatedCSISingleBaselinePolicy(BaseOnlinePolicy):
    def __init__(self, policy_key, display_name, eta_min, acc_model=None, mode="last", window_size=5):
        super().__init__(policy_key, display_name, eta_min)
        self.acc_model = acc_model
        self.mode = str(mode)
        self.window_size = max(1, int(window_size))
        self.link_history = [[] for _ in range(NUM_TRANSFER_POINTS)]

    def initialize_from_channel_estimator(self, channel_estimator):
        links = getattr(channel_estimator, "links", None)
        if links is None:
            return
        histories = []
        for link in list(links)[:NUM_TRANSFER_POINTS]:
            histories.append(
                [float(x) for x in getattr(link, "history", []) if np.isfinite(x) and float(x) > 0.0]
            )
        if len(histories) == NUM_TRANSFER_POINTS:
            self.link_history = histories

    def observe_channel(self, tp_stats):
        for idx, (bytes_sent, elapsed_sec) in enumerate(tp_stats[:NUM_TRANSFER_POINTS]):
            if elapsed_sec > 0.0 and bytes_sent > 0.0:
                throughput = float(bytes_sent) / float(elapsed_sec)
                if np.isfinite(throughput) and throughput > 0.0:
                    self.link_history[idx].append(float(throughput))

    def _estimate_c_hat(self, fallback):
        fallback_arr = np.asarray(fallback, dtype=float).reshape(NUM_TRANSFER_POINTS)
        estimates = []
        for idx in range(NUM_TRANSFER_POINTS):
            history = [float(x) for x in self.link_history[idx] if np.isfinite(x) and float(x) > 0.0]
            if not history:
                estimates.append(max(float(fallback_arr[idx]), 1e-9))
                continue
            if self.mode == "last":
                estimates.append(float(history[-1]))
            elif self.mode == "min":
                estimates.append(float(np.min(history)))
            else:
                estimates.append(float(np.mean(history[-self.window_size :])))
        return np.maximum(np.asarray(estimates, dtype=float), 1e-9)

    def select_eta(self, a_t, tau_t, c_hat, deadline):
        c_hat_est = self._estimate_c_hat(c_hat)
        deadline = float(deadline)
        target_rate_hz = (1.0 / deadline) if deadline > 0.0 else np.nan
        eta = np.divide(
            c_hat_est,
            target_rate_hz * np.asarray(a_t, dtype=float),
            out=np.ones_like(self.eta_min, dtype=float),
            where=(target_rate_hz * np.asarray(a_t, dtype=float)) > 0.0,
        )
        eta = np.minimum(1.0, eta)
        eta = np.maximum(eta, self.eta_min)
        pred_delay = compute_pipeline_delay(eta, a_t, tau_t, c_hat_est)
        pred_acc = self.acc_model.predict(eta) if self.acc_model is not None else None
        return {
            "requested_eta": eta,
            "predicted_delay": pred_delay,
            "predicted_accuracy": pred_acc,
            "solver_z": pred_delay,
        }


class MyopicSinglePolicy(EstimatedCSISingleBaselinePolicy):
    def __init__(self, eta_min, acc_model=None):
        super().__init__("myopic_baseline", "No-CSI Baseline: myopic", eta_min, acc_model=acc_model, mode="last")


class ConservativeSinglePolicy(EstimatedCSISingleBaselinePolicy):
    def __init__(self, eta_min, acc_model=None):
        super().__init__("conservative_baseline", "No-CSI Baseline: conservative", eta_min, acc_model=acc_model, mode="min")


class MovingAverageSinglePolicy(EstimatedCSISingleBaselinePolicy):
    def __init__(self, eta_min, acc_model=None, window_size=5):
        super().__init__(
            "moving_average_baseline",
            "No-CSI Baseline: moving average",
            eta_min,
            acc_model=acc_model,
            mode="moving_average",
            window_size=window_size,
        )


class NoCompressionBaselinePolicy(BaseOnlinePolicy):
    def __init__(self, eta_min):
        super().__init__("no_compression_baseline", "No Compression Baseline", eta_min)
        self.mu = None

    def select_eta(self, a_t, tau_t, c_hat, deadline):
        eta = np.ones_like(self.eta_min, dtype=float)
        return {
            "requested_eta": eta,
            "predicted_delay": compute_pipeline_delay(eta, a_t, tau_t, c_hat),
            "predicted_accuracy": None,
            "solver_z": compute_pipeline_delay(eta, a_t, tau_t, c_hat),
        }


class MaxCompressionBaselinePolicy(BaseOnlinePolicy):
    def __init__(self, eta_min):
        super().__init__("max_compression_baseline", "Max Compression Baseline", eta_min)
        self.mu = None

    def select_eta(self, a_t, tau_t, c_hat, deadline):
        eta = self.eta_min.copy()
        return {
            "requested_eta": eta,
            "predicted_delay": compute_pipeline_delay(eta, a_t, tau_t, c_hat),
            "predicted_accuracy": None,
            "solver_z": compute_pipeline_delay(eta, a_t, tau_t, c_hat),
        }


class NoCSISinglePolicy(BaseOnlinePolicy):
    def __init__(self, eta_min, acc_model, mu, epsilon=DEFAULT_EPSILON):
        display_name = "Non-CSI (mu={})".format(("{:.6g}".format(float(mu))))
        super().__init__("no_csi_single", display_name, eta_min)
        self.acc_model = acc_model
        self.mu = float(mu)
        self.epsilon = float(epsilon)
        self.lambda_t = float(epsilon)
        self._last_solution = None

    def _objective(self, x):
        eta = x[:-1]
        z = float(x[-1])
        acc = self.acc_model.predict(eta)
        grad_acc = self.acc_model.gradient(eta)
        obj = -acc + self.mu * self.lambda_t * z
        grad = np.zeros_like(x)
        grad[:-1] = -grad_acc
        grad[-1] = self.mu * self.lambda_t
        return obj, grad

    def select_eta(self, a_t, tau_t, c_hat, deadline):
        self.lambda_t = _sanitize_lambda_value(self.lambda_t, self.epsilon)
        c_hat_safe = np.asarray(c_hat, dtype=float).copy()
        invalid_mask = (~np.isfinite(c_hat_safe)) | (c_hat_safe <= 0.0)
        if np.any(invalid_mask):
            logging.warning(
                "[policy=%s] invalid c_hat detected: raw=%s; replacing invalid entries with 1e-9",
                self.policy_name,
                [float(x) if np.isfinite(x) else str(x) for x in np.asarray(c_hat, dtype=float)],
            )
            c_hat_safe[invalid_mask] = 1e-9
        lower_z = max(float(np.max(tau_t)), 0.0)
        bounds = [(float(self.eta_min[i]), 1.0) for i in range(NUM_TRANSFER_POINTS)] + [(lower_z, None)]

        if self._last_solution is not None and len(self._last_solution) == len(bounds):
            x0 = np.asarray(self._last_solution, dtype=float).copy()
            for idx, (lower, upper) in enumerate(bounds):
                x0[idx] = max(lower, x0[idx])
                if upper is not None:
                    x0[idx] = min(upper, x0[idx])
        else:
            x0 = np.array([bounds[idx][0] for idx in range(len(bounds))], dtype=float)

        constraints = []
        for idx in range(NUM_TRANSFER_POINTS):
            beta = float(a_t[idx]) / float(c_hat_safe[idx]) if float(c_hat_safe[idx]) > 0 else 1e12

            def make_constraint(local_idx=idx, local_beta=beta):
                return {
                    "type": "ineq",
                    "fun": lambda x: float(x[-1]) - local_beta * float(x[local_idx]),
                    "jac": lambda x: np.array(
                        [
                            (-local_beta if j == local_idx else (1.0 if j == len(x) - 1 else 0.0))
                            for j in range(len(x))
                        ],
                        dtype=float,
                    ),
                }

            constraints.append(make_constraint())

        result = minimize(
            fun=self._objective,
            x0=x0,
            method="SLSQP",
            jac=True,
            bounds=bounds,
            constraints=constraints,
        )
        if not result.success:
            logging.warning(
                "[policy=%s] SLSQP failed: success=%s status=%s message=%s mu=%.6g lambda=%.6g deadline=%.6g "
                "lower_z=%.6g x0=%s c_hat=%s a=%s tau=%s",
                self.policy_name,
                result.success,
                getattr(result, "status", None),
                getattr(result, "message", ""),
                float(self.mu),
                float(self.lambda_t),
                float(deadline),
                float(lower_z),
                [float(item) for item in np.asarray(x0, dtype=float)],
                [float(item) for item in np.asarray(c_hat_safe, dtype=float)],
                [float(item) for item in np.asarray(a_t, dtype=float)],
                [float(item) for item in np.asarray(tau_t, dtype=float)],
            )

        use_result = bool(result.success) and getattr(result, "x", None) is not None and np.all(
            np.isfinite(np.asarray(result.x, dtype=float))
        )
        if not use_result:
            logging.warning(
                "[policy=%s] Falling back to feasible point due to invalid solver output: raw_x=%s",
                self.policy_name,
                None if getattr(result, "x", None) is None else [float(item) for item in np.asarray(result.x, dtype=float)],
            )
        selected_x = np.asarray(result.x, dtype=float) if use_result else x0
        eta = np.asarray(selected_x[:-1], dtype=float)
        eta = np.clip(eta, self.eta_min, 1.0)
        z = float(max(lower_z, selected_x[-1]))
        self._last_solution = np.concatenate([eta, np.array([z], dtype=float)])
        return {
            "requested_eta": eta,
            "predicted_delay": compute_pipeline_delay(eta, a_t, tau_t, c_hat_safe),
            "predicted_accuracy": self.acc_model.predict(eta),
            "solver_z": z,
        }

    def update_dual(self, actual_delay, deadline):
        actual_delay = float(actual_delay)
        deadline = float(deadline)
        if not np.isfinite(actual_delay) or not np.isfinite(deadline) or deadline <= 0.0:
            logging.warning(
                "[policy=%s] skipping dual update due to invalid actual_delay=%.6g deadline=%.6g",
                self.policy_name,
                float(actual_delay),
                float(deadline),
            )
            self.lambda_t = _sanitize_lambda_value(self.lambda_t, self.epsilon)
            return
        updated = float(self.lambda_t) + actual_delay - deadline
        self.lambda_t = _sanitize_lambda_value(updated, self.epsilon)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_llmint8_mapping_entries(mapping_path, llm_policy, outlier_precision, regular_precision):
    payload = _load_json(mapping_path)
    policies = payload.get("policies", [])
    for policy in policies:
        if str(policy.get("llm_policy", "")).strip().lower() != str(llm_policy).strip().lower():
            continue
        if str(policy.get("outlier_precision", "")).strip().lower() != str(outlier_precision).strip().lower():
            continue
        if str(policy.get("regular_precision", "")).strip().lower() != str(regular_precision).strip().lower():
            continue
        entries = policy.get("llmint8_eta_to_codec_mapping", {}).get("entries", [])
        if entries:
            return validate_llmint8_mapping_entries(entries, num_links=NUM_TRANSFER_POINTS)
    for policy in policies:
        entries = policy.get("llmint8_eta_to_codec_mapping", {}).get("entries", [])
        if entries:
            return validate_llmint8_mapping_entries(entries, num_links=NUM_TRANSFER_POINTS)
    raise ValueError("No llmint8 mapping entries found in {}".format(mapping_path))


def _eta_min_from_mapping_entries(entries):
    feature_matrix = np.asarray([entry["feature_k_values"] for entry in entries], dtype=float)
    return np.min(feature_matrix, axis=0)


def _normalize_profile_name(name):
    normalized = str(name).strip().lower().replace("-", "_")
    aliases = {
        "topk": "topk",
        "quant": "quantization",
        "quantization": "quantization",
        "llmint8": "llmint8_{}".format(LLMINT8_POLICY_NAME),
        "llm_int8": "llmint8_{}".format(LLMINT8_POLICY_NAME),
        "llmint8_{}".format(LLMINT8_POLICY_NAME): "llmint8_{}".format(LLMINT8_POLICY_NAME),
    }
    return aliases.get(normalized, normalized)


def _codec_profiles():
    llmint8_mapping_entries = _load_llmint8_mapping_entries(
        mapping_path=LLMINT8_MAPPING_PATH,
        llm_policy=LLMINT8_POLICY_NAME,
        outlier_precision=LLMINT8_OUTLIER_PRECISION,
        regular_precision=LLMINT8_REGULAR_PRECISION,
    )
    return OrderedDict(
        [
            (
                "topk",
                {
                    "display_name": "TopK",
                    "codec_name": "topk",
                    "compressor_name": "topk",
                    "estimator_path": TOPK_ESTIMATOR_PATH,
                    "eta_min": np.asarray([min(K_LEVELS)] * NUM_TRANSFER_POINTS, dtype=float),
                    "llmint8_mapping_entries": [],
                    "outlier_precision": None,
                    "regular_precision": None,
                },
            ),
            (
                "quantization",
                {
                    "display_name": "Quantization",
                    "codec_name": "quantization",
                    "compressor_name": "quantization",
                    "estimator_path": QUANTIZATION_ESTIMATOR_PATH,
                    "eta_min": np.asarray([0.125] * NUM_TRANSFER_POINTS, dtype=float),
                    "llmint8_mapping_entries": [],
                    "outlier_precision": None,
                    "regular_precision": None,
                },
            ),
            (
                "llmint8_{}".format(LLMINT8_POLICY_NAME),
                {
                    "display_name": "LLMInt8_{}".format(LLMINT8_POLICY_NAME),
                    "codec_name": "llmint8",
                    "compressor_name": "llmint8",
                    "estimator_path": LLMINT8_ESTIMATOR_PATH,
                    "eta_min": _eta_min_from_mapping_entries(llmint8_mapping_entries),
                    "llmint8_mapping_entries": llmint8_mapping_entries,
                    "outlier_precision": LLMINT8_OUTLIER_PRECISION,
                    "regular_precision": LLMINT8_REGULAR_PRECISION,
                },
            ),
        ]
    )


def _build_execution_plan(profile_spec, requested_eta):
    requested_eta = [float(item) for item in requested_eta]
    if profile_spec["codec_name"] == "topk":
        return resolve_codec_execution_plan(codec_name="topk", eta=requested_eta)
    if profile_spec["codec_name"] == "quantization":
        return resolve_codec_execution_plan(codec_name="quantization", eta=requested_eta)
    return resolve_codec_execution_plan(
        codec_name="llmint8",
        eta=requested_eta,
        outlier_precision=profile_spec["outlier_precision"],
        regular_precision=profile_spec["regular_precision"],
        llmint8_mapping_entries=profile_spec["llmint8_mapping_entries"],
    )


def _build_accuracy_estimator(
    *,
    profile_spec,
    estimator_mode,
    checkpoint_path,
    device,
    batches,
    stein_sigma,
    stein_N,
):
    if estimator_mode == "fitting_model":
        estimator_path = profile_spec["estimator_path"]
        if not os.path.exists(estimator_path):
            return None
        return Poly3AccuracyAdapter(estimator_path)
    if estimator_mode == "stein_estimator":
        evaluator = ResNetFastAccuracyEvaluator(
            checkpoint_path=checkpoint_path,
            device=device,
            batches=batches,
        )
        return SteinAccuracyEstimatorAdapter(
            accuracy_callable=lambda eta: evaluator.evaluate(_build_execution_plan(profile_spec, eta)),
            num_links=NUM_TRANSFER_POINTS,
            eta_min=profile_spec["eta_min"],
            sigma=stein_sigma,
            N=stein_N,
        )
    raise ValueError("Unsupported accuracy_estimator_mode '{}'".format(estimator_mode))


def build_channel_estimator(
    device="cuda",
    model_type=DEFAULT_CHANNEL_MODEL_TYPE,
    update_mode=DEFAULT_CHANNEL_UPDATE_MODE,
    window_size=DEFAULT_CHANNEL_WINDOW_SIZE,
):
    estimator = ChannelEstimator(
        n_links=NUM_TRANSFER_POINTS,
        model_type=model_type,
        window_size=window_size,
        update_mode=update_mode,
        default_bandwidth=10e6,
        device=device,
    )
    logging.info("Initialized channel estimator: %s", estimator.summary())
    return estimator


def _build_seeded_channel_estimator(
    warmup_tp_stats,
    device,
    model_type,
    update_mode,
    window_size,
):
    estimator = build_channel_estimator(
        device=device,
        model_type=model_type,
        update_mode=update_mode,
        window_size=window_size,
    )
    for tp_stats in warmup_tp_stats:
        estimator.observe_task(tp_stats, refit=False)
    estimator.fit()
    return estimator


def _profile_from_partition(partition, input_tensor):
    t_comp = time.perf_counter()
    with torch.no_grad():
        output = partition(input_tensor)
    return output, time.perf_counter() - t_comp


def _worker_profile(node_id, compute_sec, restore_sec, prepare_sec, send_bytes, send_sec):
    service_total = float(compute_sec) + float(restore_sec) + float(prepare_sec)
    return {
        "node_id": node_id,
        "compute_sec": float(compute_sec),
        "restore_sec": float(restore_sec),
        "prepare_sec": float(prepare_sec),
        "send_bytes": int(send_bytes),
        "send_sec": float(send_sec),
        "service_total_sec": service_total,
    }


def _receiver_side_tp_stats(envelope, node_id, profile=None, task_id=None):
    payload_bytes = max(int(envelope.get("payload_bytes", 0)), 0)
    transfer_elapsed_sec = max(float(envelope.get("transfer_elapsed_sec", 0.0)), 1e-9)
    entry = {
        "task_id": task_id,
        "node_id": node_id,
        "tp_stats": {"bytes": int(payload_bytes), "elapsed": float(transfer_elapsed_sec)},
    }
    if profile is not None:
        prof = dict(profile)
        prof["send_bytes"] = int(payload_bytes)
        prof["send_sec"] = float(transfer_elapsed_sec)
        entry["profile"] = prof
    return entry


def _collect_result_payload(transport):
    try:
        return transport.receive(msg_type=MessageType.FINAL_RESULT, source="D", timeout=0.0)
    except TimeoutError:
        return None


def _dispatch_async_task(transport, partition, batch, task_id, execution_plan, device):
    images = batch["images"].to(device)
    t_comp = time.perf_counter()
    with torch.no_grad():
        hidden = partition(images)
    compute_sec = time.perf_counter() - t_comp

    t_prepare = time.perf_counter()
    exec_eta = list(execution_plan["execution_feature_k_values"])
    compression_params_list = list(execution_plan["compression_params_list"])
    compressor_name = str(execution_plan["codec_name"])
    hidden_comp, comp_stats = build_activation_payload(
        hidden,
        compression_param=compression_params_list[0],
        compressor_name=compressor_name,
        feature_k_value=exec_eta[0],
    )
    outgoing = {
        "task_id": task_id,
        "batch_idx": batch["batch_idx"],
        "compressor_name": compressor_name,
        "compression_params_list": compression_params_list,
        "feature_k_values": exec_eta,
        "requested_eta": list(execution_plan["requested_eta"]),
        "mapping_info": execution_plan.get("mapping_info"),
        "activation_comp": hidden_comp,
        "tp_original_bytes": [comp_stats["original_bytes"], 0, 0],
        "worker_stats_chain": {},
    }
    prepare_sec = time.perf_counter() - t_prepare

    bytes_sent, send_sec = transport.send("B", MessageType.TASK_INPUT, outgoing)
    a_profile = _worker_profile("A", compute_sec, 0.0, prepare_sec, bytes_sent, send_sec)
    return {
        "task_id": task_id,
        "batch_idx": batch["batch_idx"],
        "labels": batch["labels"].clone(),
        "requested_eta": list(execution_plan["requested_eta"]),
        "executed_eta": exec_eta,
        "compression_params_list": compression_params_list,
        "compressor_name": compressor_name,
        "mapping_info": execution_plan.get("mapping_info"),
        "started_at": time.perf_counter(),
        "tp_original_bytes": list(outgoing["tp_original_bytes"]),
        "tp_stats": {"A": {"bytes": int(bytes_sent), "elapsed": float(send_sec)}},
        "profiles": {"A": a_profile},
        "worker_stats": {},
    }


def _run_worker_b(transport, partition, device):
    while True:
        envelope = transport.receive(msg_type=MessageType.TASK_INPUT, source=None)
        payload = envelope["payload"]
        if payload.get("cmd") == "shutdown":
            break
        t_restore = time.perf_counter()
        hidden = restore_activation_payload(payload["activation_comp"], device)
        restore_sec = time.perf_counter() - t_restore
        hidden, compute_sec = _profile_from_partition(partition, hidden)
        t_prepare = time.perf_counter()
        hidden_comp, stats = build_activation_payload(
            hidden,
            compression_param=payload["compression_params_list"][1],
            compressor_name=payload["compressor_name"],
            feature_k_value=payload["feature_k_values"][1],
        )
        outgoing = dict(payload)
        outgoing["activation_comp"] = hidden_comp
        outgoing["tp_original_bytes"][1] = stats["original_bytes"]
        prepare_sec = time.perf_counter() - t_prepare
        outgoing["worker_stats_chain"] = dict(payload.get("worker_stats_chain", {}))
        outgoing["worker_stats_chain"]["A"] = _receiver_side_tp_stats(
            envelope,
            "A",
            task_id=payload["task_id"],
        )
        outgoing["worker_stats_chain"]["B"] = {
            "task_id": payload["task_id"],
            "node_id": "B",
            "profile": _worker_profile("B", compute_sec, restore_sec, prepare_sec, 0, 0.0),
        }
        bytes_sent, send_sec = transport.send("C", MessageType.ENCODER_OUTPUT, outgoing)
        if bytes_sent <= 0 or send_sec <= 0.0 or not np.isfinite(send_sec):
            logging.warning(
                "[B] invalid tp_stats task_id=%s bytes_sent=%s send_sec=%s",
                payload["task_id"],
                bytes_sent,
                send_sec,
            )


def _run_worker_c(transport, partition, device):
    while True:
        envelope = transport.receive(msg_type=MessageType.ENCODER_OUTPUT, source=None)
        payload = envelope["payload"]
        if payload.get("cmd") == "shutdown":
            break
        t_restore = time.perf_counter()
        hidden = restore_activation_payload(payload["activation_comp"], device)
        restore_sec = time.perf_counter() - t_restore
        hidden, compute_sec = _profile_from_partition(partition, hidden)
        t_prepare = time.perf_counter()
        hidden_comp, stats = build_activation_payload(
            hidden,
            compression_param=payload["compression_params_list"][2],
            compressor_name=payload["compressor_name"],
            feature_k_value=payload["feature_k_values"][2],
        )
        outgoing = dict(payload)
        outgoing["activation_comp"] = hidden_comp
        outgoing["tp_original_bytes"][2] = stats["original_bytes"]
        prepare_sec = time.perf_counter() - t_prepare
        outgoing["worker_stats_chain"] = dict(payload.get("worker_stats_chain", {}))
        outgoing["worker_stats_chain"]["B"] = _receiver_side_tp_stats(
            envelope,
            "B",
            profile=outgoing["worker_stats_chain"].get("B", {}).get("profile"),
            task_id=payload["task_id"],
        )
        outgoing["worker_stats_chain"]["C"] = {
            "task_id": payload["task_id"],
            "node_id": "C",
            "profile": _worker_profile("C", compute_sec, restore_sec, prepare_sec, 0, 0.0),
        }
        bytes_sent, send_sec = transport.send("D", MessageType.DECODER_STEP, outgoing)
        if bytes_sent <= 0 or send_sec <= 0.0 or not np.isfinite(send_sec):
            logging.warning(
                "[C] invalid tp_stats task_id=%s bytes_sent=%s send_sec=%s",
                payload["task_id"],
                bytes_sent,
                send_sec,
            )


def _run_worker_d(transport, partition, device):
    while True:
        envelope = transport.receive(msg_type=MessageType.DECODER_STEP, source=None)
        payload = envelope["payload"]
        if payload.get("cmd") == "shutdown":
            break
        t_restore = time.perf_counter()
        hidden = restore_activation_payload(payload["activation_comp"], device)
        restore_sec = time.perf_counter() - t_restore
        logits, compute_sec = _profile_from_partition(partition, hidden)
        predictions = torch.argmax(logits, dim=1).detach().cpu().tolist()
        t_prepare = time.perf_counter()
        result = {
            "task_id": payload["task_id"],
            "batch_idx": payload["batch_idx"],
            "requested_eta": payload["requested_eta"],
            "executed_eta": payload["feature_k_values"],
            "compressor_name": payload["compressor_name"],
            "compression_params_list": payload["compression_params_list"],
            "mapping_info": payload.get("mapping_info"),
            "predictions": predictions,
            "tp_original_bytes": list(payload["tp_original_bytes"]),
            "worker_stats_chain": dict(payload.get("worker_stats_chain", {})),
            "profile_d": None,
        }
        prepare_sec = time.perf_counter() - t_prepare
        result["worker_stats_chain"]["C"] = _receiver_side_tp_stats(
            envelope,
            "C",
            profile=result["worker_stats_chain"].get("C", {}).get("profile"),
            task_id=payload["task_id"],
        )
        result["profile_d"] = _worker_profile("D", compute_sec, restore_sec, prepare_sec, 0, 0.0)
        transport.send("A", MessageType.FINAL_RESULT, result)


def run_worker(
    node_id,
    device="cuda",
    checkpoint_path=MODEL_CHECKPOINT,
    transport_backend=TRANSPORT_BACKEND,
    tx_limit_mbps=DEFAULT_TX_LIMIT_MBPS,
    tx_bucket_capacity_bytes=DEFAULT_TX_BUCKET_CAPACITY_BYTES,
):
    if node_id == "A":
        raise ValueError("run_worker only supports B/C/D")
    factory = ResNet56PartitionFactory(checkpoint_path=checkpoint_path, device=device)
    partition = factory.build_partition_for_node(node_id)
    transport = create_transport(
        node_id,
        NODE_IPS,
        NODE_PORTS,
        backend=transport_backend,
        tx_limit_mbps=tx_limit_mbps,
        tx_bucket_capacity_bytes=tx_bucket_capacity_bytes,
    )
    transport.start()
    transport.wait_for_peers(["A"], timeout=TRANSPORT_READY_TIMEOUT_SEC)
    logging.info("[%s] Worker ready with partition %s", node_id, partition.partition_info["unit_names"])
    try:
        if node_id == "B":
            _run_worker_b(transport, partition, device)
        elif node_id == "C":
            _run_worker_c(transport, partition, device)
        else:
            _run_worker_d(transport, partition, device)
    finally:
        transport.close()


def _send_shutdown(transport):
    for peer_id, msg_type in [("B", MessageType.TASK_INPUT), ("C", MessageType.ENCODER_OUTPUT), ("D", MessageType.DECODER_STEP)]:
        try:
            transport.send(peer_id, msg_type, {"cmd": "shutdown"})
        except Exception:
            continue


def _accuracy_from_predictions(predictions, labels):
    pred_tensor = torch.as_tensor(predictions, dtype=torch.long)
    labels = labels.detach().cpu().to(torch.long)
    correct = int((pred_tensor == labels).sum().item())
    total = int(labels.numel())
    return correct, total, (float(correct) / float(total) if total > 0 else 0.0)


def _prepare_profile_context(
    transport,
    partition,
    batches,
    device,
    output_dir,
    profile_name,
    warmup_total,
    channel_model_type,
    channel_predictor_mode,
    channel_window_size,
):
    warmup_tp_stats = []
    warmup_profiles = []
    warmup_times = []
    a_samples = [[] for _ in range(NUM_TRANSFER_POINTS)]
    tau_samples = [[] for _ in range(NUM_PARTITIONS)]
    tau_list = [0.001] * NUM_PARTITIONS
    a_ref = [1.0] * NUM_TRANSFER_POINTS
    warmup_median_delay = 0.0

    if warmup_total > 0:
        logging.info("[A][%s] Warmup: %d batches with identity baseline", profile_name, warmup_total)
        pending = {}
        dispatched = 0
        completed = 0
        identity_plan = {
            "codec_name": "identity",
            "requested_eta": [1.0] * NUM_TRANSFER_POINTS,
            "execution_feature_k_values": [1.0] * NUM_TRANSFER_POINTS,
            "compression_params_list": [None] * NUM_TRANSFER_POINTS,
            "mapping_info": None,
        }

        while completed < warmup_total:
            while dispatched < warmup_total and len(pending) < ASYNC_MAX_INFLIGHT_TASKS:
                batch = batches[dispatched]
                task_id = "warmup_{}_{}_{}".format(profile_name, dispatched, uuid.uuid4().hex[:8])
                task_state = _dispatch_async_task(
                    transport=transport,
                    partition=partition,
                    batch=batch,
                    task_id=task_id,
                    execution_plan=identity_plan,
                    device=device,
                )
                pending[task_id] = task_state
                dispatched += 1

            envelope = _collect_result_payload(transport)
            if envelope is None:
                time.sleep(ASYNC_QUEUE_POLL_SEC)
                continue

            payload = envelope["payload"]
            task_id = payload["task_id"]
            if task_id not in pending:
                continue
            task_state = pending.pop(task_id)
            task_state["worker_stats"].update(payload.get("worker_stats_chain", {}))

            end_to_end_delay = time.perf_counter() - task_state["started_at"]
            stats_a = task_state["worker_stats"].get("A")
            stats_b = task_state["worker_stats"].get("B")
            stats_c = task_state["worker_stats"].get("C")
            profile_b = stats_b["profile"] if stats_b is not None else _worker_profile("B", 0.0, 0.0, 0.0, 0, 0.0)
            profile_c = stats_c["profile"] if stats_c is not None else _worker_profile("C", 0.0, 0.0, 0.0, 0, 0.0)
            tp0_stats = stats_a["tp_stats"] if stats_a is not None else task_state["tp_stats"].get("A", {"bytes": 0, "elapsed": 0.0})
            tp1_stats = stats_b["tp_stats"] if stats_b is not None else {"bytes": 0, "elapsed": 0.0}
            tp2_stats = stats_c["tp_stats"] if stats_c is not None else {"bytes": 0, "elapsed": 0.0}
            tp_stats = [
                (tp0_stats["bytes"], tp0_stats["elapsed"]),
                (tp1_stats["bytes"], tp1_stats["elapsed"]),
                (tp2_stats["bytes"], tp2_stats["elapsed"]),
            ]
            warmup_tp_stats.append(tp_stats)

            for idx, item in enumerate(payload.get("tp_original_bytes", [])):
                if float(item) > 0:
                    a_samples[idx].append(float(item))

            tau_values = [
                task_state["profiles"]["A"]["service_total_sec"],
                profile_b["service_total_sec"],
                profile_c["service_total_sec"],
                payload["profile_d"]["service_total_sec"],
            ]
            c_values = np.asarray(
                [
                    (float(item[0]) / float(item[1])) if float(item[1]) > 0.0 else 0.0
                    for item in tp_stats
                ],
                dtype=float,
            )
            a_values = np.asarray(payload.get("tp_original_bytes", [1.0] * NUM_TRANSFER_POINTS), dtype=float)
            total_delay = float(
                compute_pipeline_delay(
                    np.ones(NUM_TRANSFER_POINTS, dtype=float),
                    a_values,
                    np.asarray(tau_values, dtype=float),
                    c_values,
                )
            )
            for idx, tau_val in enumerate(tau_values):
                if tau_val > 0:
                    tau_samples[idx].append(float(tau_val))

            warmup_times.append(total_delay)
            warmup_profiles.append(
                {
                    "compressor_profile": profile_name,
                    "phase": "warmup",
                    "task_id": task_id,
                    "batch_idx": payload["batch_idx"],
                    "total_delay_sec": total_delay,
                    "end_to_end_delay_sec": float(end_to_end_delay),
                    "tp0_bytes": tp0_stats["bytes"],
                    "tp1_bytes": tp1_stats["bytes"],
                    "tp2_bytes": tp2_stats["bytes"],
                    "tp0_elapsed": tp0_stats["elapsed"],
                    "tp1_elapsed": tp1_stats["elapsed"],
                    "tp2_elapsed": tp2_stats["elapsed"],
                    **_flatten_dict(
                        {
                            "profiles": {
                                "A": task_state["profiles"]["A"],
                                "B": profile_b,
                                "C": profile_c,
                                "D": payload["profile_d"],
                            }
                        }
                    ),
                }
            )
            completed += 1

        tau_list = [float(np.median(tau_samples[idx])) if tau_samples[idx] else 0.001 for idx in range(NUM_PARTITIONS)]
        a_ref = [float(np.median(a_samples[idx])) if a_samples[idx] else 1.0 for idx in range(NUM_TRANSFER_POINTS)]
        warmup_median_delay = float(np.median(np.asarray(warmup_times))) if warmup_times else 0.0
        _save_rows(
            os.path.join(output_dir, "task_profiles_warmup_{}.csv".format(_sanitize_tag(profile_name))),
            warmup_profiles,
        )
        logging.info("[A][%s] Warmup done: median_delay=%.4fs a_ref=%s tau=%s", profile_name, warmup_median_delay, a_ref, tau_list)

    seeded_channel_estimator = _build_seeded_channel_estimator(
        warmup_tp_stats=warmup_tp_stats,
        device=device,
        model_type=channel_model_type,
        update_mode=channel_predictor_mode,
        window_size=channel_window_size,
    )
    return {
        "warmup_tp_stats": warmup_tp_stats,
        "warmup_median_delay": warmup_median_delay,
        "seeded_channel_estimator": seeded_channel_estimator,
        "tau_list": tau_list,
        "a_ref": a_ref,
    }


def _mean_array_from_records(records, key, length, fallback):
    if records:
        stacked = np.asarray([np.asarray(item[key], dtype=float) for item in records], dtype=float)
        if stacked.ndim == 2 and stacked.shape[1] == int(length):
            return np.mean(stacked, axis=0)
    return np.asarray(fallback, dtype=float).reshape(int(length))


def _solve_policy_snapshot(policy, a_t, tau_t, c_hat, deadline, lambda_value, window_id):
    start = time.perf_counter()
    if hasattr(policy, "epsilon"):
        policy.lambda_t = _sanitize_lambda_value(lambda_value, policy.epsilon)
    else:
        policy.lambda_t = float(lambda_value)
    selection = policy.select_eta(
        a_t=np.asarray(a_t, dtype=float),
        tau_t=np.asarray(tau_t, dtype=float),
        c_hat=np.asarray(c_hat, dtype=float),
        deadline=float(deadline),
    )
    return {
        "window_id": int(window_id),
        "lambda_value": float(policy.lambda_t),
        "selection": selection,
        "solver_elapsed_sec": time.perf_counter() - start,
        "c_hat": [float(item) for item in np.asarray(c_hat, dtype=float)],
        "a_t": [float(item) for item in np.asarray(a_t, dtype=float)],
        "tau_t": [float(item) for item in np.asarray(tau_t, dtype=float)],
    }


def _dynamic_timeslot_output_dir(base_dir, dynamic_timeslot_size):
    path = os.path.join(base_dir, "dynamic_timeslot_{}".format(int(dynamic_timeslot_size)))
    os.makedirs(path, exist_ok=True)
    return path


def _run_policy(
    transport,
    partition,
    batches,
    device,
    profile_name,
    profile_spec,
    t0_ratio,
    t0_deadline,
    seeded_channel_estimator,
    tau_list,
    a_ref,
    policy,
    dynamic_timeslot_size,
    max_dynamic_timeslot_count=None,
    max_total_batches=None,
):
    logging.info(
        "[A][%s][%s] Policy start t0=%.4fs dynamic_timeslot=%d",
        profile_name,
        policy.policy_name,
        t0_deadline,
        int(dynamic_timeslot_size),
    )
    channel_estimator = copy.deepcopy(seeded_channel_estimator)
    policy.initialize_from_channel_estimator(channel_estimator)
    pending = {}
    dispatched = 0
    completed = 0
    total_correct = 0
    total_seen = 0
    base_batch_count = len(batches)
    if base_batch_count <= 0:
        raise RuntimeError("No batches available for policy execution")
    max_window_count = (
        None if max_dynamic_timeslot_count is None else max(1, int(max_dynamic_timeslot_count))
    )
    max_total_batches = None if max_total_batches is None else max(1, int(max_total_batches))
    task_profiles = []
    accuracy_by_t = []
    delay_by_t = []
    lambda_by_t = []
    predicted_accuracy_by_t = []
    predicted_delay_by_t = []
    requested_eta_by_t = []
    executed_eta_by_t = []
    current_c_hat = np.asarray(channel_estimator.predict_all(), dtype=float)
    initial_solution = _solve_policy_snapshot(
        policy=policy,
        a_t=np.asarray(a_ref, dtype=float),
        tau_t=np.asarray(tau_list, dtype=float),
        c_hat=current_c_hat,
        deadline=t0_deadline,
        lambda_value=float(policy.lambda_t),
        window_id=0,
    )
    current_control_state = {
        "policy_version": 0,
        "window_id": 0,
        "requested_eta": [float(item) for item in np.asarray(initial_solution["selection"]["requested_eta"], dtype=float)],
        "predicted_accuracy": initial_solution["selection"]["predicted_accuracy"],
        "predicted_delay": initial_solution["selection"]["predicted_delay"],
        "solver_z": initial_solution["selection"]["solver_z"],
        "lambda_value": float(initial_solution["lambda_value"]),
        "solver_elapsed_sec": float(initial_solution["solver_elapsed_sec"]),
        "c_hat": list(initial_solution["c_hat"]),
        "window_avg_a": list(initial_solution["a_t"]),
        "window_avg_tau": list(initial_solution["tau_t"]),
        "window_size": 0,
    }
    active_window_records = []
    solver_job = None
    latest_window_id = 0
    next_policy_version = 1
    window_summaries = []
    window_summary_by_id = {}

    def _maybe_collect_solver_result():
        nonlocal solver_job, current_control_state, next_policy_version
        if solver_job is None or not solver_job.done():
            return
        solved = solver_job.result()
        selection = solved["selection"]
        current_control_state = {
            "policy_version": int(next_policy_version),
            "window_id": int(solved["window_id"]),
            "requested_eta": [float(item) for item in np.asarray(selection["requested_eta"], dtype=float)],
            "predicted_accuracy": selection["predicted_accuracy"],
            "predicted_delay": selection["predicted_delay"],
            "solver_z": selection["solver_z"],
            "lambda_value": float(solved["lambda_value"]),
            "solver_elapsed_sec": float(solved["solver_elapsed_sec"]),
            "c_hat": list(solved["c_hat"]),
            "window_avg_a": list(solved["a_t"]),
            "window_avg_tau": list(solved["tau_t"]),
            "window_size": 0,
        }
        summary_row = window_summary_by_id.get(int(solved["window_id"]))
        if summary_row is not None:
            current_control_state["window_size"] = int(summary_row.get("sample_count", 0))
            summary_row["solver_elapsed_sec"] = float(solved["solver_elapsed_sec"])
            summary_row["lambda_input"] = float(solved["lambda_value"])
            summary_row["predicted_accuracy"] = (
                np.nan if selection["predicted_accuracy"] is None else float(selection["predicted_accuracy"])
            )
            summary_row["predicted_delay_sec"] = (
                np.nan if selection["predicted_delay"] is None else float(selection["predicted_delay"])
            )
            summary_row["solver_z"] = float(selection["solver_z"])
            requested_eta = np.asarray(selection["requested_eta"], dtype=float)
            for idx, value in enumerate(requested_eta):
                summary_row["requested_eta_{}".format(idx)] = float(value)
        next_policy_version += 1
        solver_job = None

    def _try_start_solver(executor):
        nonlocal latest_window_id, active_window_records, solver_job
        if solver_job is not None:
            return
        if len(active_window_records) < max(1, int(dynamic_timeslot_size)):
            return
        if max_window_count is not None and latest_window_id >= max_window_count:
            return
        latest_window_id += 1
        snapshot_records = list(active_window_records)
        active_window_records = []
        avg_a = _mean_array_from_records(snapshot_records, "a_values", NUM_TRANSFER_POINTS, a_ref)
        avg_tau = _mean_array_from_records(snapshot_records, "tau_values", NUM_PARTITIONS, tau_list)
        avg_c = _mean_array_from_records(snapshot_records, "c_values", NUM_TRANSFER_POINTS, current_control_state["c_hat"])
        avg_delay = _finite_mean((float(item["delay"]) for item in snapshot_records), float(t0_deadline))
        total_window_correct = int(sum(int(item.get("correct_count", 0)) for item in snapshot_records))
        total_window_samples = int(sum(int(item.get("sample_total", 0)) for item in snapshot_records))
        avg_acc = (
            float(total_window_correct) / float(total_window_samples)
            if total_window_samples > 0 else np.nan
        )
        if hasattr(policy, "epsilon"):
            lambda_value = _sanitize_lambda_value(
                float(current_control_state["lambda_value"]) + avg_delay - float(t0_deadline),
                policy.epsilon,
            )
        else:
            lambda_value = float(current_control_state["lambda_value"])
        summary_row = {
            "window_id": int(latest_window_id),
            "algorithm": policy.policy_name,
            "policy": policy.policy_name,
            "policy_key": policy.policy_key,
            "dynamic_timeslot_size": int(dynamic_timeslot_size),
            "sample_count": int(len(snapshot_records)),
            "batch_count": int(len(snapshot_records)),
            "window_total_samples": int(total_window_samples),
            "window_correct_samples": int(total_window_correct),
            "start_t_index": int(snapshot_records[0]["t_index"]) if snapshot_records else np.nan,
            "end_t_index": int(snapshot_records[-1]["t_index"]) if snapshot_records else np.nan,
            "start_dataset_epoch": int(snapshot_records[0]["dataset_epoch"]) if snapshot_records else np.nan,
            "end_dataset_epoch": int(snapshot_records[-1]["dataset_epoch"]) if snapshot_records else np.nan,
            "start_dataset_index": int(snapshot_records[0]["dataset_index"]) if snapshot_records else np.nan,
            "end_dataset_index": int(snapshot_records[-1]["dataset_index"]) if snapshot_records else np.nan,
            "start_batch_idx": int(snapshot_records[0]["batch_idx"]) if snapshot_records else np.nan,
            "end_batch_idx": int(snapshot_records[-1]["batch_idx"]) if snapshot_records else np.nan,
            "avg_accuracy": float(avg_acc) if not np.isnan(avg_acc) else np.nan,
            "avg_delay_sec": float(avg_delay),
            "target_time_sec": float(t0_deadline),
            "avg_excess_delay_sec": float(avg_delay - float(t0_deadline)),
            "lambda_input": float(lambda_value),
            "solver_elapsed_sec": np.nan,
            "predicted_accuracy": np.nan,
            "predicted_delay_sec": np.nan,
            "solver_z": np.nan,
        }
        for idx, value in enumerate(np.asarray(avg_a, dtype=float)):
            summary_row["avg_a_{}".format(idx)] = float(value)
        for idx, value in enumerate(np.asarray(avg_tau, dtype=float)):
            summary_row["avg_tau_{}".format(idx)] = float(value)
        for idx, value in enumerate(np.asarray(avg_c, dtype=float)):
            summary_row["avg_c_{}".format(idx)] = float(value)
        window_summaries.append(summary_row)
        window_summary_by_id[int(latest_window_id)] = summary_row
        solver_job = executor.submit(
            _solve_policy_snapshot,
            policy,
            np.asarray(avg_a, dtype=float),
            np.asarray(avg_tau, dtype=float),
            np.asarray(avg_c, dtype=float),
            t0_deadline,
            float(lambda_value),
            int(latest_window_id),
        )

    def _maybe_advance_control(executor):
        previous_job = solver_job
        _maybe_collect_solver_result()
        if previous_job is None or solver_job is None:
            _try_start_solver(executor)

    def _dispatch_allowed():
        if max_total_batches is not None and dispatched >= int(max_total_batches):
            return False
        if max_window_count is None:
            return dispatched < int(base_batch_count)
        return int(latest_window_id) < int(max_window_count)

    with ThreadPoolExecutor(max_workers=1) as solver_executor:
        while True:
            _maybe_advance_control(solver_executor)
            while _dispatch_allowed() and len(pending) < ASYNC_MAX_INFLIGHT_TASKS:
                dataset_index = int(dispatched % base_batch_count)
                dataset_epoch = int(dispatched // base_batch_count)
                batch = batches[dataset_index]
                execution_plan = _build_execution_plan(profile_spec, current_control_state["requested_eta"])
                task_id = "{}_{}_{}_{}".format(profile_name, policy.policy_key, dispatched, uuid.uuid4().hex[:8])
                task_state = _dispatch_async_task(
                    transport=transport,
                    partition=partition,
                    batch=batch,
                    task_id=task_id,
                    execution_plan=execution_plan,
                    device=device,
                )
                task_state["t_index"] = dispatched
                task_state["policy_name"] = policy.policy_name
                task_state["policy_key"] = policy.policy_key
                task_state["selected_accuracy_est"] = current_control_state["predicted_accuracy"]
                task_state["selected_delay_est"] = current_control_state["predicted_delay"]
                task_state["solver_z"] = current_control_state["solver_z"]
                task_state["lambda_before"] = current_control_state["lambda_value"]
                task_state["c_hat"] = list(current_control_state["c_hat"])
                task_state["policy_version"] = int(current_control_state["policy_version"])
                task_state["window_id"] = int(current_control_state["window_id"])
                task_state["dataset_epoch"] = int(dataset_epoch)
                task_state["dataset_index"] = int(dataset_index)
                task_state["window_avg_a"] = list(current_control_state["window_avg_a"])
                task_state["window_avg_tau"] = list(current_control_state["window_avg_tau"])
                task_state["solver_elapsed_sec"] = float(current_control_state["solver_elapsed_sec"])
                pending[task_id] = task_state
                dispatched += 1

            envelope = _collect_result_payload(transport)
            if envelope is None:
                if (not _dispatch_allowed()) and not pending:
                    break
                time.sleep(ASYNC_QUEUE_POLL_SEC)
                continue

            payload = envelope["payload"]
            task_id = payload["task_id"]
            if task_id not in pending:
                continue
            task_state = pending.pop(task_id)
            task_state["worker_stats"].update(payload.get("worker_stats_chain", {}))

            t_index = int(task_state["t_index"])
            labels = task_state["labels"]
            predictions = payload["predictions"]
            correct, total, batch_acc = _accuracy_from_predictions(predictions, labels)
            total_correct += correct
            total_seen += total
            end_to_end_delay = time.perf_counter() - task_state["started_at"]

            stats_a = task_state["worker_stats"].get("A")
            stats_b = task_state["worker_stats"].get("B")
            stats_c = task_state["worker_stats"].get("C")
            profile_b = stats_b["profile"] if stats_b is not None else _worker_profile("B", 0.0, 0.0, 0.0, 0, 0.0)
            profile_c = stats_c["profile"] if stats_c is not None else _worker_profile("C", 0.0, 0.0, 0.0, 0, 0.0)
            tp0_stats = stats_a["tp_stats"] if stats_a is not None else task_state["tp_stats"].get("A", {"bytes": 0, "elapsed": 0.0})
            tp1_stats = stats_b["tp_stats"] if stats_b is not None else {"bytes": 0, "elapsed": 0.0}
            tp2_stats = stats_c["tp_stats"] if stats_c is not None else {"bytes": 0, "elapsed": 0.0}
            tp_stats = [
                (tp0_stats["bytes"], tp0_stats["elapsed"]),
                (tp1_stats["bytes"], tp1_stats["elapsed"]),
                (tp2_stats["bytes"], tp2_stats["elapsed"]),
            ]
            channel_estimator.observe_task(tp_stats)
            policy.observe_channel(tp_stats)
            c_values = np.asarray(
                [
                    (float(item[0]) / float(item[1])) if float(item[1]) > 0.0 else 0.0
                    for item in tp_stats
                ],
                dtype=float,
            )
            tau_values = np.asarray(
                [
                    task_state["profiles"]["A"]["service_total_sec"],
                    profile_b["service_total_sec"],
                    profile_c["service_total_sec"],
                    payload["profile_d"]["service_total_sec"],
                ],
                dtype=float,
            )
            actual_delay = float(
                compute_pipeline_delay(
                    np.asarray(task_state["executed_eta"], dtype=float),
                    np.asarray(payload.get("tp_original_bytes", [1.0] * NUM_TRANSFER_POINTS), dtype=float),
                    tau_values,
                    c_values,
                )
            )
            if (not np.all(np.isfinite(c_values))) or np.any(c_values <= 0.0) or (not np.isfinite(actual_delay)):
                logging.warning(
                    "[control] invalid delay inputs task_id=%s policy=%s t=%s batch=%s worker_nodes=%s tp_stats=%s c_values=%s a_values=%s exec_eta=%s actual_delay=%s",
                    task_id,
                    policy.policy_name,
                    t_index,
                    payload["batch_idx"],
                    sorted(task_state["worker_stats"].keys()),
                    tp_stats,
                    [float(x) if np.isfinite(x) else str(x) for x in c_values],
                    [float(x) for x in np.asarray(payload.get('tp_original_bytes', [1.0] * NUM_TRANSFER_POINTS), dtype=float)],
                    [float(x) for x in np.asarray(task_state["executed_eta"], dtype=float)],
                    actual_delay,
                )
            active_window_records.append(
                {
                    "t_index": int(t_index),
                    "dataset_epoch": int(task_state["dataset_epoch"]),
                    "dataset_index": int(task_state["dataset_index"]),
                    "batch_idx": int(payload["batch_idx"]),
                    "a_values": np.asarray(payload.get("tp_original_bytes", [1.0] * NUM_TRANSFER_POINTS), dtype=float),
                    "tau_values": tau_values,
                    "c_values": c_values,
                    "delay": float(actual_delay),
                    "accuracy": float(batch_acc),
                    "correct_count": int(correct),
                    "sample_total": int(total),
                }
            )
            _maybe_advance_control(solver_executor)

            accuracy_by_t.append(batch_acc)
            delay_by_t.append(actual_delay)
            lambda_by_t.append(float(task_state["lambda_before"]))
            predicted_accuracy_by_t.append(
                np.nan if task_state["selected_accuracy_est"] is None else float(task_state["selected_accuracy_est"])
            )
            predicted_delay_by_t.append(
                np.nan if task_state["selected_delay_est"] is None else float(task_state["selected_delay_est"])
            )
            requested_eta_by_t.append(list(task_state["requested_eta"]))
            executed_eta_by_t.append(list(task_state["executed_eta"]))

            profile_row = {
                "compressor_profile": profile_name,
                "compressor_display_name": profile_spec["display_name"],
                "t0_ratio": float(t0_ratio),
                "t0_deadline_sec": float(t0_deadline),
                "dynamic_timeslot_size": int(dynamic_timeslot_size),
                "policy": policy.policy_name,
                "task_id": task_id,
                "batch_idx": payload["batch_idx"],
                "dataset_epoch": int(task_state["dataset_epoch"]),
                "dataset_index": int(task_state["dataset_index"]),
                "t_index": t_index,
                "batch_accuracy": batch_acc,
                "correct_count": int(correct),
                "sample_total": int(total),
                "running_accuracy": float(total_correct) / float(total_seen) if total_seen > 0 else 0.0,
                "lambda_before": task_state["lambda_before"],
                "lambda_after": float(task_state["lambda_before"]),
                "requested_eta": ",".join(_format_float4(item) for item in task_state["requested_eta"]),
                "executed_eta": ",".join(_format_float4(item) for item in task_state["executed_eta"]),
                "predicted_accuracy": task_state["selected_accuracy_est"],
                "predicted_delay_sec": task_state["selected_delay_est"],
                "solver_z": task_state["solver_z"],
                "solver_elapsed_sec": float(task_state["solver_elapsed_sec"]),
                "policy_version": int(task_state["policy_version"]),
                "window_id": int(task_state["window_id"]),
                "compressor_name": task_state["compressor_name"],
                "mapping_distance_l2": float(task_state["mapping_info"]["match_distance_l2"]) if task_state.get("mapping_info") else 0.0,
                "total_delay_sec": actual_delay,
                "end_to_end_delay_sec": float(end_to_end_delay),
                "tp0_bytes": tp0_stats["bytes"],
                "tp1_bytes": tp1_stats["bytes"],
                "tp2_bytes": tp2_stats["bytes"],
                "tp0_elapsed": tp0_stats["elapsed"],
                "tp1_elapsed": tp1_stats["elapsed"],
                "tp2_elapsed": tp2_stats["elapsed"],
                "c_hat_0": float(task_state["c_hat"][0]),
                "c_hat_1": float(task_state["c_hat"][1]),
                "c_hat_2": float(task_state["c_hat"][2]),
                **_flatten_dict({"profiles": {"A": task_state["profiles"]["A"], "B": profile_b, "C": profile_c, "D": payload["profile_d"]}}),
            }
            task_profiles.append(profile_row)
            completed += 1

            logging.info(
                "[A][%s][%s][t0=%.2f][dyn=%d] t=%d batch=%s req_eta=%s exec_eta=%s c=%s c_hat=%s acc=%.4f delay=%.4fs service=%.4fs comm=%.4fs channel_mbps=%s lambda=%.4f window=%d policy_v=%d",
                profile_name,
                policy.policy_name,
                t0_ratio,
                int(dynamic_timeslot_size),
                t_index,
                payload["batch_idx"],
                task_state["requested_eta"],
                task_state["executed_eta"],
                [float(x) for x in c_values],
                [float(x) for x in task_state["c_hat"]],
                batch_acc,
                actual_delay,
                float(np.max(np.asarray(tau_values, dtype=float))) if len(tau_values) > 0 else 0.0,
                float(np.max(np.asarray([item[1] for item in tp_stats], dtype=float))) if tp_stats else 0.0,
                [round(float(x) / 1e6, 4) for x in np.asarray(c_values, dtype=float)],
                task_state["lambda_before"],
                int(task_state["window_id"]),
                int(task_state["policy_version"]),
            )

        while solver_job is not None or (
            max_window_count is None and len(active_window_records) >= max(1, int(dynamic_timeslot_size))
        ):
            _maybe_advance_control(solver_executor)
            if solver_job is None and (
                max_window_count is not None or len(active_window_records) < max(1, int(dynamic_timeslot_size))
            ):
                break
            time.sleep(ASYNC_QUEUE_POLL_SEC)

    return {
        "compressor_profile": profile_name,
        "compressor_display_name": profile_spec["display_name"],
        "accuracy_estimator_mode": getattr(policy.acc_model, "mode_name", None) if hasattr(policy, "acc_model") else None,
        "dynamic_timeslot_size": int(dynamic_timeslot_size),
        "dynamic_timeslot_mode": "tumbling_window_count",
        "max_dynamic_timeslot_count": (int(max_window_count) if max_window_count is not None else None),
        "policy_name": policy.policy_name,
        "policy_key": policy.policy_key,
        "mu_value": (float(policy.mu) if policy.mu is not None else None),
        "epsilon_value": (float(policy.epsilon) if hasattr(policy, "epsilon") else None),
        "accuracy_list": accuracy_by_t,
        "delay_list": delay_by_t,
        "lambda_list": lambda_by_t,
        "predicted_accuracy_history": predicted_accuracy_by_t,
        "predicted_delay_history": predicted_delay_by_t,
        "requested_eta_history": requested_eta_by_t,
        "executed_eta_history": executed_eta_by_t,
        "window_summaries": window_summaries,
        "task_profiles": list(task_profiles),
        "t0_deadline_sec": float(t0_deadline),
    }


def _safe_nanmean(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.isnan(arr).all():
        return np.nan
    return float(np.nanmean(arr))


def _summarize_policy_results(all_results):
    rows = []
    for result in all_results:
        acc_arr = np.asarray(result["accuracy_list"], dtype=float)
        delay_arr = np.asarray(result["delay_list"], dtype=float)
        pred_acc_arr = np.asarray(result.get("predicted_accuracy_history", []), dtype=float)
        pred_delay_arr = np.asarray(result.get("predicted_delay_history", []), dtype=float)
        deadline = float(result["t0_deadline_sec"])
        target_rate_hz = (1.0 / deadline) if deadline > 0.0 else np.nan
        excess_delay_arr = delay_arr - deadline
        rows.append(
            {
                "compressor_profile": result["compressor_profile"],
                "compressor_display_name": result["compressor_display_name"],
                "accuracy_estimator_mode": result.get("accuracy_estimator_mode"),
                "dynamic_timeslot_size": int(result.get("dynamic_timeslot_size", 0)),
                "target_rate_hz": float(target_rate_hz) if not np.isnan(target_rate_hz) else np.nan,
                "algorithm": result["policy_name"],
                "policy": result["policy_name"],
                "policy_key": result.get("policy_key"),
                "mu": (float(result["mu_value"]) if result.get("mu_value") is not None else np.nan),
                "epsilon": (float(result["epsilon_value"]) if result.get("epsilon_value") is not None else np.nan),
                "avg_utility": _safe_nanmean(acc_arr),
                "avg_acc": _safe_nanmean(acc_arr),
                "avg_delay": _safe_nanmean(delay_arr),
                "avg_excess_delay": _safe_nanmean(excess_delay_arr),
                "avg_delay_ratio": _safe_nanmean(delay_arr * target_rate_hz) if not np.isnan(target_rate_hz) else np.nan,
                "avg_accuracy_fit": _safe_nanmean(pred_acc_arr),
                "avg_accuracy_true": _safe_nanmean(acc_arr),
                "avg_delay_actual_sec": _safe_nanmean(delay_arr),
                "avg_delay_pred_sec": _safe_nanmean(pred_delay_arr),
                "avg_delay_ratio_actual": _safe_nanmean(delay_arr * target_rate_hz) if not np.isnan(target_rate_hz) else np.nan,
                "avg_delay_ratio_pred": _safe_nanmean(pred_delay_arr * target_rate_hz) if not np.isnan(target_rate_hz) else np.nan,
                "avg_accuracy": _safe_nanmean(acc_arr),
                "avg_delay_sec": _safe_nanmean(delay_arr),
                "violation_rate": float(np.mean(delay_arr > deadline)) if len(delay_arr) > 0 else 0.0,
                "final_lambda": float(result["lambda_list"][-1]) if result["lambda_list"] else 0.0,
                "t0_deadline_sec": deadline,
            }
        )
    return rows


def _summarize_last_timeslot(all_results):
    rows = []
    for result in all_results:
        window_summaries = list(result.get("window_summaries", []))
        if not window_summaries:
            continue
        last_window = dict(window_summaries[-1])
        task_profiles = list(result.get("task_profiles", []))
        acc_all_samples = np.nan
        if task_profiles:
            acc_all_samples = float(task_profiles[-1].get("running_accuracy", np.nan))
        deadline = float(result["t0_deadline_sec"])
        target_rate_hz = (1.0 / deadline) if deadline > 0.0 else np.nan
        rows.append(
            {
                "compressor_profile": result["compressor_profile"],
                "compressor_display_name": result["compressor_display_name"],
                "accuracy_estimator_mode": result.get("accuracy_estimator_mode"),
                "dynamic_timeslot_size": int(result.get("dynamic_timeslot_size", 0)),
                "target_rate_hz": float(target_rate_hz) if not np.isnan(target_rate_hz) else np.nan,
                "algorithm": result["policy_name"],
                "policy": result["policy_name"],
                "policy_key": result.get("policy_key"),
                "mu": (float(result["mu_value"]) if result.get("mu_value") is not None else np.nan),
                "epsilon": (float(result["epsilon_value"]) if result.get("epsilon_value") is not None else np.nan),
                "window_id": int(last_window.get("window_id", -1)),
                "sample_count_thisslot": int(last_window.get("sample_count", 0)),
                "batch_count_thisslot": int(last_window.get("batch_count", 0)),
                "acc_all_samples": float(acc_all_samples) if not np.isnan(acc_all_samples) else np.nan,
                "acc_thisslot": float(last_window.get("avg_accuracy", np.nan)),
                "avg_utility": float(last_window.get("avg_accuracy", np.nan)),
                "avg_acc": float(last_window.get("avg_accuracy", np.nan)),
                "avg_delay": float(last_window.get("avg_delay_sec", np.nan)),
                "avg_excess_delay": float(last_window.get("avg_excess_delay_sec", np.nan)),
                "avg_delay_ratio": (
                    float(last_window.get("avg_delay_sec", np.nan)) * float(target_rate_hz)
                    if not np.isnan(target_rate_hz) and not np.isnan(float(last_window.get("avg_delay_sec", np.nan)))
                    else np.nan
                ),
                "lambda_input": float(last_window.get("lambda_input", np.nan)),
                "solver_elapsed_sec": float(last_window.get("solver_elapsed_sec", np.nan)),
                "predicted_accuracy": float(last_window.get("predicted_accuracy", np.nan)),
                "predicted_delay_sec": float(last_window.get("predicted_delay_sec", np.nan)),
                "solver_z": float(last_window.get("solver_z", np.nan)),
                "t0_deadline_sec": deadline,
            }
        )
    return rows


def _aggregate_accuracy_summary(summary_rows):
    if not summary_rows:
        return {
            "policy_count": 0,
            "avg_accuracy_over_policies": np.nan,
            "best_policy_by_accuracy": None,
            "best_avg_accuracy": np.nan,
        }
    frame = pd.DataFrame.from_records(summary_rows)
    summary = {
        "policy_count": int(len(frame)),
        "avg_accuracy_over_policies": float(frame["avg_accuracy"].mean()) if "avg_accuracy" in frame else np.nan,
        "best_policy_by_accuracy": None,
        "best_avg_accuracy": np.nan,
    }
    if "avg_accuracy" in frame and not frame["avg_accuracy"].isna().all():
        best_idx = frame["avg_accuracy"].idxmax()
        summary["best_policy_by_accuracy"] = str(frame.loc[best_idx, "policy"])
        summary["best_avg_accuracy"] = float(frame.loc[best_idx, "avg_accuracy"])
    return summary


def _write_plot_context(
    output_dir,
    profile_name,
    profile_spec,
    target_rate_hz,
    t0_deadline,
    batch_size,
    dynamic_timeslot_size,
    max_dynamic_timeslot_count=None,
    accuracy_estimator_mode=None,
    stein_sigma=None,
    stein_N=None,
):
    plot_context = {
        "experiment": "online",
        "platform": "jetson",
        "model": MODEL_TAG,
        "dataset": "cifar10",
        "codec_name": profile_spec["codec_name"],
        "codec_profile": profile_name,
        "accuracy_estimator_mode": accuracy_estimator_mode,
        "target_rate_hz": float(target_rate_hz),
        "target_time_sec": float(t0_deadline),
        "batch_size": int(batch_size),
        "dynamic_timeslot_size": int(dynamic_timeslot_size),
        "dynamic_timeslot_mode": "tumbling_window_count",
        "max_dynamic_timeslot_count": (
            int(max_dynamic_timeslot_count) if max_dynamic_timeslot_count is not None else None
        ),
    }
    if accuracy_estimator_mode == "stein_estimator":
        plot_context["stein_sigma"] = float(stein_sigma) if stein_sigma is not None else None
        plot_context["stein_N"] = int(stein_N) if stein_N is not None else None
    with open(os.path.join(output_dir, "plot_context.json"), "w", encoding="utf-8") as handle:
        json.dump(plot_context, handle, ensure_ascii=False, indent=2)
    return plot_context


def run_node_a(
    device="cuda",
    checkpoint_path=MODEL_CHECKPOINT,
    batch_size=DEFAULT_BATCH_SIZE,
    max_batches=DEFAULT_MAX_BATCHES,
    data_root=DEFAULT_DATA_ROOT,
    download_data=DEFAULT_DOWNLOAD_DATA,
    channel_predictor_mode=DEFAULT_CHANNEL_UPDATE_MODE,
    channel_model_type=DEFAULT_CHANNEL_MODEL_TYPE,
    channel_window_size=DEFAULT_CHANNEL_WINDOW_SIZE,
    warmup_steps=DEFAULT_WARMUP_STEPS,
    t0_ratio=DEFAULT_T0_RATIO,
    mu_values_arg=None,
    dynamic_timeslot_sizes_arg=None,
    accuracy_estimator_modes_arg=DEFAULT_ACCURACY_ESTIMATOR_MODES,
    epsilon=DEFAULT_EPSILON,
    stein_sigma=DEFAULT_STEIN_SIGMA,
    stein_N=DEFAULT_STEIN_N,
    stein_fast_max_batches=DEFAULT_STEIN_FAST_MAX_BATCHES,
    transport_backend=TRANSPORT_BACKEND,
    tx_limit_mbps=DEFAULT_TX_LIMIT_MBPS,
    tx_bucket_capacity_bytes=DEFAULT_TX_BUCKET_CAPACITY_BYTES,
    compressor_profiles_arg="topk,quantization,llmint8_fp16_int4",
):
    output_dir = _make_run_output_dir("A", "experiment", device=device, tx_limit_mbps=tx_limit_mbps)
    logging.info("[A] Output directory: %s", output_dir)
    factory = ResNet56PartitionFactory(checkpoint_path=checkpoint_path, device=device)
    partition = factory.build_partition_for_node("A")
    batches = load_cifar10_batches(batch_size=batch_size, max_batches=max_batches, data_root=data_root, download=download_data)
    if not batches:
        raise RuntimeError("No CIFAR-10 batches were loaded; aborting experiment")

    requested_profiles = [
        _normalize_profile_name(item) for item in str(compressor_profiles_arg).split(",") if item.strip()
    ]
    profile_specs = _codec_profiles()
    for item in requested_profiles:
        if item not in profile_specs:
            raise ValueError("Unknown compressor profile '{}'. Choose from {}".format(item, list(profile_specs.keys())))

    warmup_total = min(max(0, int(warmup_steps)), len(batches))
    stein_fast_batches = batches[: max(1, int(stein_fast_max_batches))]
    mu_values = _parse_mu_values(mu_values_arg)
    dynamic_timeslot_sizes = _parse_dynamic_timeslot_sizes(dynamic_timeslot_sizes_arg)
    accuracy_estimator_modes = _parse_accuracy_estimator_modes(accuracy_estimator_modes_arg)
    t0_ratio_values = [float(t0_ratio)] if t0_ratio is not None else list(T0_RATIOS)

    transport = create_transport(
        "A",
        NODE_IPS,
        NODE_PORTS,
        backend=transport_backend,
        tx_limit_mbps=tx_limit_mbps,
        tx_bucket_capacity_bytes=tx_bucket_capacity_bytes,
    )
    transport.start()
    transport.wait_for_peers(["B", "C", "D"], timeout=TRANSPORT_READY_TIMEOUT_SEC)
    logging.info("[A] All workers are ready")

    run_start = time.perf_counter()
    run_summary = {
        "model_tag": MODEL_TAG,
        "checkpoint_path": checkpoint_path,
        "batch_size": batch_size,
        "experiment_batches": len(batches),
        "warmup_batches": warmup_total,
        "mu_values": list(mu_values),
        "dynamic_timeslot_sizes": list(dynamic_timeslot_sizes),
        "epsilon": float(epsilon),
        "accuracy_estimator_modes": list(accuracy_estimator_modes),
        "stein_sigma": float(stein_sigma),
        "stein_N": int(stein_N),
        "stein_fast_max_batches": int(len(stein_fast_batches)),
        "t0_ratios": list(t0_ratio_values),
        "partition_plan": factory.partition_plan,
        "profiles": [],
    }

    try:
        for profile_name in requested_profiles:
            profile_spec = profile_specs[profile_name]
            profile_dir = os.path.join(output_dir, _sanitize_tag(profile_name))
            os.makedirs(profile_dir, exist_ok=True)
            profile_ctx = _prepare_profile_context(
                transport=transport,
                partition=partition,
                batches=batches,
                device=device,
                output_dir=profile_dir,
                profile_name=profile_name,
                warmup_total=warmup_total,
                channel_model_type=channel_model_type,
                channel_predictor_mode=channel_predictor_mode,
                channel_window_size=channel_window_size,
            )

            profile_summary = {
                "compressor_profile": profile_name,
                "compressor_display_name": profile_spec["display_name"],
                "eta_min": [float(item) for item in profile_spec["eta_min"]],
                "warmup_median_delay_sec": float(profile_ctx["warmup_median_delay"]),
                "a_ref": [float(item) for item in profile_ctx["a_ref"]],
                "tau_list": [float(item) for item in profile_ctx["tau_list"]],
                "accuracy_estimator_runs": [],
            }

            for accuracy_estimator_mode in accuracy_estimator_modes:
                estimator_dir = os.path.join(profile_dir, _sanitize_tag(accuracy_estimator_mode))
                os.makedirs(estimator_dir, exist_ok=True)
                estimator = _build_accuracy_estimator(
                    profile_spec=profile_spec,
                    estimator_mode=accuracy_estimator_mode,
                    checkpoint_path=checkpoint_path,
                    device=device,
                    batches=stein_fast_batches,
                    stein_sigma=stein_sigma,
                    stein_N=stein_N,
                )
                if estimator is None:
                    logging.warning(
                        "[A][%s][%s] Estimator unavailable; no_csi_single will be skipped",
                        profile_name,
                        accuracy_estimator_mode,
                    )

                estimator_summary = {
                    "accuracy_estimator_mode": accuracy_estimator_mode,
                    "estimator_path": profile_spec["estimator_path"] if accuracy_estimator_mode == "fitting_model" else None,
                    "t0_runs": [],
                }

                for current_t0_ratio in t0_ratio_values:
                    t0_deadline = float(profile_ctx["warmup_median_delay"]) * float(current_t0_ratio) if profile_ctx["warmup_median_delay"] > 0 else 0.0
                    t0_output_dir = _t0_output_dir(estimator_dir, current_t0_ratio)
                    target_rate_hz = (1.0 / float(t0_deadline)) if float(t0_deadline) > 0.0 else 0.0
                    policies = []
                    if estimator is not None:
                        for mu_value in mu_values:
                            policies.append(NoCSISinglePolicy(profile_spec["eta_min"], estimator, mu=mu_value, epsilon=epsilon))
                    policies.extend(
                        [
                            MyopicSinglePolicy(profile_spec["eta_min"], estimator),
                            ConservativeSinglePolicy(profile_spec["eta_min"], estimator),
                            MovingAverageSinglePolicy(profile_spec["eta_min"], estimator, window_size=5),
                            NoCompressionBaselinePolicy(profile_spec["eta_min"]),
                            MaxCompressionBaselinePolicy(profile_spec["eta_min"]),
                        ]
                    )

                    policy_results = []
                    for policy in policies:
                        result = _run_policy(
                            transport=transport,
                            partition=partition,
                            batches=batches,
                            device=device,
                            profile_name=profile_name,
                            profile_spec=profile_spec,
                            t0_ratio=current_t0_ratio,
                            t0_deadline=t0_deadline,
                            seeded_channel_estimator=profile_ctx["seeded_channel_estimator"],
                            tau_list=profile_ctx["tau_list"],
                            a_ref=profile_ctx["a_ref"],
                            policy=policy,
                        )
                        result["accuracy_estimator_mode"] = accuracy_estimator_mode
                        policy_results.append(result)
                        _save_rows(os.path.join(t0_output_dir, "task_profiles_{}_{}.csv".format(_sanitize_tag(profile_name), _sanitize_tag(policy.policy_name))), result["task_profiles"])
                        timeseries_rows = []
                        for idx in range(len(result["accuracy_list"])):
                            plot_accuracy = float(
                                result["task_profiles"][idx].get("running_accuracy", result["accuracy_list"][idx])
                            )
                            requested_eta = [float(x) for x in result["requested_eta_history"][idx]]
                            executed_eta = [float(x) for x in result["executed_eta_history"][idx]]
                            pred_acc = result["predicted_accuracy_history"][idx]
                            pred_delay = result["predicted_delay_history"][idx]
                            timeseries_rows.append(
                                {
                                    "t": idx,
                                    "algorithm": result["policy_name"],
                                    "policy": result["policy_name"],
                                    "policy_key": result.get("policy_key"),
                                    "codec_name": profile_spec["codec_name"],
                                    "accuracy_estimator_mode": accuracy_estimator_mode,
                                    "target_rate_hz": float(target_rate_hz),
                                    "mu": (float(result["mu_value"]) if result.get("mu_value") is not None else np.nan),
                                    "epsilon": (float(epsilon) if str(result.get("policy_key")) == "no_csi_single" else np.nan),
                                    "accuracy_true": plot_accuracy,
                                    "accuracy_fit": pred_acc,
                                    "delay_actual_sec": result["delay_list"][idx],
                                    "delay_pred_sec": pred_delay,
                                    "delay_ratio_actual": (
                                        float(result["delay_list"][idx]) * float(target_rate_hz)
                                        if float(target_rate_hz) > 0.0 else np.nan
                                    ),
                                    "delay_ratio_pred": (
                                        float(pred_delay) * float(target_rate_hz)
                                        if float(target_rate_hz) > 0.0 and not pd.isna(pred_delay) else np.nan
                                    ),
                                    "lambda_value": result["lambda_list"][idx],
                                    "accuracy": plot_accuracy,
                                    "delay_sec": result["delay_list"][idx],
                                    "lambda": result["lambda_list"][idx],
                                    "requested_eta": ",".join(_format_float4(x) for x in requested_eta),
                                    "executed_eta": ",".join(_format_float4(x) for x in executed_eta),
                                    "eta_0": requested_eta[0],
                                    "eta_1": requested_eta[1],
                                    "eta_2": requested_eta[2],
                                    "executed_eta_0": executed_eta[0],
                                    "executed_eta_1": executed_eta[1],
                                    "executed_eta_2": executed_eta[2],
                                }
                            )
                        _save_rows(os.path.join(t0_output_dir, "timeseries_{}_{}.csv".format(_sanitize_tag(profile_name), _sanitize_tag(policy.policy_name))), timeseries_rows)

                    summary_rows = _summarize_policy_results(policy_results)
                    _save_rows(os.path.join(t0_output_dir, "policy_summary_{}.csv".format(_sanitize_tag(profile_name))), summary_rows)
                    plot_context = _write_plot_context(
                        output_dir=t0_output_dir,
                        profile_name=profile_name,
                        profile_spec=profile_spec,
                        target_rate_hz=target_rate_hz,
                        t0_deadline=t0_deadline,
                        batch_size=batch_size,
                        accuracy_estimator_mode=accuracy_estimator_mode,
                        stein_sigma=stein_sigma,
                        stein_N=stein_N,
                    )
                    try:
                        render_trial_plots(t0_output_dir, plot_context=plot_context)
                    except Exception as exc:
                        logging.exception(
                            "[A][%s][%s][t0=%.2f] Plot rendering failed: %s",
                            profile_name,
                            accuracy_estimator_mode,
                            current_t0_ratio,
                            exc,
                        )
                    estimator_summary["t0_runs"].append(
                        {
                            "t0_ratio": float(current_t0_ratio),
                            "t0_deadline_sec": float(t0_deadline),
                            "output_dir": t0_output_dir,
                            "accuracy_summary": _aggregate_accuracy_summary(summary_rows),
                            "policy_summary": summary_rows,
                        }
                    )

                estimator_summary["accuracy_summary"] = _aggregate_accuracy_summary(
                    [row for t0_run in estimator_summary["t0_runs"] for row in t0_run["policy_summary"]]
                )
                profile_summary["accuracy_estimator_runs"].append(estimator_summary)

            profile_summary["accuracy_summary"] = _aggregate_accuracy_summary(
                [
                    row
                    for estimator_run in profile_summary["accuracy_estimator_runs"]
                    for t0_run in estimator_run["t0_runs"]
                    for row in t0_run["policy_summary"]
                ]
            )
            with open(os.path.join(profile_dir, "profile_summary.json"), "w", encoding="utf-8") as handle:
                json.dump(profile_summary, handle, ensure_ascii=False, indent=2)
            run_summary["profiles"].append(profile_summary)
    finally:
        _send_shutdown(transport)
        transport.close()

    run_summary["run_wall_sec"] = time.perf_counter() - run_start
    run_summary["accuracy_summary"] = _aggregate_accuracy_summary(
        [
            row
            for profile in run_summary["profiles"]
            for estimator_run in profile.get("accuracy_estimator_runs", [])
            for t0_run in estimator_run.get("t0_runs", [])
            for row in t0_run.get("policy_summary", [])
        ]
    )
    with open(os.path.join(output_dir, "run_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, ensure_ascii=False, indent=2)
    logging.info("[A] Completed experiment")
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


def _run_node_a_dynamic(
    device="cuda",
    checkpoint_path=MODEL_CHECKPOINT,
    batch_size=DEFAULT_BATCH_SIZE,
    max_batches=DEFAULT_MAX_BATCHES,
    data_root=DEFAULT_DATA_ROOT,
    download_data=DEFAULT_DOWNLOAD_DATA,
    channel_predictor_mode=DEFAULT_CHANNEL_UPDATE_MODE,
    channel_model_type=DEFAULT_CHANNEL_MODEL_TYPE,
    channel_window_size=DEFAULT_CHANNEL_WINDOW_SIZE,
    warmup_steps=DEFAULT_WARMUP_STEPS,
    t0_ratio=DEFAULT_T0_RATIO,
    mu_values_arg=None,
    dynamic_timeslot_sizes_arg=None,
    max_dynamic_timeslot_count=DEFAULT_MAX_DYNAMIC_TIMESLOT_COUNT,
    max_total_batches=None,
    accuracy_estimator_modes_arg=DEFAULT_ACCURACY_ESTIMATOR_MODES,
    epsilon=DEFAULT_EPSILON,
    stein_sigma=DEFAULT_STEIN_SIGMA,
    stein_N=DEFAULT_STEIN_N,
    stein_fast_max_batches=DEFAULT_STEIN_FAST_MAX_BATCHES,
    transport_backend=TRANSPORT_BACKEND,
    tx_limit_mbps=DEFAULT_TX_LIMIT_MBPS,
    tx_bucket_capacity_bytes=DEFAULT_TX_BUCKET_CAPACITY_BYTES,
    compressor_profiles_arg="topk,quantization,llmint8_fp16_int4",
):
    output_dir = _make_run_output_dir("A", "experiment", device=device, tx_limit_mbps=tx_limit_mbps)
    logging.info("[A] Output directory: %s", output_dir)
    factory = ResNet56PartitionFactory(checkpoint_path=checkpoint_path, device=device)
    partition = factory.build_partition_for_node("A")
    batches = load_cifar10_batches(batch_size=batch_size, max_batches=max_batches, data_root=data_root, download=download_data)
    if not batches:
        raise RuntimeError("No CIFAR-10 batches were loaded; aborting experiment")

    requested_profiles = [
        _normalize_profile_name(item) for item in str(compressor_profiles_arg).split(",") if item.strip()
    ]
    profile_specs = _codec_profiles()
    for item in requested_profiles:
        if item not in profile_specs:
            raise ValueError("Unknown compressor profile '{}'. Choose from {}".format(item, list(profile_specs.keys())))

    warmup_total = min(max(0, int(warmup_steps)), len(batches))
    stein_fast_batches = batches[: max(1, int(stein_fast_max_batches))]
    mu_values = _parse_mu_values(mu_values_arg)
    dynamic_timeslot_sizes = _parse_dynamic_timeslot_sizes(dynamic_timeslot_sizes_arg)
    accuracy_estimator_modes = _parse_accuracy_estimator_modes(accuracy_estimator_modes_arg)
    t0_ratio_values = [float(t0_ratio)] if t0_ratio is not None else list(T0_RATIOS)

    transport = create_transport(
        "A",
        NODE_IPS,
        NODE_PORTS,
        backend=transport_backend,
        tx_limit_mbps=tx_limit_mbps,
        tx_bucket_capacity_bytes=tx_bucket_capacity_bytes,
    )
    transport.start()
    transport.wait_for_peers(["B", "C", "D"], timeout=TRANSPORT_READY_TIMEOUT_SEC)
    logging.info("[A] All workers are ready")

    run_start = time.perf_counter()
    run_summary = {
        "model_tag": MODEL_TAG,
        "checkpoint_path": checkpoint_path,
        "batch_size": batch_size,
        "experiment_batches": len(batches),
        "warmup_batches": warmup_total,
        "mu_values": list(mu_values),
        "dynamic_timeslot_sizes": list(dynamic_timeslot_sizes),
        "max_dynamic_timeslot_count": (
            None if max_dynamic_timeslot_count is None else int(max_dynamic_timeslot_count)
        ),
        "max_total_batches": (
            None if max_total_batches is None else int(max_total_batches)
        ),
        "epsilon": float(epsilon),
        "accuracy_estimator_modes": list(accuracy_estimator_modes),
        "stein_sigma": float(stein_sigma),
        "stein_N": int(stein_N),
        "stein_fast_max_batches": int(len(stein_fast_batches)),
        "t0_ratios": list(t0_ratio_values),
        "partition_plan": factory.partition_plan,
        "profiles": [],
    }

    try:
        for profile_name in requested_profiles:
            profile_spec = profile_specs[profile_name]
            profile_dir = os.path.join(output_dir, _sanitize_tag(profile_name))
            os.makedirs(profile_dir, exist_ok=True)
            profile_ctx = _prepare_profile_context(
                transport=transport,
                partition=partition,
                batches=batches,
                device=device,
                output_dir=profile_dir,
                profile_name=profile_name,
                warmup_total=warmup_total,
                channel_model_type=channel_model_type,
                channel_predictor_mode=channel_predictor_mode,
                channel_window_size=channel_window_size,
            )

            profile_summary = {
                "compressor_profile": profile_name,
                "compressor_display_name": profile_spec["display_name"],
                "eta_min": [float(item) for item in profile_spec["eta_min"]],
                "warmup_median_delay_sec": float(profile_ctx["warmup_median_delay"]),
                "a_ref": [float(item) for item in profile_ctx["a_ref"]],
                "tau_list": [float(item) for item in profile_ctx["tau_list"]],
                "accuracy_estimator_runs": [],
            }

            for accuracy_estimator_mode in accuracy_estimator_modes:
                estimator_dir = os.path.join(profile_dir, _sanitize_tag(accuracy_estimator_mode))
                os.makedirs(estimator_dir, exist_ok=True)
                estimator = _build_accuracy_estimator(
                    profile_spec=profile_spec,
                    estimator_mode=accuracy_estimator_mode,
                    checkpoint_path=checkpoint_path,
                    device=device,
                    batches=stein_fast_batches,
                    stein_sigma=stein_sigma,
                    stein_N=stein_N,
                )
                if estimator is None:
                    logging.warning(
                        "[A][%s][%s] Estimator unavailable; no_csi_single will be skipped",
                        profile_name,
                        accuracy_estimator_mode,
                    )

                estimator_summary = {
                    "accuracy_estimator_mode": accuracy_estimator_mode,
                    "estimator_path": profile_spec["estimator_path"] if accuracy_estimator_mode == "fitting_model" else None,
                    "t0_runs": [],
                }

                for current_t0_ratio in t0_ratio_values:
                    t0_deadline = float(profile_ctx["warmup_median_delay"]) * float(current_t0_ratio) if profile_ctx["warmup_median_delay"] > 0 else 0.0
                    target_rate_hz = (1.0 / float(t0_deadline)) if float(t0_deadline) > 0.0 else 0.0
                    t0_output_dir = _t0_output_dir(estimator_dir, current_t0_ratio)

                    for dynamic_timeslot_size in dynamic_timeslot_sizes:
                        dynamic_output_dir = _dynamic_timeslot_output_dir(t0_output_dir, dynamic_timeslot_size)
                        policies = []
                        if estimator is not None:
                            for mu_value in mu_values:
                                policies.append(NoCSISinglePolicy(profile_spec["eta_min"], estimator, mu=mu_value, epsilon=epsilon))
                        policies.extend(
                            [
                                MyopicSinglePolicy(profile_spec["eta_min"], estimator),
                                ConservativeSinglePolicy(profile_spec["eta_min"], estimator),
                                MovingAverageSinglePolicy(profile_spec["eta_min"], estimator, window_size=5),
                                NoCompressionBaselinePolicy(profile_spec["eta_min"]),
                                MaxCompressionBaselinePolicy(profile_spec["eta_min"]),
                            ]
                        )

                        policy_results = []
                        for policy in policies:
                            result = _run_policy(
                                transport=transport,
                                partition=partition,
                                batches=batches,
                                device=device,
                                profile_name=profile_name,
                                profile_spec=profile_spec,
                                t0_ratio=current_t0_ratio,
                                t0_deadline=t0_deadline,
                                seeded_channel_estimator=profile_ctx["seeded_channel_estimator"],
                                tau_list=profile_ctx["tau_list"],
                                a_ref=profile_ctx["a_ref"],
                                policy=policy,
                                dynamic_timeslot_size=dynamic_timeslot_size,
                                max_dynamic_timeslot_count=max_dynamic_timeslot_count,
                                max_total_batches=max_total_batches,
                            )
                            result["accuracy_estimator_mode"] = accuracy_estimator_mode
                            policy_results.append(result)
                            _save_rows(
                                os.path.join(dynamic_output_dir, "task_profiles_{}_{}.csv".format(_sanitize_tag(profile_name), _sanitize_tag(policy.policy_name))),
                                result["task_profiles"],
                            )
                            _save_rows(
                                os.path.join(dynamic_output_dir, "window_summary_{}_{}.csv".format(_sanitize_tag(profile_name), _sanitize_tag(policy.policy_name))),
                                result.get("window_summaries", []),
                            )
                            timeseries_rows = []
                            for idx in range(len(result["accuracy_list"])):
                                plot_accuracy = float(
                                    result["task_profiles"][idx].get("running_accuracy", result["accuracy_list"][idx])
                                )
                                requested_eta = [float(x) for x in result["requested_eta_history"][idx]]
                                executed_eta = [float(x) for x in result["executed_eta_history"][idx]]
                                pred_acc = result["predicted_accuracy_history"][idx]
                                pred_delay = result["predicted_delay_history"][idx]
                                timeseries_rows.append(
                                    {
                                        "t": idx,
                                        "algorithm": result["policy_name"],
                                        "policy": result["policy_name"],
                                        "policy_key": result.get("policy_key"),
                                        "codec_name": profile_spec["codec_name"],
                                        "accuracy_estimator_mode": accuracy_estimator_mode,
                                        "dynamic_timeslot_size": int(dynamic_timeslot_size),
                                        "target_rate_hz": float(target_rate_hz),
                                        "mu": (float(result["mu_value"]) if result.get("mu_value") is not None else np.nan),
                                        "epsilon": (float(epsilon) if str(result.get("policy_key")) == "no_csi_single" else np.nan),
                                        "accuracy_true": plot_accuracy,
                                        "accuracy_fit": pred_acc,
                                        "delay_actual_sec": result["delay_list"][idx],
                                        "delay_pred_sec": pred_delay,
                                        "delay_ratio_actual": (
                                            float(result["delay_list"][idx]) * float(target_rate_hz)
                                            if float(target_rate_hz) > 0.0 else np.nan
                                        ),
                                        "delay_ratio_pred": (
                                            float(pred_delay) * float(target_rate_hz)
                                            if float(target_rate_hz) > 0.0 and not pd.isna(pred_delay) else np.nan
                                        ),
                                        "lambda_value": result["lambda_list"][idx],
                                        "accuracy": plot_accuracy,
                                        "delay_sec": result["delay_list"][idx],
                                        "lambda": result["lambda_list"][idx],
                                        "requested_eta": ",".join(_format_float4(x) for x in requested_eta),
                                        "executed_eta": ",".join(_format_float4(x) for x in executed_eta),
                                        "eta_0": requested_eta[0],
                                        "eta_1": requested_eta[1],
                                        "eta_2": requested_eta[2],
                                        "executed_eta_0": executed_eta[0],
                                        "executed_eta_1": executed_eta[1],
                                        "executed_eta_2": executed_eta[2],
                                    }
                                )
                            _save_rows(
                                os.path.join(dynamic_output_dir, "timeseries_{}_{}.csv".format(_sanitize_tag(profile_name), _sanitize_tag(policy.policy_name))),
                                timeseries_rows,
                            )

                        summary_rows = _summarize_policy_results(policy_results)
                        lastslot_summary_rows = _summarize_last_timeslot(policy_results)
                        _save_rows(os.path.join(dynamic_output_dir, "policy_summary_{}.csv".format(_sanitize_tag(profile_name))), summary_rows)
                        _save_rows(os.path.join(dynamic_output_dir, "summary_lasttimeslot_{}.csv".format(_sanitize_tag(profile_name))), lastslot_summary_rows)
                        plot_context = _write_plot_context(
                            output_dir=dynamic_output_dir,
                            profile_name=profile_name,
                            profile_spec=profile_spec,
                            target_rate_hz=target_rate_hz,
                            t0_deadline=t0_deadline,
                            batch_size=batch_size,
                            dynamic_timeslot_size=dynamic_timeslot_size,
                            max_dynamic_timeslot_count=max_dynamic_timeslot_count,
                            accuracy_estimator_mode=accuracy_estimator_mode,
                            stein_sigma=stein_sigma,
                            stein_N=stein_N,
                        )
                        try:
                            render_trial_plots(dynamic_output_dir, plot_context=plot_context)
                        except Exception as exc:
                            logging.exception(
                                "[A][%s][%s][t0=%.2f][dyn=%d] Plot rendering failed: %s",
                                profile_name,
                                accuracy_estimator_mode,
                                current_t0_ratio,
                                int(dynamic_timeslot_size),
                                exc,
                            )
                        estimator_summary["t0_runs"].append(
                            {
                                "t0_ratio": float(current_t0_ratio),
                                "t0_deadline_sec": float(t0_deadline),
                                "dynamic_timeslot_size": int(dynamic_timeslot_size),
                                "output_dir": dynamic_output_dir,
                                "accuracy_summary": _aggregate_accuracy_summary(summary_rows),
                                "policy_summary": summary_rows,
                                "summary_lasttimeslot": lastslot_summary_rows,
                            }
                        )

                estimator_summary["accuracy_summary"] = _aggregate_accuracy_summary(
                    [row for t0_run in estimator_summary["t0_runs"] for row in t0_run["policy_summary"]]
                )
                profile_summary["accuracy_estimator_runs"].append(estimator_summary)

            profile_summary["accuracy_summary"] = _aggregate_accuracy_summary(
                [
                    row
                    for estimator_run in profile_summary["accuracy_estimator_runs"]
                    for t0_run in estimator_run["t0_runs"]
                    for row in t0_run["policy_summary"]
                ]
            )
            with open(os.path.join(profile_dir, "profile_summary.json"), "w", encoding="utf-8") as handle:
                json.dump(profile_summary, handle, ensure_ascii=False, indent=2)
            run_summary["profiles"].append(profile_summary)
    finally:
        _send_shutdown(transport)
        transport.close()

    run_summary["run_wall_sec"] = time.perf_counter() - run_start
    run_summary["accuracy_summary"] = _aggregate_accuracy_summary(
        [
            row
            for profile in run_summary["profiles"]
            for estimator_run in profile.get("accuracy_estimator_runs", [])
            for t0_run in estimator_run.get("t0_runs", [])
            for row in t0_run.get("policy_summary", [])
        ]
    )
    with open(os.path.join(output_dir, "run_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, ensure_ascii=False, indent=2)
    logging.info("[A] Completed experiment")
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


run_node_a = _run_node_a_dynamic


def print_partition_plan(factory):
    print("=" * 80)
    print("ResNet56 partition plan")
    print("=" * 80)
    for item in factory.partition_plan:
        print("[{node_id}] partition={partition_idx} units={unit_start}:{unit_end} cost={total_cost:.0f} names={unit_names}".format(**item))


def main():
    parser = argparse.ArgumentParser(description="Standalone ResNet56 4-node Jetson pipeline with single-task style policies.")
    parser.add_argument("--node", choices=NODE_ORDER, help="Node ID. A is the controller.")
    parser.add_argument("--mode", choices=["experiment", "plan"], default="experiment")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--checkpoint_path", default=MODEL_CHECKPOINT)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_batches", type=int, default=DEFAULT_MAX_BATCHES)
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--download_data", action="store_true")
    parser.add_argument("--warmup_steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--t0_ratio", type=float, default=DEFAULT_T0_RATIO)
    parser.add_argument("--mu_values", default="0.01,0.1,1")
    parser.add_argument("--dynamic_timeslot_sizes", default="5,10,25")
    parser.add_argument("--max_dynamic_timeslot_count", type=int, default=DEFAULT_MAX_DYNAMIC_TIMESLOT_COUNT)
    parser.add_argument("--max_total_batches", type=int, default=None)
    parser.add_argument("--accuracy_estimator_modes", default=DEFAULT_ACCURACY_ESTIMATOR_MODES)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--stein_sigma", type=float, default=DEFAULT_STEIN_SIGMA)
    parser.add_argument("--stein_N", type=int, default=DEFAULT_STEIN_N)
    parser.add_argument("--stein_fast_max_batches", type=int, default=DEFAULT_STEIN_FAST_MAX_BATCHES)
    parser.add_argument("--compressor_profiles", default="topk,quantization,llmint8_fp16_int4")
    parser.add_argument("--channel_predictor_mode", default=DEFAULT_CHANNEL_UPDATE_MODE, choices=["warmup", "online"])
    parser.add_argument("--channel_model_type", default=DEFAULT_CHANNEL_MODEL_TYPE, choices=SUPPORTED_CHANNEL_MODEL_TYPES)
    parser.add_argument("--channel_window_size", type=int, default=DEFAULT_CHANNEL_WINDOW_SIZE)
    parser.add_argument("--transport_backend", default=TRANSPORT_BACKEND, choices=["http", "zeromq"])
    parser.add_argument("--tx_limit_mbps", type=float, default=DEFAULT_TX_LIMIT_MBPS)
    parser.add_argument("--tx_bucket_capacity_kb", type=float, default=DEFAULT_TX_BUCKET_CAPACITY_KB)
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    tx_bucket_capacity_bytes = int(float(args.tx_bucket_capacity_kb) * 1024.0)
    factory = ResNet56PartitionFactory(checkpoint_path=args.checkpoint_path, device=args.device)
    if args.mode == "plan":
        print_partition_plan(factory)
        return
    if args.node is None:
        parser.error("--node is required in experiment mode")

    if args.node == "A":
        run_node_a(
            device=args.device,
            checkpoint_path=args.checkpoint_path,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            data_root=args.data_root,
            download_data=args.download_data,
            warmup_steps=args.warmup_steps,
            t0_ratio=args.t0_ratio,
            mu_values_arg=args.mu_values,
            dynamic_timeslot_sizes_arg=args.dynamic_timeslot_sizes,
            max_dynamic_timeslot_count=args.max_dynamic_timeslot_count,
            max_total_batches=args.max_total_batches,
            accuracy_estimator_modes_arg=args.accuracy_estimator_modes,
            epsilon=args.epsilon,
            stein_sigma=args.stein_sigma,
            stein_N=args.stein_N,
            stein_fast_max_batches=args.stein_fast_max_batches,
            channel_predictor_mode=args.channel_predictor_mode,
            channel_model_type=args.channel_model_type,
            channel_window_size=args.channel_window_size,
            transport_backend=args.transport_backend,
            tx_limit_mbps=args.tx_limit_mbps,
            tx_bucket_capacity_bytes=tx_bucket_capacity_bytes,
            compressor_profiles_arg=args.compressor_profiles,
        )
    else:
        run_worker(
            node_id=args.node,
            device=args.device,
            checkpoint_path=args.checkpoint_path,
            transport_backend=args.transport_backend,
            tx_limit_mbps=args.tx_limit_mbps,
            tx_bucket_capacity_bytes=tx_bucket_capacity_bytes,
        )


if __name__ == "__main__":
    main()
