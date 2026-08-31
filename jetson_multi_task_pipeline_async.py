#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Event-driven async 4-node Jetson multi-task pipeline for mixed ResNet56 + Flan-T5.
"""

import argparse
import copy
import datetime
import gc
import json
import logging
import os
import pickle
import queue
import threading
import time
import uuid
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import torch

from compressors import get_compressor
from channel_estimator import ChannelEstimator, SUPPORTED_CHANNEL_MODEL_TYPES
from http_transport_dynamic import MessageType, create_transport_dynamic
from llmint8_eta_mapping import resolve_codec_execution_plan, validate_llmint8_mapping_entries
from multi_task_offline_plots import render_trial_plots as render_multi_trial_plots

from jetson_resnet_pipeline_singletask_policy import (
    DEFAULT_DATA_ROOT as RESNET_DEFAULT_DATA_ROOT,
    DEFAULT_DOWNLOAD_DATA as RESNET_DEFAULT_DOWNLOAD_DATA,
    MODEL_CHECKPOINT as RESNET_CHECKPOINT_PATH,
    Poly3AccuracyAdapter,
    ResNetFastAccuracyEvaluator,
    ResNet56PartitionFactory,
    SteinAccuracyEstimatorAdapter as SteinAccuracyEstimatorAdapter,
    load_cifar10_batches,
)
from jetson_flan_t5_pipeline_singletask_policy import (
    DEFAULT_MAX_INPUT_LENGTH as FLAN_DEFAULT_MAX_INPUT_LENGTH,
    DEFAULT_MODEL_NAME as FLAN_DEFAULT_MODEL_NAME,
    DEFAULT_NEGATIVE_TOKEN as FLAN_DEFAULT_NEGATIVE_TOKEN,
    DEFAULT_POSITIVE_TOKEN as FLAN_DEFAULT_POSITIVE_TOKEN,
    DEFAULT_PROMPT_TEMPLATE as FLAN_DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_SPLIT as FLAN_DEFAULT_SPLIT,
    FlanT5FastAccuracyEvaluator,
    FlanT5PartitionFactory,
    build_prompt as build_flan_prompt,
    load_sst2_samples,
    resolve_single_token_verbalizers,
)


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

TRANSPORT_BACKEND = "http_dynamic"
TRANSPORT_READY_TIMEOUT_SEC = 180.0
NUM_TRANSFER_POINTS = 3
NUM_NODES = 4
DEFAULT_DEVICE = "cuda"
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "multi_task_async")
DEFAULT_TX_LIMIT_MBPS = 25.0
DEFAULT_INITIAL_TOTAL_BANDWIDTH_MBPS = DEFAULT_TX_LIMIT_MBPS
DEFAULT_TX_BUCKET_CAPACITY_BYTES = 64 * 1024
DEFAULT_MAX_INFLIGHT_PER_TASK = 2
DEFAULT_WORKER_THREADS = 2
DEFAULT_WARMUP_SAMPLES_PER_TASK = 10
DEFAULT_CHANNEL_PROBE_STEPS = 10
DEFAULT_TARGET_TIME_RATIO = 0.75
DEFAULT_NO_CSI_TARGET_TIME_RATIOS = [0.9, 0.75, 0.5]
DEFAULT_TARGET_TASK1_RATIOS = [0.75]
DEFAULT_TARGET_TASK2_RATIOS = [0.9]
MIN_LINK_SHARE = 1e-3
DEFAULT_MU_VALUES = [0.1, 0.01]
DEFAULT_MU_FALLBACK = float(DEFAULT_MU_VALUES[0])
DEFAULT_EPSILON = 0.1
DEFAULT_J = 1
DEFAULT_BANDWIDTH_WARNING_RATIO = 0.75
DEFAULT_CODEC_NAMES = ["topk", "quantization", "llmint8"]
DEFAULT_ACCURACY_ESTIMATOR_MODES = "fitting_model,stein_estimator"
DEFAULT_STEIN_SIGMA = 0.05
DEFAULT_STEIN_N = 5
DEFAULT_STEIN_FAST_MAX_BATCHES = 1
DEFAULT_STEIN_FAST_MAX_SAMPLES = 1
DEFAULT_CHANNEL_MODEL_TYPE = "mean_factor"
DEFAULT_CHANNEL_UPDATE_MODE = "online"
DEFAULT_CHANNEL_WINDOW_SIZE = 5
DEFAULT_DYNAMIC_TIMESLOT_SIZES = [5, 10, 25]
DEFAULT_MAX_DYNAMIC_TIMESLOT_COUNT = None
DEFAULT_NON_MU_POLICY_COMPLETION_LIMIT = 50

DEFAULT_RESNET_TARGET_RATE_HZ = 1.0
DEFAULT_RESNET_MAX_ITEMS = 100
DEFAULT_RESNET_WEIGHT = 1.0
DEFAULT_RESNET_BATCH_SIZE = 100
ESTIMATOR_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "accuracy_estimators")
DEFAULT_RESNET_TOPK_ESTIMATOR_PATH = os.path.join(ESTIMATOR_MODEL_DIR, "jetson_resnet_3tp_poly3_flex.pkl")
DEFAULT_RESNET_QUANTIZATION_ESTIMATOR_PATH = os.path.join(ESTIMATOR_MODEL_DIR, "jetson_resnet_3tp_quantization_poly3_flex.pkl")
DEFAULT_RESNET_LLMINT8_ESTIMATOR_PATH = os.path.join(ESTIMATOR_MODEL_DIR, "jetson_resnet_3tp_llmint8_fp16_int4_poly3_flex.pkl")
DEFAULT_RESNET_LLMINT8_MAPPING_PATH = os.path.join(ESTIMATOR_MODEL_DIR, "raw_accuracy_resnet56_llmint8_eta_mapping.json")
DEFAULT_RESNET_OUTLIER_PRECISION = "fp16"
DEFAULT_RESNET_REGULAR_PRECISION = "int4"

DEFAULT_FLAN_TARGET_RATE_HZ = 5.0
DEFAULT_FLAN_MAX_ITEMS = 100
DEFAULT_FLAN_WEIGHT = 1.0
DEFAULT_FLAN_BATCH_SIZE = 1
DEFAULT_FLAN_TOPK_ESTIMATOR_PATH = os.path.join(ESTIMATOR_MODEL_DIR, "flan_t5_sst2_3tp_topk_poly3_flex.pkl")
DEFAULT_FLAN_QUANTIZATION_ESTIMATOR_PATH = os.path.join(ESTIMATOR_MODEL_DIR, "flan_t5_sst2_3tp_quantization_poly3_flex.pkl")
DEFAULT_FLAN_LLMINT8_ESTIMATOR_PATH = os.path.join(ESTIMATOR_MODEL_DIR, "flan_t5_sst2_3tp_llmint8_fp16_int8_poly3_flex.pkl")
DEFAULT_FLAN_LLMINT8_MAPPING_PATH = os.path.join(ESTIMATOR_MODEL_DIR, "raw_accuracy_flan_t5_sst2_3tp_llmint8_eta_mapping.json")
DEFAULT_FLAN_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "sst2_validation.jsonl")
DEFAULT_FLAN_OUTLIER_PRECISION = "fp16"
DEFAULT_FLAN_REGULAR_PRECISION = "int8"


def _current_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _sanitize_tag(text):
    chars = []
    for ch in str(text).strip().lower():
        chars.append(ch if ch.isalnum() else "_")
    return "".join(chars).strip("_")


def _format_tx_limit_tag(tx_limit_mbps=None):
    if tx_limit_mbps is None:
        return "tx_unlimited"
    rate = float(tx_limit_mbps)
    if rate.is_integer():
        rate_str = str(int(rate))
    else:
        rate_str = ("{:.3f}".format(rate)).rstrip("0").rstrip(".")
    return "tx_{}mbps".format(rate_str.replace(".", "p"))


def _format_float4(value):
    return "{:.4f}".format(float(value))


def _per_link_bandwidth_bps(tx_limit_mbps):
    return float(tx_limit_mbps) * 1e6


def _make_run_output_dir(node_id, base_output_dir, device, tx_limit_mbps):
    device_tag = str(device or "na").replace(":", "_")
    bw_tag = _format_tx_limit_tag(tx_limit_mbps)
    run_name = "{}_async_multitask_{}_{}_{}".format(
        node_id,
        device_tag,
        bw_tag,
        _current_timestamp(),
    )
    output_dir = os.path.join(base_output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_path(base_dir, path_value):
    if path_value is None:
        return None
    if os.path.isabs(path_value):
        return path_value
    return os.path.normpath(os.path.join(base_dir, path_value))


def _coalesce(config, key, default):
    return copy.deepcopy(config[key]) if key in config else copy.deepcopy(default)


def _normalize_codec_name(codec_name):
    name = str(codec_name).strip().lower()
    if name in {"topk"}:
        return "topk"
    if name in {"quant", "quantization"}:
        return "quantization"
    if name in {"llmint8", "llm_int8"}:
        return "llmint8"
    raise ValueError("Unsupported codec '{}'".format(codec_name))


def _parse_target_time_ratios(ratios_arg):
    if ratios_arg is None:
        return [float(x) for x in DEFAULT_NO_CSI_TARGET_TIME_RATIOS]
    if isinstance(ratios_arg, (list, tuple)):
        values = [float(x) for x in ratios_arg]
        if not values:
            raise ValueError("Expected at least one target_time_ratio")
        return values
    values = []
    for item in str(ratios_arg).split(","):
        text = item.strip()
        if not text:
            continue
        values.append(float(text))
    if not values:
        raise ValueError("Expected at least one target_time_ratio")
    return values


def _target_ratio_arg_key_for_task_id(task_id):
    try:
        task_id_int = int(task_id)
    except Exception:
        return None
    return "target_task{}_ratios".format(task_id_int)


def _resolve_target_ratio_groups(runtime_config, tasks):
    global_ratio_arg = runtime_config.get("target_ratios")
    global_ratios = (
        _parse_target_time_ratios(global_ratio_arg)
        if global_ratio_arg is not None
        else []
    )
    ratio_lists_by_task = {}
    max_len = 0
    for task in tasks:
        task_key = _target_ratio_arg_key_for_task_id(int(task.logical_task_id))
        task_specific_arg = runtime_config.get(task_key) if task_key is not None else None
        if task_specific_arg is not None:
            ratio_list = _parse_target_time_ratios(task_specific_arg)
        elif global_ratios:
            ratio_list = list(global_ratios)
        elif int(task.logical_task_id) == 1:
            ratio_list = list(DEFAULT_TARGET_TASK1_RATIOS)
        elif int(task.logical_task_id) == 2:
            ratio_list = list(DEFAULT_TARGET_TASK2_RATIOS)
        else:
            ratio_list = [float(DEFAULT_NO_CSI_TARGET_TIME_RATIOS[0])]
        ratio_lists_by_task[int(task.logical_task_id)] = list(ratio_list)
        max_len = max(max_len, len(ratio_list))

    groups = []
    for task_id, ratio_list in ratio_lists_by_task.items():
        if len(ratio_list) not in {1, max_len}:
            raise ValueError(
                "target_time_ratios for task {} has length {}, expected 1 or {}".format(
                    int(task_id), int(len(ratio_list)), int(max_len)
                )
            )
    for idx in range(max_len):
        group = {}
        for task in tasks:
            task_id = int(task.logical_task_id)
            ratio_list = ratio_lists_by_task[task_id]
            ratio_value = ratio_list[0] if len(ratio_list) == 1 else ratio_list[idx]
            group[task_id] = float(ratio_value)
        groups.append(group)
    return groups


def _parse_dynamic_timeslot_sizes(values_arg):
    if values_arg is None:
        return [int(x) for x in DEFAULT_DYNAMIC_TIMESLOT_SIZES]
    if isinstance(values_arg, (list, tuple)):
        values = [max(1, int(x)) for x in values_arg]
        if not values:
            raise ValueError("Expected at least one dynamic_timeslot_size")
        return values
    values = []
    for item in str(values_arg).split(","):
        text = item.strip()
        if not text:
            continue
        values.append(max(1, int(text)))
    if not values:
        raise ValueError("Expected at least one dynamic_timeslot_size")
    return values


def _build_inline_config(args):
    legacy_rate_task1 = getattr(args, "resnet_target_rate_hz_legacy", None)
    legacy_rate_task2 = getattr(args, "flan_target_rate_hz_legacy", None)
    if legacy_rate_task1 is not None or legacy_rate_task2 is not None:
        raise ValueError(
            "online multi-task no longer accepts explicit target rate inputs; use "
            "--target_task1_ratios and --target_task2_ratios instead"
        )
    global_target_ratios = getattr(args, "target_ratios", None)
    if global_target_ratios is None:
        global_target_ratios = getattr(args, "target_time_ratios_legacy", None)
    task1_target_ratios = getattr(args, "target_task1_ratios", None)
    if task1_target_ratios is None:
        task1_target_ratios = getattr(args, "resnet_target_time_ratios_legacy", None)
    task2_target_ratios = getattr(args, "target_task2_ratios", None)
    if task2_target_ratios is None:
        task2_target_ratios = getattr(args, "flan_target_time_ratios_legacy", None)
    return {
        "codec_names": [_normalize_codec_name(item) for item in str(args.compressor_profiles).split(",") if item.strip()],
        "accuracy_estimator_modes": _parse_accuracy_estimator_modes(args.accuracy_estimator_modes),
        "target_ratios": (
            None if global_target_ratios is None else _parse_target_time_ratios(global_target_ratios)
        ),
        "target_task1_ratios": (
            None if task1_target_ratios is None else _parse_target_time_ratios(task1_target_ratios)
        ),
        "target_task2_ratios": (
            None if task2_target_ratios is None else _parse_target_time_ratios(task2_target_ratios)
        ),
        "dynamic_timeslot_sizes": _parse_dynamic_timeslot_sizes(args.dynamic_timeslot_sizes),
        "max_dynamic_timeslot_count": (
            None if args.max_dynamic_timeslot_count is None else int(args.max_dynamic_timeslot_count)
        ),
        "max_total_batches": (
            None if getattr(args, "max_total_batches", None) is None else int(args.max_total_batches)
        ),
        "mu_values": _parse_mu_values(args.mu_values),
        "warmup_samples_per_task": int(args.warmup_samples_per_task),
        "stein_sigma": float(args.stein_sigma),
        "stein_N": int(args.stein_N),
        "stein_fast_max_batches": int(args.stein_fast_max_batches),
        "stein_fast_max_samples": int(args.stein_fast_max_samples),
        "tasks": [
            {
                "task_id": 1,
                "family": "resnet",
                "name": "resnet_task",
                "model": "resnet56",
                "dataset": "cifar10",
                "weight": float(args.resnet_weight),
                "target_rate_hz": float(DEFAULT_RESNET_TARGET_RATE_HZ),
                "eta_min": [0.125, 0.125, 0.125],
                "batch_size": int(args.resnet_batch_size),
                "max_items": int(args.resnet_max_items),
                "checkpoint_path": str(args.resnet_checkpoint_path),
                "data_root": str(args.resnet_data_root),
                "download_data": bool(args.resnet_download_data),
                "codec_variants": {
                    "topk": {
                        "accuracy_model": {"type": "poly3", "path": str(args.resnet_topk_estimator_path)}
                    },
                    "quantization": {
                        "accuracy_model": {"type": "poly3", "path": str(args.resnet_quantization_estimator_path)}
                    },
                    "llmint8": {
                        "accuracy_model": {"type": "poly3", "path": str(args.resnet_llmint8_estimator_path)},
                        "outlier_precision": str(args.resnet_llmint8_outlier_precision),
                        "regular_precision": str(args.resnet_llmint8_regular_precision),
                        "llmint8_mapping_source": str(args.resnet_llmint8_mapping_path),
                    },
                },
            },
            {
                "task_id": 2,
                "family": "flan_t5",
                "name": "flan_t5_task",
                "model": "flan_t5_base",
                "dataset": "sst2",
                "weight": float(args.flan_weight),
                "target_rate_hz": float(DEFAULT_FLAN_TARGET_RATE_HZ),
                "eta_min": [0.125, 0.125, 0.125],
                "batch_size": int(args.flan_batch_size),
                "max_items": int(args.flan_max_items),
                "model_name": str(args.flan_model_name),
                "dataset_path": args.flan_dataset_path,
                "split": str(args.flan_split),
                "max_input_length": int(args.flan_max_input_length),
                "prompt_template": str(args.flan_prompt_template),
                "positive_token": str(args.flan_positive_token),
                "negative_token": str(args.flan_negative_token),
                "codec_variants": {
                    "topk": {
                        "accuracy_model": {"type": "poly3", "path": str(args.flan_topk_estimator_path)}
                    },
                    "quantization": {
                        "accuracy_model": {"type": "poly3", "path": str(args.flan_quantization_estimator_path)}
                    },
                    "llmint8": {
                        "accuracy_model": {"type": "poly3", "path": str(args.flan_llmint8_estimator_path)},
                        "outlier_precision": str(args.flan_llmint8_outlier_precision),
                        "regular_precision": str(args.flan_llmint8_regular_precision),
                        "llmint8_mapping_source": str(args.flan_llmint8_mapping_path),
                    },
                },
            },
        ],
    }


def _save_rows(output_path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def _tp_stats_rows_from_history(tp_stats_history):
    rows = []
    for idx, tp_stats in enumerate(tp_stats_history):
        row = {"t": int(idx)}
        for link_idx, link_stats in enumerate(tp_stats):
            bytes_per_sec = float(link_stats[0])
            elapsed_sec = float(link_stats[1])
            row["link_{}_bytes_per_sec".format(link_idx)] = bytes_per_sec
            row["link_{}_elapsed_sec".format(link_idx)] = elapsed_sec
            row["link_{}_bandwidth_bps".format(link_idx)] = bytes_per_sec * 8.0 / max(elapsed_sec, 1e-9)
        rows.append(row)
    return rows


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
        requested = ["fitting_model"]
    elif isinstance(modes_arg, (list, tuple)):
        requested = [str(item).strip() for item in modes_arg if str(item).strip()]
    else:
        requested = [item.strip() for item in str(modes_arg).split(",") if item.strip()]
    if not requested:
        requested = ["fitting_model"]
    normalized = []
    for item in requested:
        mode = str(item).strip().lower()
        if mode in {"fitting", "fitting_model"}:
            mode = "fitting_model"
        elif mode in {"stein", "steins", "steains", "stein_estimator"}:
            mode = "stein_estimator"
        else:
            raise ValueError("Unsupported accuracy_estimator_mode '{}'".format(item))
        if mode not in normalized:
            normalized.append(mode)
    return normalized


def _order_mu_values_for_multitask(mu_values):
    ordered = []
    seen = set()
    for value in [float(item) for item in mu_values]:
        key = float(value)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


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


@dataclass
class RuntimeTaskDef:
    logical_task_id: int
    family: str
    name: str
    model: str
    dataset: str
    weight: float
    eta_min: np.ndarray
    target_rate_hz: float
    batch_size: int
    estimator: Any
    codec_name: str
    outlier_precision: Optional[str]
    regular_precision: Optional[str]
    llmint8_mapping_entries: List[Dict[str, Any]]
    warmup_items: List[Dict[str, Any]]
    experiment_items: List[Dict[str, Any]]
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_links(self) -> int:
        return int(len(self.eta_min))

    def predict_accuracy(self, eta: Sequence[float]) -> float:
        return float(self.estimator.predict(eta))

    def gradient_accuracy(self, eta: Sequence[float]) -> np.ndarray:
        return np.asarray(self.estimator.gradient(eta), dtype=float)


@dataclass
class LatestSnapshot:
    a_ref: np.ndarray
    fixed_overhead_bytes: np.ndarray
    tau_list: np.ndarray
    total_delay_sec: float
    end_to_end_delay_sec: float
    tp_bandwidth_bps: np.ndarray
    sample_correct: float
    running_accuracy: float
    service_delay_sec: float
    communication_delay_sec: float
    miscellaneous_delay_sec: float

    @property
    def compute_floor(self) -> float:
        return float(np.max(self.tau_list))

    def estimated_payload_bytes(self, eta: Sequence[float]) -> np.ndarray:
        eta_arr = np.asarray(eta, dtype=float)
        return np.asarray(self.fixed_overhead_bytes, dtype=float) + np.asarray(self.a_ref, dtype=float) * eta_arr


def _compute_task_predictions(tasks_by_id, allocations, snapshots, total_bw_bps, enable_accuracy_prediction=True):
    predicted_accuracy_by_task = {}
    predicted_delay_by_task = {}
    if not enable_accuracy_prediction:
        for tid in tasks_by_id.keys():
            predicted_accuracy_by_task[int(tid)] = np.nan
            predicted_delay_by_task[int(tid)] = np.nan
        return predicted_accuracy_by_task, predicted_delay_by_task
    total_bw_bps = np.asarray(total_bw_bps, dtype=float)
    for tid, task in tasks_by_id.items():
        snap = snapshots.get(int(tid))
        alloc = allocations.get(int(tid))
        if snap is None or alloc is None:
            predicted_accuracy_by_task[int(tid)] = np.nan
            predicted_delay_by_task[int(tid)] = np.nan
            continue
        eta = np.asarray(alloc["eta"], dtype=float)
        s_comm = np.asarray(alloc["s_comm"], dtype=float)
        s_comp = np.asarray(alloc["s_comp"], dtype=float)
        predicted_accuracy_by_task[int(tid)] = float(task.predict_accuracy(eta))
        comp_delays = [
            float(snap.tau_list[node_idx]) / max(float(s_comp[node_idx]), 1e-12)
            for node_idx in range(NUM_NODES)
        ]
        comm_delays = [
            float(snap.estimated_payload_bytes(eta)[link_idx])
            / max(float(total_bw_bps[link_idx]) * max(float(s_comm[link_idx]), 1e-12), 1e-12)
            for link_idx in range(task.num_links)
        ]
        predicted_delay_by_task[int(tid)] = float(max(comp_delays + comm_delays))
    return predicted_accuracy_by_task, predicted_delay_by_task


class SharedBandwidthState:
    def __init__(self, task_ids: Sequence[int], num_links: int, initial_total_bps: float):
        self.task_ids = [int(item) for item in task_ids]
        self.num_links = int(num_links)
        self.initial_total_bps = float(initial_total_bps)
        per_task_seed = self.initial_total_bps / max(len(self.task_ids), 1)
        self.latest_by_task = {
            int(task_id): np.full(self.num_links, per_task_seed, dtype=float)
            for task_id in self.task_ids
        }

    def observe(self, logical_task_id: int, tp_bandwidth_bps: Sequence[float]):
        self.latest_by_task[int(logical_task_id)] = np.asarray(tp_bandwidth_bps, dtype=float)

    def total_bandwidth_bps(self) -> np.ndarray:
        total = np.zeros(self.num_links, dtype=float)
        for values in self.latest_by_task.values():
            total += np.asarray(values, dtype=float)
        return np.where(total > 1e-6, total, self.initial_total_bps)

    def total_tp_stats(self) -> List[Tuple[float, float]]:
        total_bps = self.total_bandwidth_bps()
        return [(float(bps) / 8.0, 1.0) for bps in total_bps]


def build_channel_estimator(device="cpu",
                            model_type=DEFAULT_CHANNEL_MODEL_TYPE,
                            update_mode=DEFAULT_CHANNEL_UPDATE_MODE,
                            window_size=DEFAULT_CHANNEL_WINDOW_SIZE,
                            default_bandwidth_bps=10e6):
    # ChannelEstimator stores/returns throughput in bytes/sec.
    return ChannelEstimator(
        n_links=NUM_TRANSFER_POINTS,
        model_type=model_type,
        update_mode=update_mode,
        window_size=window_size,
        default_bandwidth=float(default_bandwidth_bps) / 8.0,
        device=device,
    )


def _build_seeded_channel_estimator(warmup_tp_stats,
                                    device,
                                    model_type,
                                    update_mode,
                                    window_size,
                                    default_bandwidth_bps):
    estimator = build_channel_estimator(
        device=device,
        model_type=model_type,
        update_mode=update_mode,
        window_size=window_size,
        default_bandwidth_bps=default_bandwidth_bps,
    )
    for tp_stats in warmup_tp_stats:
        estimator.observe_task(tp_stats, refit=False)
    estimator.fit()
    return estimator


def _probe_mean_total_bandwidth_bps(tp_stats_history, fallback_total_bps):
    fallback = np.full(NUM_TRANSFER_POINTS, float(fallback_total_bps), dtype=float)
    if not tp_stats_history:
        return fallback
    rows = []
    for tp_stats in tp_stats_history:
        per_link = []
        for bytes_sent, elapsed_sec in tp_stats:
            elapsed = max(float(elapsed_sec), 1e-9)
            per_link.append((float(bytes_sent) * 8.0) / elapsed)
        if len(per_link) == NUM_TRANSFER_POINTS:
            rows.append(np.asarray(per_link, dtype=float))
    if not rows:
        return fallback
    mean_values = np.mean(np.stack(rows, axis=0), axis=0)
    return np.asarray(mean_values, dtype=float)


def _current_total_bandwidth_bps(channel_estimator, shared_bandwidth):
    if channel_estimator is None:
        return shared_bandwidth.total_bandwidth_bps()
    try:
        has_history = any(len(link.history) > 0 for link in channel_estimator.links)
    except Exception:
        has_history = False
    if not has_history:
        return shared_bandwidth.total_bandwidth_bps()
    predicted_bytes_per_sec = np.asarray(channel_estimator.predict_all(), dtype=float)
    return predicted_bytes_per_sec * 8.0


def _warn_if_per_link_bandwidth_low(link_bw_bps, tx_limit_mbps, context_label, ratio=DEFAULT_BANDWIDTH_WARNING_RATIO):
    if tx_limit_mbps is None:
        return
    link_bw_bps = np.asarray(link_bw_bps, dtype=float)
    threshold_bps = float(tx_limit_mbps) * float(ratio) * 1e6
    low_links = [
        {
            "link_idx": int(idx),
            "predicted_mbps": float(value) / 1e6,
            "threshold_mbps": threshold_bps / 1e6,
        }
        for idx, value in enumerate(link_bw_bps)
        if float(value) < threshold_bps
    ]
    if not low_links:
        return
    logging.warning(
        "[bandwidth-check][%s] Per-link bandwidth fell below %.0f%% of tx_limit_mbps=%.4f. details=%s",
        str(context_label),
        float(ratio) * 100.0,
        float(tx_limit_mbps),
        low_links,
    )


class ThreadSafeTransportSender:
    def __init__(self, transport):
        self.transport = transport

    def send(self, peer_id, msg_type, payload, logical_task_id: Optional[int] = None, link_idx: Optional[int] = None, rate_bps: Optional[float] = None, handshake_meta: Optional[Dict[str, Any]] = None):
        if (
            logical_task_id is None
            or link_idx is None
            or rate_bps is None
            or not hasattr(self.transport, "send_dynamic")
        ):
            bytes_sent, send_sec = self.transport.send(peer_id, msg_type, payload, handshake_meta=handshake_meta)
            return bytes_sent, float(send_sec)
        bytes_sent, send_sec = self.transport.send_dynamic(
            peer_id,
            msg_type,
            payload,
            logical_task_id=int(logical_task_id),
            link_idx=int(link_idx),
            rate_bps=float(rate_bps),
            handshake_meta=handshake_meta,
        )
        return bytes_sent, float(send_sec)


class BaseCommOnlyPolicy:
    def __init__(self, policy_key: str, policy_name: str, tasks: Sequence[RuntimeTaskDef]):
        self.policy_key = str(policy_key)
        self.policy_name = str(policy_name)
        self.tasks = list(tasks)
        self.mu = None

    def update_dual(self, actual_delay_by_task: Dict[int, float], tasks_by_id: Dict[int, RuntimeTaskDef]):
        return None

    def current_allocations(self, snapshots: Dict[int, LatestSnapshot], total_bw_bps: np.ndarray) -> Dict[int, Dict[str, np.ndarray]]:
        raise NotImplementedError


def _policy_uses_fixed_channel(policy=None, algorithm_type=None):
    policy_key = str(algorithm_type or getattr(policy, "policy_key", "")).strip().lower()
    return policy_key in {
        "no_compression_multi_baseline",
        "max_compression_multi_baseline",
    }


def _policy_uses_mu(policy=None, algorithm_type=None):
    policy_key = str(algorithm_type or getattr(policy, "policy_key", "")).strip().lower()
    return policy_key in {
        "no_csi_multi",
        "decoupled_equal_split_multi_baseline",
        "queue_proportional_multi_baseline",
    }


def _policy_uses_estimator(policy=None, algorithm_type=None):
    policy_key = str(algorithm_type or getattr(policy, "policy_key", "")).strip().lower()
    return policy_key in {
        "no_csi_multi",
        "decoupled_equal_split_multi_baseline",
        "queue_proportional_multi_baseline",
        "certainty_equivalence_multi_baseline",
    }


def _normalize_allocations_to_unit_sum(tasks, allocations, link_floor=0.0):
    task_ids = [int(task.logical_task_id) for task in tasks]
    if not task_ids:
        return allocations
    for link_idx in range(NUM_TRANSFER_POINTS):
        raw = np.asarray(
            [max(float(np.asarray(allocations[tid]["s_comm"], dtype=float)[link_idx]), 0.0) for tid in task_ids],
            dtype=float,
        )
        raw_sum = float(np.sum(raw))
        if raw_sum <= 1e-12:
            shares = np.full(len(task_ids), 1.0 / float(len(task_ids)), dtype=float)
        else:
            shares = raw / raw_sum
            if float(link_floor) > 0.0:
                floor_total = float(link_floor) * len(task_ids)
                if floor_total < 1.0:
                    shares = np.maximum(shares, float(link_floor))
                    shares = shares / max(float(np.sum(shares)), 1e-12)
        for pos, tid in enumerate(task_ids):
            allocations[tid]["s_comm"][link_idx] = float(shares[pos])
    for node_idx in range(NUM_NODES):
        raw = np.asarray(
            [max(float(np.asarray(allocations[tid]["s_comp"], dtype=float)[node_idx]), 0.0) for tid in task_ids],
            dtype=float,
        )
        raw_sum = float(np.sum(raw))
        if raw_sum <= 1e-12:
            shares = np.full(len(task_ids), 1.0 / float(len(task_ids)), dtype=float)
        else:
            shares = raw / raw_sum
        for pos, tid in enumerate(task_ids):
            allocations[tid]["s_comp"][node_idx] = float(shares[pos])
    return allocations


def _solve_single_task_eta_with_fixed_resources_online(task, snapshot, c_eff, max_comp_delay, mu, lambda_t):
    c_eff = np.asarray(c_eff, dtype=float)
    bounds = [(float(task.eta_min[idx]), 1.0) for idx in range(task.num_links)] + [(float(max_comp_delay), None)]
    x0 = np.asarray([bound[0] for bound in bounds], dtype=float)

    def objective(x):
        eta_flat = np.asarray(x[:-1], dtype=float)
        z_val = float(x[-1])
        acc = float(task.predict_accuracy(eta_flat))
        grad_acc = np.asarray(task.gradient_accuracy(eta_flat), dtype=float)
        grad = np.zeros_like(x)
        grad[:-1] = -float(task.weight) * grad_acc
        grad[-1] = float(mu) * float(lambda_t)
        return -float(task.weight) * acc + float(mu) * float(lambda_t) * z_val, grad

    constraints = []
    for link_idx in range(task.num_links):
        scale = max(float(c_eff[link_idx]), 1e-12)
        overhead_beta = float(snapshot.fixed_overhead_bytes[link_idx]) / scale
        activation_beta = float(snapshot.a_ref[link_idx]) / scale
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x, link_idx=link_idx, overhead_beta=overhead_beta, activation_beta=activation_beta: (
                    float(x[-1]) - overhead_beta - activation_beta * float(x[link_idx])
                ),
                "jac": lambda x, link_idx=link_idx, activation_beta=activation_beta: np.asarray(
                    [-activation_beta if j == link_idx else (1.0 if j == len(x) - 1 else 0.0) for j in range(len(x))],
                    dtype=float,
                ),
            }
        )
    result = minimize(fun=objective, x0=x0, method="SLSQP", jac=True, bounds=bounds, constraints=constraints)
    eta = np.asarray(result.x[:-1] if result.success else x0[:-1], dtype=float)
    return np.clip(eta, np.asarray(task.eta_min, dtype=float), 1.0)


class NoCompressionMultiTaskBaseline(BaseCommOnlyPolicy):
    def __init__(self, tasks):
        super().__init__("no_compression_multi_baseline", "Baseline: no compression", tasks)

    def current_allocations(self, snapshots, total_bw_bps):
        num_tasks = max(len(self.tasks), 1)
        allocations = {}
        for task in self.tasks:
            allocations[task.logical_task_id] = {
                "eta": np.ones(task.num_links, dtype=float),
                "s_comm": np.full(task.num_links, 1.0 / float(num_tasks), dtype=float),
                "s_comp": np.full(NUM_NODES, 1.0 / float(num_tasks), dtype=float),
            }
        return allocations


class MaxCompressionMultiTaskBaseline(BaseCommOnlyPolicy):
    def __init__(self, tasks):
        super().__init__("max_compression_multi_baseline", "Baseline: max compression", tasks)

    def current_allocations(self, snapshots, total_bw_bps):
        num_tasks = max(len(self.tasks), 1)
        allocations = {}
        for task in self.tasks:
            allocations[task.logical_task_id] = {
                "eta": np.asarray(task.eta_min, dtype=float).copy(),
                "s_comm": np.full(task.num_links, 1.0 / float(num_tasks), dtype=float),
                "s_comp": np.full(NUM_NODES, 1.0 / float(num_tasks), dtype=float),
            }
        return allocations


class EqualShareMultiTaskBaseline(BaseCommOnlyPolicy):
    def __init__(self, tasks):
        super().__init__("equal_share_multi_baseline", "Baseline: equal share", tasks)

    def current_allocations(self, snapshots, total_bw_bps):
        num_tasks = max(len(self.tasks), 1)
        share_vec = np.full(NUM_TRANSFER_POINTS, 1.0 / float(num_tasks), dtype=float)
        allocations = {}
        for task in self.tasks:
            snap = snapshots.get(task.logical_task_id)
            if snap is None:
                eta = np.asarray(task.eta_min, dtype=float).copy()
            else:
                eta = []
                for link_idx in range(task.num_links):
                    raw_budget = (float(share_vec[link_idx]) * float(total_bw_bps[link_idx])) / max(
                        float(task.target_rate_hz),
                        1e-12,
                    )
                    raw = (raw_budget - float(snap.fixed_overhead_bytes[link_idx])) / max(float(snap.a_ref[link_idx]), 1e-12)
                    eta.append(max(float(task.eta_min[link_idx]), min(1.0, raw)))
                eta = np.asarray(eta, dtype=float)
            allocations[task.logical_task_id] = {
                "eta": eta,
                "s_comm": share_vec.copy(),
                "s_comp": np.full(NUM_NODES, 1.0 / float(num_tasks), dtype=float),
            }
        return allocations


class HistoricalAverageCertaintyEquivalenceMultiTaskBaseline(BaseCommOnlyPolicy):
    def __init__(self, tasks):
        super().__init__("certainty_equivalence_multi_baseline", "No-CSI Baseline: moving average", tasks)

    def current_allocations(self, snapshots, total_bw_bps):
        if not snapshots:
            return EqualShareMultiTaskBaseline(self.tasks).current_allocations(snapshots, total_bw_bps)
        for node_idx in range(NUM_NODES):
            comp_sum = sum(
                float(snapshots[int(task.logical_task_id)].tau_list[node_idx]) * float(task.target_rate_hz)
                for task in self.tasks
            )
            if comp_sum > 1.0:
                return MaxCompressionMultiTaskBaseline(self.tasks).current_allocations(snapshots, total_bw_bps)
        for link_idx in range(NUM_TRANSFER_POINTS):
            comm_sum = sum(
                float(snapshots[int(task.logical_task_id)].a_ref[link_idx]) * float(task.eta_min[link_idx]) * float(task.target_rate_hz)
                / max(float(total_bw_bps[link_idx]), 1e-12)
                for task in self.tasks
            )
            if comm_sum > 1.0:
                return MaxCompressionMultiTaskBaseline(self.tasks).current_allocations(snapshots, total_bw_bps)
        context = []
        for task in self.tasks:
            tid = int(task.logical_task_id)
            for link_idx in range(task.num_links):
                beta = float(total_bw_bps[link_idx]) / max(
                    float(task.target_rate_hz) * float(snapshots[tid].a_ref[link_idx]),
                    1e-12,
                )
                s_min = (
                    float(snapshots[tid].a_ref[link_idx]) * float(task.eta_min[link_idx]) * float(task.target_rate_hz)
                ) / max(float(total_bw_bps[link_idx]), 1e-12)
                s_max = min(1.0, 1.0 / max(beta, 1e-12))
                context.append(
                    {
                        "task": task,
                        "task_id": tid,
                        "link_idx": link_idx,
                        "beta": beta,
                        "s_min": s_min,
                        "s_max": s_max,
                    }
                )
        bounds = [(float(ctx["s_min"]), float(ctx["s_max"])) for ctx in context]
        x0 = np.asarray([float(ctx["s_min"]) for ctx in context], dtype=float)

        def objective(s_flat):
            total_obj = 0.0
            grad_flat = np.zeros_like(s_flat)
            for task in self.tasks:
                task_indices = [idx for idx, ctx in enumerate(context) if int(ctx["task_id"]) == int(task.logical_task_id)]
                eta_vec = []
                d_eta_ds = []
                for idx in task_indices:
                    beta = float(context[idx]["beta"])
                    s_val = float(s_flat[idx])
                    eta_vec.append(min(1.0, beta * s_val))
                    d_eta_ds.append(beta)
                eta_arr = np.asarray(eta_vec, dtype=float)
                acc = float(task.predict_accuracy(eta_arr))
                grad_acc = np.asarray(task.gradient_accuracy(eta_arr), dtype=float)
                total_obj += float(task.weight) * acc
                for local_i, flat_i in enumerate(task_indices):
                    grad_flat[flat_i] = float(task.weight) * float(grad_acc[local_i]) * float(d_eta_ds[local_i])
            return -float(total_obj), -grad_flat

        constraints = []
        for link_idx in range(NUM_TRANSFER_POINTS):
            indices = [idx for idx, ctx in enumerate(context) if int(ctx["link_idx"]) == int(link_idx)]
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda s_flat, idxs=indices: 1.0 - sum(float(s_flat[j]) for j in idxs),
                }
            )
        result = minimize(fun=objective, x0=x0, method="SLSQP", jac=True, bounds=bounds, constraints=constraints)
        s_flat = np.asarray(result.x if result.success else x0, dtype=float)
        allocations = {}
        for task in self.tasks:
            tid = int(task.logical_task_id)
            s_comp = np.asarray(
                [float(snapshots[tid].tau_list[node_idx]) * float(task.target_rate_hz) for node_idx in range(NUM_NODES)],
                dtype=float,
            )
            task_indices = [idx for idx, ctx in enumerate(context) if int(ctx["task_id"]) == tid]
            s_comm = []
            eta = []
            for idx in task_indices:
                s_val = float(s_flat[idx])
                beta = float(context[idx]["beta"])
                s_comm.append(s_val)
                eta.append(min(1.0, beta * s_val))
            allocations[tid] = {
                "eta": np.asarray(eta, dtype=float),
                "s_comm": np.asarray(s_comm, dtype=float),
                "s_comp": s_comp,
            }
        return _normalize_allocations_to_unit_sum(self.tasks, allocations)


class NoCSICommOnlyMultiTaskPolicy(BaseCommOnlyPolicy):
    def __init__(self, tasks: Sequence[RuntimeTaskDef], mu: float, epsilon: float, J: int):
        super().__init__("no_csi_multi", "No-CSI (Alg2) (mu={})".format(mu), tasks)
        self.mu = float(mu)
        self.epsilon = float(epsilon)
        self.J = int(J)
        self.lambda_k = {task.logical_task_id: float(self.epsilon) for task in self.tasks}

    def _phase_a_objective(self, z_flat, task_ids):
        grad = np.asarray([self.mu * self.lambda_k[int(tid)] for tid in task_ids], dtype=float)
        return float(np.dot(grad, z_flat)), grad

    def _phase_a_constraints(self, snapshots, current_eta, total_bw_bps, task_ids):
        constraints = []
        for node_idx in range(NUM_NODES):
            active_task_ids = [int(tid) for tid in task_ids]
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda z, node_idx=node_idx, active_task_ids=active_task_ids: 1.0
                    - sum(
                        float(snapshots[int(tid)].tau_list[node_idx]) / max(float(z[pos]), 1e-12)
                        for pos, tid in enumerate(active_task_ids)
                    ),
                }
            )
        for link_idx in range(NUM_TRANSFER_POINTS):
            active_task_ids = [int(tid) for tid in task_ids]
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda z, link_idx=link_idx, active_task_ids=active_task_ids: 1.0
                    - sum(
                        float(
                            snapshots[int(tid)].estimated_payload_bytes(current_eta[int(tid)])[link_idx]
                        )
                        / (max(float(total_bw_bps[link_idx]), 1e-12) * max(float(z[pos]), 1e-12))
                        for pos, tid in enumerate(active_task_ids)
                    ),
                }
            )
        return constraints

    def _phase_a_optimize_resources(self, snapshots, total_bw_bps, current_eta):
        task_ids = [int(task.logical_task_id) for task in self.tasks]
        bounds = [(max(float(snapshots[int(tid)].compute_floor), 1e-9), None) for tid in task_ids]
        x0 = np.asarray([float(bound[0]) * 1.1 for bound in bounds], dtype=float)
        result = minimize(
            fun=self._phase_a_objective,
            x0=x0,
            args=(task_ids,),
            method="SLSQP",
            jac=True,
            bounds=bounds,
            constraints=self._phase_a_constraints(snapshots, current_eta, total_bw_bps, task_ids),
        )
        z_flat = np.asarray(result.x if result.success else x0, dtype=float)
        updated = {}
        num_tasks = max(len(self.tasks), 1)
        for pos, task in enumerate(self.tasks):
            tid = int(task.logical_task_id)
            z_val = max(float(z_flat[pos]), 1e-9)
            snap = snapshots[tid]
            s_comp = np.asarray(
                [
                    float(snap.tau_list[node_idx]) / z_val
                    for node_idx in range(NUM_NODES)
                ],
                dtype=float,
            )
            s_comm = np.asarray(
                [
                    float(snap.estimated_payload_bytes(current_eta[tid])[link_idx])
                    / (max(float(total_bw_bps[link_idx]), 1e-12) * z_val)
                    for link_idx in range(task.num_links)
                ],
                dtype=float,
            )
            updated[tid] = {
                "s_comm": s_comm,
                "s_comp": s_comp,
            }
        return self._stabilize_allocations(updated)

    def _stabilize_allocations(self, allocations):
        task_ids = [int(task.logical_task_id) for task in self.tasks]
        if not task_ids:
            return allocations
        num_tasks = float(len(task_ids))

        for link_idx in range(NUM_TRANSFER_POINTS):
            raw = np.asarray(
                [max(float(allocations[tid]["s_comm"][link_idx]), 0.0) for tid in task_ids],
                dtype=float,
            )
            raw_sum = float(np.sum(raw))
            if np.any(raw <= 1e-12):
                logging.warning(
                    "[No-CSI] raw s_comm on link %s is degenerate before stabilization: %s",
                    int(link_idx),
                    {int(tid): float(raw[pos]) for pos, tid in enumerate(task_ids)},
                )
            if raw_sum <= 1e-12:
                shares = np.full(len(task_ids), 1.0 / num_tasks, dtype=float)
            else:
                shares = raw / raw_sum
                floor_total = MIN_LINK_SHARE * len(task_ids)
                if floor_total < 1.0:
                    shares = np.maximum(shares, MIN_LINK_SHARE)
                    shares = shares / max(float(np.sum(shares)), 1e-12)
            for pos, tid in enumerate(task_ids):
                allocations[tid]["s_comm"][link_idx] = float(shares[pos])

        for node_idx in range(NUM_NODES):
            raw = np.asarray(
                [max(float(allocations[tid]["s_comp"][node_idx]), 0.0) for tid in task_ids],
                dtype=float,
            )
            raw_sum = float(np.sum(raw))
            if raw_sum <= 1e-12:
                shares = np.full(len(task_ids), 1.0 / num_tasks, dtype=float)
            else:
                shares = raw / raw_sum
            for pos, tid in enumerate(task_ids):
                allocations[tid]["s_comp"][node_idx] = float(shares[pos])

        return allocations

    def _phase_b_optimize_configuration(self, snapshots, total_bw_bps, current_s):
        updated_eta = {}
        for task in self.tasks:
            tid = int(task.logical_task_id)
            snap = snapshots[tid]
            s_comm = np.asarray(current_s[tid]["s_comm"], dtype=float)
            s_comp = np.asarray(current_s[tid]["s_comp"], dtype=float)
            max_comp_delay = max(
                float(snap.tau_list[node_idx]) / max(float(s_comp[node_idx]), 1e-12)
                for node_idx in range(NUM_NODES)
            )
            bounds = [(float(task.eta_min[idx]), 1.0) for idx in range(task.num_links)] + [(max_comp_delay, None)]
            x0 = np.asarray([bound[0] for bound in bounds], dtype=float)

            def objective(x):
                eta_flat = np.asarray(x[:-1], dtype=float)
                z_val = float(x[-1])
                acc = float(task.predict_accuracy(eta_flat))
                grad_acc = np.asarray(task.gradient_accuracy(eta_flat), dtype=float)
                grad = np.zeros_like(x)
                grad[:-1] = -float(task.weight) * grad_acc
                grad[-1] = self.mu * self.lambda_k[tid]
                return -float(task.weight) * acc + self.mu * self.lambda_k[tid] * z_val, grad

            constraints = []
            for node_idx in range(NUM_NODES):
                comp_beta = float(snap.tau_list[node_idx]) / max(float(s_comp[node_idx]), 1e-12)
                constraints.append(
                    {
                        "type": "ineq",
                        "fun": lambda x, comp_beta=comp_beta: float(x[-1]) - comp_beta,
                        "jac": lambda x, comp_beta=comp_beta: np.asarray(
                            [1.0 if j == len(x) - 1 else 0.0 for j in range(len(x))],
                            dtype=float,
                        ),
                    }
                )
            for link_idx in range(task.num_links):
                scale = max(float(total_bw_bps[link_idx]) * max(float(s_comm[link_idx]), 1e-12), 1e-12)
                overhead_beta = float(snap.fixed_overhead_bytes[link_idx]) / scale
                activation_beta = float(snap.a_ref[link_idx]) / scale
                constraints.append(
                    {
                        "type": "ineq",
                        "fun": lambda x, link_idx=link_idx, overhead_beta=overhead_beta, activation_beta=activation_beta: (
                            float(x[-1]) - overhead_beta - activation_beta * float(x[link_idx])
                        ),
                        "jac": lambda x, link_idx=link_idx, activation_beta=activation_beta: np.asarray(
                            [-activation_beta if j == link_idx else (1.0 if j == len(x) - 1 else 0.0) for j in range(len(x))],
                            dtype=float,
                        ),
                    }
                )
            result = minimize(fun=objective, x0=x0, method="SLSQP", jac=True, bounds=bounds, constraints=constraints)
            updated_eta[tid] = np.asarray(result.x[:-1] if result.success else x0[:-1], dtype=float)
        return updated_eta

    def current_allocations(self, snapshots, total_bw_bps):
        if not snapshots:
            return EqualShareMultiTaskBaseline(self.tasks).current_allocations(snapshots, total_bw_bps)
        current_eta = {int(task.logical_task_id): np.asarray(task.eta_min, dtype=float).copy() for task in self.tasks}
        current_s = None
        for _ in range(self.J):
            current_s = self._phase_a_optimize_resources(snapshots, total_bw_bps, current_eta)
            current_eta = self._phase_b_optimize_configuration(snapshots, total_bw_bps, current_s)
        assert current_s is not None
        allocations = {}
        for task in self.tasks:
            tid = int(task.logical_task_id)
            allocations[tid] = {
                "eta": np.asarray(current_eta[tid], dtype=float),
                "s_comm": np.asarray(current_s[tid]["s_comm"], dtype=float),
                "s_comp": np.asarray(current_s[tid]["s_comp"], dtype=float),
            }
        return allocations

    def update_dual(self, actual_delay_by_task, tasks_by_id):
        for tid, actual_delay in actual_delay_by_task.items():
            task = tasks_by_id[int(tid)]
            self.lambda_k[int(tid)] = max(
                self.epsilon,
                float(self.lambda_k[int(tid)]) + float(actual_delay) - (1.0 / float(task.target_rate_hz)),
            )


class DecoupledEqualSplitStochasticDescentMultiTaskBaseline(BaseCommOnlyPolicy):
    def __init__(self, tasks, mu, epsilon):
        super().__init__("decoupled_equal_split_multi_baseline", "Baseline: decoupled equal split (mu={})".format(mu), tasks)
        self.mu = float(mu)
        self.epsilon = float(epsilon)
        self.lambda_k = {int(task.logical_task_id): float(self.epsilon) for task in self.tasks}

    def current_allocations(self, snapshots, total_bw_bps):
        if not snapshots:
            return EqualShareMultiTaskBaseline(self.tasks).current_allocations(snapshots, total_bw_bps)
        num_tasks = max(len(self.tasks), 1)
        allocations = {}
        for task in self.tasks:
            tid = int(task.logical_task_id)
            snapshot = snapshots[tid]
            s_comp = np.full(NUM_NODES, 1.0 / float(num_tasks), dtype=float)
            s_comm = np.full(task.num_links, 1.0 / float(num_tasks), dtype=float)
            c_eff = np.asarray(total_bw_bps, dtype=float) * np.asarray(s_comm, dtype=float)
            max_comp_delay = max(float(snapshot.tau_list[node_idx]) / max(float(s_comp[node_idx]), 1e-12) for node_idx in range(NUM_NODES))
            eta = _solve_single_task_eta_with_fixed_resources_online(
                task=task,
                snapshot=snapshot,
                c_eff=c_eff,
                max_comp_delay=max_comp_delay,
                mu=self.mu,
                lambda_t=float(self.lambda_k[tid]),
            )
            allocations[tid] = {"eta": eta, "s_comm": s_comm, "s_comp": s_comp}
        return _normalize_allocations_to_unit_sum(self.tasks, allocations)

    def update_dual(self, actual_delay_by_task, tasks_by_id):
        for tid, actual_delay in actual_delay_by_task.items():
            task = tasks_by_id[int(tid)]
            self.lambda_k[int(tid)] = max(
                self.epsilon,
                float(self.lambda_k[int(tid)]) + float(actual_delay) - (1.0 / float(task.target_rate_hz)),
            )


class QueueProportionalHeuristicMultiTaskBaseline(BaseCommOnlyPolicy):
    def __init__(self, tasks, mu, epsilon):
        super().__init__("queue_proportional_multi_baseline", "Baseline: queue proportional (mu={})".format(mu), tasks)
        self.mu = float(mu)
        self.epsilon = float(epsilon)
        self.lambda_k = {int(task.logical_task_id): float(self.epsilon) for task in self.tasks}

    def current_allocations(self, snapshots, total_bw_bps):
        if not snapshots:
            return EqualShareMultiTaskBaseline(self.tasks).current_allocations(snapshots, total_bw_bps)
        allocations = {}
        s_comp_map = {int(task.logical_task_id): np.zeros(NUM_NODES, dtype=float) for task in self.tasks}
        s_comm_map = {int(task.logical_task_id): np.zeros(task.num_links, dtype=float) for task in self.tasks}
        for node_idx in range(NUM_NODES):
            denom = sum(float(self.lambda_k[int(task.logical_task_id)]) for task in self.tasks)
            for task in self.tasks:
                tid = int(task.logical_task_id)
                s_comp_map[tid][node_idx] = (float(self.lambda_k[tid]) / denom) if denom > 0.0 else (1.0 / max(len(self.tasks), 1))
        for link_idx in range(NUM_TRANSFER_POINTS):
            denom = sum(float(self.lambda_k[int(task.logical_task_id)]) for task in self.tasks)
            for task in self.tasks:
                tid = int(task.logical_task_id)
                s_comm_map[tid][link_idx] = (float(self.lambda_k[tid]) / denom) if denom > 0.0 else (1.0 / max(len(self.tasks), 1))
        for task in self.tasks:
            tid = int(task.logical_task_id)
            snapshot = snapshots[tid]
            s_comp = s_comp_map[tid]
            s_comm = s_comm_map[tid]
            c_eff = np.asarray(total_bw_bps, dtype=float) * np.asarray(s_comm, dtype=float)
            max_comp_delay = max(float(snapshot.tau_list[node_idx]) / max(float(s_comp[node_idx]), 1e-12) for node_idx in range(NUM_NODES))
            eta = _solve_single_task_eta_with_fixed_resources_online(
                task=task,
                snapshot=snapshot,
                c_eff=c_eff,
                max_comp_delay=max_comp_delay,
                mu=self.mu,
                lambda_t=float(self.lambda_k[tid]),
            )
            allocations[tid] = {"eta": eta, "s_comm": s_comm, "s_comp": s_comp}
        return _normalize_allocations_to_unit_sum(self.tasks, allocations)

    def update_dual(self, actual_delay_by_task, tasks_by_id):
        for tid, actual_delay in actual_delay_by_task.items():
            task = tasks_by_id[int(tid)]
            self.lambda_k[int(tid)] = max(
                self.epsilon,
                float(self.lambda_k[int(tid)]) + float(actual_delay) - (1.0 / float(task.target_rate_hz)),
            )


def create_policy(policy_cfg, tasks):
    algo_type = str(policy_cfg["type"]).lower()
    if algo_type == "no_csi_multi":
        return NoCSICommOnlyMultiTaskPolicy(
            tasks,
            mu=float(policy_cfg.get("mu", DEFAULT_MU_FALLBACK)),
            epsilon=float(policy_cfg.get("epsilon", DEFAULT_EPSILON)),
            J=int(policy_cfg.get("J", DEFAULT_J)),
        )
    if algo_type == "equal_share_multi_baseline":
        return EqualShareMultiTaskBaseline(tasks)
    if algo_type == "no_compression_multi_baseline":
        return NoCompressionMultiTaskBaseline(tasks)
    if algo_type == "max_compression_multi_baseline":
        return MaxCompressionMultiTaskBaseline(tasks)
    if algo_type == "certainty_equivalence_multi_baseline":
        return HistoricalAverageCertaintyEquivalenceMultiTaskBaseline(tasks)
    if algo_type == "decoupled_equal_split_multi_baseline":
        return DecoupledEqualSplitStochasticDescentMultiTaskBaseline(
            tasks,
            mu=float(policy_cfg.get("mu", DEFAULT_MU_FALLBACK)),
            epsilon=float(policy_cfg.get("epsilon", DEFAULT_EPSILON)),
        )
    if algo_type == "queue_proportional_multi_baseline":
        return QueueProportionalHeuristicMultiTaskBaseline(
            tasks,
            mu=float(policy_cfg.get("mu", DEFAULT_MU_FALLBACK)),
            epsilon=float(policy_cfg.get("epsilon", DEFAULT_EPSILON)),
        )
    raise ValueError("Unsupported algorithm type '{}'".format(algo_type))


def _load_sst2_batches(factory, dataset_path, split, max_samples, prompt_template):
    samples = load_sst2_samples(dataset_path=dataset_path, split=split)
    if max_samples is not None and int(max_samples) > 0:
        samples = samples[: int(max_samples)]
    batches = []
    for idx, sample in enumerate(samples):
        prompt_text = build_flan_prompt(sample.sentence, prompt_template)
        encoded = factory.encode_prompt(prompt_text)
        batches.append(
            {
                "batch_idx": idx,
                "sample_id": sample.sample_id,
                "label": int(sample.label),
                "prompt_text": prompt_text,
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }
        )
    return batches


def _load_llmint8_mapping_entries(base_dir, source_path):
    if not source_path:
        return []
    payload = _load_json(_resolve_path(base_dir, source_path))
    entries = payload.get("policies", [{}])[0].get("llmint8_eta_to_codec_mapping", {}).get("entries", payload.get("entries", []))
    return validate_llmint8_mapping_entries(entries, num_links=NUM_TRANSFER_POINTS)


def _build_execution_plan_for_codec(codec_name, requested_eta, outlier_precision=None, regular_precision=None, llmint8_mapping_entries=None):
    if str(codec_name) == "topk":
        return resolve_codec_execution_plan(codec_name="topk", eta=requested_eta)
    if str(codec_name) == "quantization":
        return resolve_codec_execution_plan(codec_name="quantization", eta=requested_eta)
    return resolve_codec_execution_plan(
        codec_name="llmint8",
        eta=requested_eta,
        outlier_precision=outlier_precision or "fp16",
        regular_precision=regular_precision or "int8",
        llmint8_mapping_entries=llmint8_mapping_entries or [],
    )


def _build_accuracy_estimator(
    task_family,
    estimator_path,
    eta_min,
    accuracy_estimator_mode,
    codec_name,
    device,
    task_cfg,
    batches,
    outlier_precision=None,
    regular_precision=None,
    llmint8_entries=None,
    stein_sigma=DEFAULT_STEIN_SIGMA,
    stein_N=DEFAULT_STEIN_N,
    stein_fast_max_batches=DEFAULT_STEIN_FAST_MAX_BATCHES,
    stein_fast_max_samples=DEFAULT_STEIN_FAST_MAX_SAMPLES,
):
    if str(accuracy_estimator_mode) == "fitting_model":
        return Poly3AccuracyAdapter(estimator_path)
    if str(task_family) == "resnet":
        fast_batches = list(batches[: max(1, int(stein_fast_max_batches))])
        evaluator = ResNetFastAccuracyEvaluator(
            checkpoint_path=str(task_cfg.get("checkpoint_path", RESNET_CHECKPOINT_PATH)),
            device=device,
            batches=fast_batches,
        )

        def accuracy_callable(eta_vec):
            execution_plan = _build_execution_plan_for_codec(
                codec_name=codec_name,
                requested_eta=eta_vec,
                outlier_precision=outlier_precision,
                regular_precision=regular_precision,
                llmint8_mapping_entries=llmint8_entries,
            )
            return float(evaluator.evaluate(execution_plan))

    elif str(task_family) == "flan_t5":
        fast_batches = list(batches[: max(1, int(stein_fast_max_samples))])
        evaluator = FlanT5FastAccuracyEvaluator(
            model_name=str(task_cfg.get("model_name", FLAN_DEFAULT_MODEL_NAME)),
            device=device,
            max_input_length=int(task_cfg.get("max_input_length", FLAN_DEFAULT_MAX_INPUT_LENGTH)),
            positive_token=str(task_cfg.get("positive_token", FLAN_DEFAULT_POSITIVE_TOKEN)),
            negative_token=str(task_cfg.get("negative_token", FLAN_DEFAULT_NEGATIVE_TOKEN)),
            batches=fast_batches,
        )

        def accuracy_callable(eta_vec):
            execution_plan = _build_execution_plan_for_codec(
                codec_name=codec_name,
                requested_eta=eta_vec,
                outlier_precision=outlier_precision,
                regular_precision=regular_precision,
                llmint8_mapping_entries=llmint8_entries,
            )
            return float(evaluator.evaluate(execution_plan))

    else:
        raise ValueError("Unsupported task family '{}'".format(task_family))
    return SteinAccuracyEstimatorAdapter(
        accuracy_callable=accuracy_callable,
        num_links=len(eta_min),
        eta_min=np.asarray(eta_min, dtype=float),
        sigma=float(stein_sigma),
        N=int(stein_N),
    )


def _build_task_defs(
    manifest_dir,
    manifest,
    codec_name,
    device,
    accuracy_estimator_mode="fitting_model",
    stein_sigma=DEFAULT_STEIN_SIGMA,
    stein_N=DEFAULT_STEIN_N,
    stein_fast_max_batches=DEFAULT_STEIN_FAST_MAX_BATCHES,
    stein_fast_max_samples=DEFAULT_STEIN_FAST_MAX_SAMPLES,
):
    tasks = []
    flan_factory_cache = {}
    for task_cfg in manifest["tasks"]:
        task_family = str(task_cfg["family"]).lower()
        profile_cfg = copy.deepcopy(task_cfg["codec_variants"][codec_name])
        estimator_path = _resolve_path(manifest_dir, profile_cfg["accuracy_model"]["path"])
        eta_min = np.asarray(task_cfg.get("eta_min", [0.125, 0.125, 0.125]), dtype=float)
        logical_task_id = int(task_cfg["task_id"])
        weight = float(task_cfg.get("weight", 1.0))
        batch_size = int(task_cfg.get("batch_size", 1 if task_family == "flan_t5" else 100))
        target_rate_hz = float(task_cfg["target_rate_hz"])
        outlier_precision = str(profile_cfg.get("outlier_precision", task_cfg.get("outlier_precision", "fp16"))) if codec_name == "llmint8" else None
        regular_precision = str(profile_cfg.get("regular_precision", task_cfg.get("regular_precision", "int8"))) if codec_name == "llmint8" else None
        llmint8_entries = _load_llmint8_mapping_entries(manifest_dir, profile_cfg.get("llmint8_mapping_source")) if codec_name == "llmint8" else []

        if task_family == "resnet":
            batches = load_cifar10_batches(
                batch_size=batch_size,
                max_batches=int(task_cfg.get("max_items", 100)),
                data_root=str(task_cfg.get("data_root", RESNET_DEFAULT_DATA_ROOT)),
                download=bool(task_cfg.get("download_data", RESNET_DEFAULT_DOWNLOAD_DATA)),
            )
            extra = {"checkpoint_path": str(task_cfg.get("checkpoint_path", RESNET_CHECKPOINT_PATH))}
        elif task_family == "flan_t5":
            factory_key = (
                str(task_cfg.get("model_name", FLAN_DEFAULT_MODEL_NAME)),
                int(task_cfg.get("max_input_length", FLAN_DEFAULT_MAX_INPUT_LENGTH)),
            )
            if factory_key not in flan_factory_cache:
                flan_factory_cache[factory_key] = FlanT5PartitionFactory(
                    model_name=factory_key[0],
                    device=device,
                    max_input_length=factory_key[1],
                )
            flan_factory = flan_factory_cache[factory_key]
            batches = _load_sst2_batches(
                factory=flan_factory,
                dataset_path=_resolve_path(manifest_dir, task_cfg.get("dataset_path")),
                split=str(task_cfg.get("split", FLAN_DEFAULT_SPLIT)),
                max_samples=int(task_cfg.get("max_items", 100)),
                prompt_template=str(task_cfg.get("prompt_template", FLAN_DEFAULT_PROMPT_TEMPLATE)),
            )
            verbalizers = resolve_single_token_verbalizers(
                flan_factory.tokenizer,
                positive_text=str(task_cfg.get("positive_token", FLAN_DEFAULT_POSITIVE_TOKEN)),
                negative_text=str(task_cfg.get("negative_token", FLAN_DEFAULT_NEGATIVE_TOKEN)),
            )
            extra = {
                "model_name": str(task_cfg.get("model_name", FLAN_DEFAULT_MODEL_NAME)),
                "max_input_length": int(task_cfg.get("max_input_length", FLAN_DEFAULT_MAX_INPUT_LENGTH)),
                "decoder_start_token_id": int(flan_factory.decoder_start_token_id),
                "verbalizers": verbalizers,
            }
        else:
            raise ValueError("Unsupported task family '{}'".format(task_family))

        estimator = _build_accuracy_estimator(
            task_family=task_family,
            estimator_path=estimator_path,
            eta_min=eta_min,
            accuracy_estimator_mode=accuracy_estimator_mode,
            codec_name=codec_name,
            device=device,
            task_cfg=task_cfg,
            batches=batches,
            outlier_precision=outlier_precision,
            regular_precision=regular_precision,
            llmint8_entries=llmint8_entries,
            stein_sigma=stein_sigma,
            stein_N=stein_N,
            stein_fast_max_batches=stein_fast_max_batches,
            stein_fast_max_samples=stein_fast_max_samples,
        )

        tasks.append(
            RuntimeTaskDef(
                logical_task_id=logical_task_id,
                family=task_family,
                name=str(task_cfg.get("name", "task_{}".format(logical_task_id))),
                model=str(task_cfg.get("model", task_family)),
                dataset=str(task_cfg.get("dataset", "")),
                weight=weight,
                eta_min=eta_min,
                target_rate_hz=target_rate_hz,
                batch_size=batch_size,
                estimator=estimator,
                codec_name=str(codec_name),
                outlier_precision=outlier_precision,
                regular_precision=regular_precision,
                llmint8_mapping_entries=llmint8_entries,
                warmup_items=list(copy.deepcopy(batches)),
                experiment_items=list(copy.deepcopy(batches)),
                extra=extra,
            )
        )
    return tasks


def _build_worker_task_defs(manifest):
    tasks = []
    for task_cfg in manifest["tasks"]:
        task_family = str(task_cfg["family"]).lower()
        logical_task_id = int(task_cfg["task_id"])
        eta_min = np.asarray(task_cfg.get("eta_min", [0.125, 0.125, 0.125]), dtype=float)
        batch_size = int(task_cfg.get("batch_size", 1 if task_family == "flan_t5" else 100))
        extra = {}
        if task_family == "resnet":
            extra["checkpoint_path"] = str(task_cfg.get("checkpoint_path", RESNET_CHECKPOINT_PATH))
        elif task_family == "flan_t5":
            extra = {
                "model_name": str(task_cfg.get("model_name", FLAN_DEFAULT_MODEL_NAME)),
                "max_input_length": int(task_cfg.get("max_input_length", FLAN_DEFAULT_MAX_INPUT_LENGTH)),
                "decoder_start_token_id": 0,
                "positive_token": str(task_cfg.get("positive_token", FLAN_DEFAULT_POSITIVE_TOKEN)),
                "negative_token": str(task_cfg.get("negative_token", FLAN_DEFAULT_NEGATIVE_TOKEN)),
            }
        else:
            raise ValueError("Unsupported task family '{}'".format(task_family))
        tasks.append(
            RuntimeTaskDef(
                logical_task_id=logical_task_id,
                family=task_family,
                name=str(task_cfg.get("name", "task_{}".format(logical_task_id))),
                model=str(task_cfg.get("model", task_family)),
                dataset=str(task_cfg.get("dataset", "")),
                weight=float(task_cfg.get("weight", 1.0)),
                eta_min=eta_min,
                target_rate_hz=float(task_cfg.get("target_rate_hz", 1.0)),
                batch_size=batch_size,
                estimator=None,
                codec_name="topk",
                outlier_precision=None,
                regular_precision=None,
                llmint8_mapping_entries=[],
                warmup_items=[],
                experiment_items=[],
                extra=extra,
            )
        )
    return tasks


def _build_execution_plan(task_def: RuntimeTaskDef, requested_eta: Sequence[float]):
    return _build_execution_plan_for_codec(
        codec_name=task_def.codec_name,
        requested_eta=requested_eta,
        outlier_precision=task_def.outlier_precision,
        regular_precision=task_def.regular_precision,
        llmint8_mapping_entries=task_def.llmint8_mapping_entries,
    )


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


def _build_handshake_meta(request_id, logical_task_id, link_idx, expected_payload_bytes, stage_msg_type, task_family=None, src_node=None, dst_node=None):
    meta = {
        "request_id": str(request_id),
        "logical_task_id": int(logical_task_id),
        "link_idx": int(link_idx),
        "expected_payload_bytes": int(expected_payload_bytes),
        "stage_msg_type": int(stage_msg_type),
    }
    if task_family is not None:
        meta["task_family"] = str(task_family)
    if src_node is not None:
        meta["src_node"] = str(src_node)
    if dst_node is not None:
        meta["dst_node"] = str(dst_node)
    return meta


def _receiver_side_tp_stats(envelope, node_id, profile=None, request_id=None, logical_task_id=None):
    handshake_meta = dict(envelope.get("handshake_meta", {}) or {})
    handshake_started_at = envelope.get("handshake_started_at", None)
    received_at = envelope.get("received_at", None)
    if (
        handshake_started_at is not None
        and received_at is not None
        and np.isfinite(float(handshake_started_at))
        and np.isfinite(float(received_at))
    ):
        payload_bytes = max(int(handshake_meta.get("expected_payload_bytes", envelope.get("payload_bytes", 0))), 0)
        elapsed_sec = max(float(received_at) - float(handshake_started_at), 1e-9)
    else:
        payload_bytes = max(int(envelope.get("payload_bytes", 0)), 0)
        elapsed_sec = max(float(envelope.get("transfer_elapsed_sec", 0.0)), 1e-9)
    entry = {
        "request_id": request_id,
        "logical_task_id": int(logical_task_id) if logical_task_id is not None else -1,
        "node_id": node_id,
        "tp_stats": {"bytes": int(payload_bytes), "elapsed": float(elapsed_sec)},
    }
    if profile is not None:
        prof = dict(profile)
        prof["send_bytes"] = int(payload_bytes)
        prof["send_sec"] = float(elapsed_sec)
        entry["profile"] = prof
    return entry


def _extract_final_prediction(task_family: str, payload: Dict[str, Any]) -> Tuple[int, float]:
    if task_family == "resnet":
        labels = payload["labels"]
        predicted = torch.argmax(payload["logits"], dim=1)
        correct = float((predicted.cpu() == labels).float().mean().item())
        return int(predicted[0].item()), correct
    predicted = int(payload["predicted_label"])
    correct = float(int(predicted == int(payload["label"])))
    return predicted, correct


def _build_runtime_map(tasks: Sequence[RuntimeTaskDef], node_id: str, device: str):
    runtime_map = {}
    for task in tasks:
        if task.family == "resnet":
            factory = ResNet56PartitionFactory(
                checkpoint_path=str(task.extra.get("checkpoint_path", RESNET_CHECKPOINT_PATH)),
                device=device,
            )
            runtime_map[int(task.logical_task_id)] = {
                "family": "resnet",
                "factory": factory,
                "partition": factory.build_partition_for_node(node_id),
            }
        elif task.family == "flan_t5":
            factory = FlanT5PartitionFactory(
                model_name=str(task.extra.get("model_name", FLAN_DEFAULT_MODEL_NAME)),
                device=device,
                max_input_length=int(task.extra.get("max_input_length", FLAN_DEFAULT_MAX_INPUT_LENGTH)),
            )
            verbalizers = task.extra.get("verbalizers")
            if verbalizers is None:
                verbalizers = resolve_single_token_verbalizers(
                    factory.tokenizer,
                    positive_text=str(task.extra.get("positive_token", FLAN_DEFAULT_POSITIVE_TOKEN)),
                    negative_text=str(task.extra.get("negative_token", FLAN_DEFAULT_NEGATIVE_TOKEN)),
                )
            runtime_map[int(task.logical_task_id)] = {
                "family": "flan_t5",
                "factory": factory,
                "partition": factory.build_partition_for_node(node_id),
                "decoder_start_token_id": int(task.extra.get("decoder_start_token_id", factory.decoder_start_token_id) or factory.decoder_start_token_id),
                "verbalizers": dict(verbalizers),
            }
        else:
            raise ValueError("Unsupported task family '{}'".format(task.family))
    return runtime_map


def _dispatch_resnet_request(sender, runtime, task_def, batch, request_id, execution_plan, link_rate_limits_bps, device):
    partition = runtime["partition"]
    images = batch["images"].to(device)
    started_at = time.perf_counter()
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
        "request_id": request_id,
        "logical_task_id": int(task_def.logical_task_id),
        "task_family": "resnet",
        "batch_idx": int(batch["batch_idx"]),
        "compressor_name": compressor_name,
        "compression_params_list": compression_params_list,
        "feature_k_values": exec_eta,
        "requested_eta": list(execution_plan["requested_eta"]),
        "mapping_info": execution_plan.get("mapping_info"),
        "activation_comp": hidden_comp,
        "tp_original_bytes": [comp_stats["original_bytes"], 0, 0],
        "tp_compressed_bytes": [comp_stats["compressed_bytes"], 0, 0],
        "link_rate_limits_bps": [float(item) for item in link_rate_limits_bps],
        "worker_stats_chain": {},
    }
    prepare_sec = time.perf_counter() - t_prepare
    outgoing["worker_stats_chain"]["A"] = {
        "request_id": request_id,
        "logical_task_id": int(task_def.logical_task_id),
        "node_id": "A",
        "profile": _worker_profile("A", compute_sec, 0.0, prepare_sec, 0, 0.0),
    }
    handshake_meta = _build_handshake_meta(
        request_id=request_id,
        logical_task_id=int(task_def.logical_task_id),
        link_idx=0,
        expected_payload_bytes=len(pickle.dumps(outgoing, protocol=pickle.HIGHEST_PROTOCOL)),
        stage_msg_type=MessageType.TASK_INPUT,
        task_family="resnet",
        src_node="A",
        dst_node="B",
    )
    bytes_sent, send_sec = sender.send(
        "B",
        MessageType.TASK_INPUT,
        outgoing,
        logical_task_id=task_def.logical_task_id,
        link_idx=0,
        rate_bps=float(link_rate_limits_bps[0]),
        handshake_meta=handshake_meta,
    )
    return {
        "request_id": request_id,
        "logical_task_id": int(task_def.logical_task_id),
        "task_family": "resnet",
        "batch_idx": int(batch["batch_idx"]),
        "labels": batch["labels"].clone(),
        "requested_eta": list(execution_plan["requested_eta"]),
        "executed_eta": exec_eta,
        "compression_params_list": compression_params_list,
        "compressor_name": compressor_name,
        "mapping_info": execution_plan.get("mapping_info"),
        "started_at": float(started_at),
        "tp_original_bytes": list(outgoing["tp_original_bytes"]),
        "tp_compressed_bytes": list(outgoing["tp_compressed_bytes"]),
    }


def _dispatch_flan_request(sender, runtime, task_def, batch, request_id, execution_plan, link_rate_limits_bps, device):
    partition = runtime["partition"]
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    started_at = time.perf_counter()
    t_comp = time.perf_counter()
    with torch.no_grad():
        hidden, position_bias = partition(input_ids, attention_mask=attention_mask, position_bias=None)
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
        "request_id": request_id,
        "logical_task_id": int(task_def.logical_task_id),
        "task_family": "flan_t5",
        "batch_idx": int(batch["batch_idx"]),
        "sample_id": batch["sample_id"],
        "prompt_text": batch["prompt_text"],
        "decoder_start_token_id": int(runtime["decoder_start_token_id"]),
        "positive_id": int(runtime["verbalizers"]["positive_id"]),
        "negative_id": int(runtime["verbalizers"]["negative_id"]),
        "compressor_name": compressor_name,
        "compression_params_list": compression_params_list,
        "feature_k_values": exec_eta,
        "requested_eta": list(execution_plan["requested_eta"]),
        "mapping_info": execution_plan.get("mapping_info"),
        "activation_comp": hidden_comp,
        "position_bias": None if position_bias is None else position_bias.detach().cpu(),
        "encoder_attention_mask": attention_mask.detach().cpu(),
        "tp_original_bytes": [comp_stats["original_bytes"], 0, 0],
        "tp_compressed_bytes": [comp_stats["compressed_bytes"], 0, 0],
        "link_rate_limits_bps": [float(item) for item in link_rate_limits_bps],
        "worker_stats_chain": {},
    }
    prepare_sec = time.perf_counter() - t_prepare
    outgoing["worker_stats_chain"]["A"] = {
        "request_id": request_id,
        "logical_task_id": int(task_def.logical_task_id),
        "node_id": "A",
        "profile": _worker_profile("A", compute_sec, 0.0, prepare_sec, 0, 0.0),
    }
    handshake_meta = _build_handshake_meta(
        request_id=request_id,
        logical_task_id=int(task_def.logical_task_id),
        link_idx=0,
        expected_payload_bytes=len(pickle.dumps(outgoing, protocol=pickle.HIGHEST_PROTOCOL)),
        stage_msg_type=MessageType.TASK_INPUT,
        task_family="flan_t5",
        src_node="A",
        dst_node="B",
    )
    bytes_sent, send_sec = sender.send(
        "B",
        MessageType.TASK_INPUT,
        outgoing,
        logical_task_id=task_def.logical_task_id,
        link_idx=0,
        rate_bps=float(link_rate_limits_bps[0]),
        handshake_meta=handshake_meta,
    )
    return {
        "request_id": request_id,
        "logical_task_id": int(task_def.logical_task_id),
        "task_family": "flan_t5",
        "batch_idx": int(batch["batch_idx"]),
        "sample_id": batch["sample_id"],
        "label": int(batch["label"]),
        "requested_eta": list(execution_plan["requested_eta"]),
        "executed_eta": exec_eta,
        "compression_params_list": compression_params_list,
        "compressor_name": compressor_name,
        "mapping_info": execution_plan.get("mapping_info"),
        "started_at": float(started_at),
        "tp_original_bytes": list(outgoing["tp_original_bytes"]),
        "tp_compressed_bytes": list(outgoing["tp_compressed_bytes"]),
    }


def _dispatch_async_request(sender, runtime_map, tasks_by_id, batch, logical_task_id, allocation, device, request_id=None):
    task_def = tasks_by_id[int(logical_task_id)]
    execution_plan = _build_execution_plan(task_def, allocation["eta"])
    total_bw = np.asarray(allocation["total_bw_bps"], dtype=float)
    s_comm = np.asarray(allocation["s_comm"], dtype=float)
    link_rate_limits_bps = np.maximum(total_bw * s_comm, 1.0)
    if request_id is None:
        request_id = str(uuid.uuid4())
    runtime = runtime_map[int(logical_task_id)]
    if task_def.family == "resnet":
        return _dispatch_resnet_request(
            sender,
            runtime,
            task_def,
            batch,
            request_id,
            execution_plan,
            link_rate_limits_bps,
            device,
        )
    return _dispatch_flan_request(
        sender,
        runtime,
        task_def,
        batch,
        request_id,
        execution_plan,
        link_rate_limits_bps,
        device,
    )


def _finalize_request(request_id, result_payload, pending):
    if request_id not in pending:
        return None
    pending_entry = pending.pop(request_id)
    pending_entry["worker_stats_chain"] = dict(result_payload.get("worker_stats_chain", {}))
    if "tp_original_bytes" in result_payload:
        pending_entry["tp_original_bytes"] = list(result_payload["tp_original_bytes"])
    if "tp_compressed_bytes" in result_payload:
        pending_entry["tp_compressed_bytes"] = list(result_payload["tp_compressed_bytes"])
    if "requested_eta" in result_payload:
        pending_entry["requested_eta"] = list(result_payload["requested_eta"])
    if "executed_eta" in result_payload:
        pending_entry["executed_eta"] = list(result_payload["executed_eta"])
    if "compressor_name" in result_payload:
        pending_entry["compressor_name"] = result_payload["compressor_name"]
    return pending_entry


def _build_snapshot_from_completion(meta, aggregate_stats):
    tp_original_bytes = np.asarray(meta["tp_original_bytes"], dtype=float)
    tp_compressed_bytes = np.asarray(meta["tp_compressed_bytes"], dtype=float)
    worker_stats_chain = dict(meta.get("worker_stats_chain", {}))
    stats_a = worker_stats_chain.get("A") or {}
    stats_b = worker_stats_chain.get("B") or {}
    stats_c = worker_stats_chain.get("C") or {}
    stats_d = worker_stats_chain.get("D") or {}
    tp_stats = {
        "A": dict(stats_a.get("tp_stats", {})),
        "B": dict(stats_b.get("tp_stats", {})),
        "C": dict(stats_c.get("tp_stats", {})),
    }
    profiles = {
        "A": dict(stats_a.get("profile", {})),
        "B": dict(stats_b.get("profile", {})),
        "C": dict(stats_c.get("profile", {})),
        "D": dict(stats_d.get("profile", {})),
    }
    end_to_end_delay_sec = float(time.perf_counter() - float(meta["started_at"]))
    tp_bandwidth_bps = []
    fixed_overhead_bytes = []
    link_elapsed_values = []
    for node_id in ["A", "B", "C"]:
        stats = tp_stats.get(node_id, {})
        bytes_sent = max(float(stats.get("bytes", 0.0)), 0.0)
        elapsed = max(float(stats.get("elapsed", 0.0)), 1e-9)
        tp_bandwidth_bps.append((bytes_sent * 8.0) / elapsed)
        link_elapsed_values.append(float(stats.get("elapsed", 0.0)))
    for link_idx, node_id in enumerate(["A", "B", "C"]):
        stats = tp_stats.get(node_id, {})
        bytes_sent = max(float(stats.get("bytes", 0.0)), 0.0)
        compressed_bytes = max(float(tp_compressed_bytes[link_idx]), 0.0)
        fixed_overhead_bytes.append(max(0.0, bytes_sent - compressed_bytes))
    tau_list = []
    for node_id in NODE_ORDER:
        profile = profiles.get(node_id) or {}
        tau_list.append(float(profile.get("service_total_sec", 0.0)))
    service_delay_sec = float(np.max(np.asarray(tau_list, dtype=float))) if tau_list else 0.0
    communication_delay_sec = (
        float(np.max(np.asarray(link_elapsed_values, dtype=float))) if link_elapsed_values else 0.0
    )
    total_delay_sec = max(float(service_delay_sec), float(communication_delay_sec))
    miscellaneous_delay_sec = 0.0
    task_stats = aggregate_stats[int(meta["logical_task_id"])]
    return LatestSnapshot(
        a_ref=np.asarray(tp_original_bytes, dtype=float),
        fixed_overhead_bytes=np.asarray(fixed_overhead_bytes, dtype=float),
        tau_list=np.asarray(tau_list, dtype=float),
        total_delay_sec=total_delay_sec,
        end_to_end_delay_sec=end_to_end_delay_sec,
        tp_bandwidth_bps=np.asarray(tp_bandwidth_bps, dtype=float),
        sample_correct=float(task_stats["last_correct"]),
        running_accuracy=float(task_stats["correct"]) / max(float(task_stats["seen"]), 1.0),
        service_delay_sec=service_delay_sec,
        communication_delay_sec=communication_delay_sec,
        miscellaneous_delay_sec=miscellaneous_delay_sec,
    )


def _mean_snapshot_from_list(snapshot_list: Sequence[LatestSnapshot]) -> LatestSnapshot:
    if not snapshot_list:
        raise ValueError("Expected non-empty snapshot_list")

    def _mean_array(attr_name):
        return np.mean(
            np.stack([np.asarray(getattr(item, attr_name), dtype=float) for item in snapshot_list], axis=0),
            axis=0,
        )

    def _mean_scalar(attr_name):
        return float(np.mean(np.asarray([getattr(item, attr_name) for item in snapshot_list], dtype=float)))

    return LatestSnapshot(
        a_ref=np.asarray(_mean_array("a_ref"), dtype=float),
        fixed_overhead_bytes=np.asarray(_mean_array("fixed_overhead_bytes"), dtype=float),
        tau_list=np.asarray(_mean_array("tau_list"), dtype=float),
        total_delay_sec=_mean_scalar("total_delay_sec"),
        end_to_end_delay_sec=_mean_scalar("end_to_end_delay_sec"),
        tp_bandwidth_bps=np.asarray(_mean_array("tp_bandwidth_bps"), dtype=float),
        sample_correct=_mean_scalar("sample_correct"),
        running_accuracy=_mean_scalar("running_accuracy"),
        service_delay_sec=_mean_scalar("service_delay_sec"),
        communication_delay_sec=_mean_scalar("communication_delay_sec"),
        miscellaneous_delay_sec=_mean_scalar("miscellaneous_delay_sec"),
    )


def _aggregate_control_window_snapshots(control_records, tasks, latest_snapshots):
    grouped = {int(task.logical_task_id): [] for task in tasks}
    for item in control_records:
        grouped[int(item["logical_task_id"])].append(item["snapshot"])
    averaged = {}
    for task in tasks:
        tid = int(task.logical_task_id)
        snapshots_for_task = grouped.get(tid, [])
        if snapshots_for_task:
            averaged[tid] = _mean_snapshot_from_list(snapshots_for_task)
        elif tid in latest_snapshots:
            averaged[tid] = latest_snapshots[tid]
    return averaged


def _aggregate_control_window_total_bandwidth_bps(control_records, tasks, latest_snapshots):
    grouped = {int(task.logical_task_id): [] for task in tasks}
    for item in control_records:
        grouped[int(item["logical_task_id"])].append(np.asarray(item["snapshot"].tp_bandwidth_bps, dtype=float))
    total_bw_bps = np.zeros(NUM_TRANSFER_POINTS, dtype=float)
    for task in tasks:
        tid = int(task.logical_task_id)
        bandwidth_rows = grouped.get(tid, [])
        if bandwidth_rows:
            task_mean_bw_bps = np.mean(np.stack(bandwidth_rows, axis=0), axis=0)
        elif tid in latest_snapshots:
            task_mean_bw_bps = np.asarray(latest_snapshots[tid].tp_bandwidth_bps, dtype=float)
        else:
            task_mean_bw_bps = np.zeros(NUM_TRANSFER_POINTS, dtype=float)
        total_bw_bps += np.asarray(task_mean_bw_bps, dtype=float)
    return np.asarray(total_bw_bps, dtype=float)


def _tp_stats_from_total_bandwidth_bps(total_bw_bps):
    return [(float(bps) / 8.0, 1.0) for bps in np.asarray(total_bw_bps, dtype=float)]


def _predict_total_bandwidth_bps_from_estimator(channel_estimator, fallback_total_bw_bps):
    if channel_estimator is None:
        return np.asarray(fallback_total_bw_bps, dtype=float)
    try:
        has_history = any(len(link.history) > 0 for link in channel_estimator.links)
    except Exception:
        has_history = False
    if not has_history:
        return np.asarray(fallback_total_bw_bps, dtype=float)
    predicted_bytes_per_sec = np.asarray(channel_estimator.predict_all(), dtype=float)
    return predicted_bytes_per_sec * 8.0


def _solve_multi_policy_snapshot(policy, averaged_snapshots, total_bw_bps, tasks_by_id, window_id):
    started = time.perf_counter()
    policy_copy = copy.deepcopy(policy)
    actual_delay_by_task = {
        int(tid): float(snapshot.total_delay_sec)
        for tid, snapshot in averaged_snapshots.items()
    }
    policy_copy.update_dual(actual_delay_by_task, tasks_by_id)
    allocations = policy_copy.current_allocations(averaged_snapshots, np.asarray(total_bw_bps, dtype=float))
    predicted_accuracy_by_task, predicted_delay_by_task = _compute_task_predictions(
        tasks_by_id,
        allocations,
        averaged_snapshots,
        total_bw_bps,
        enable_accuracy_prediction=_policy_uses_estimator(policy=policy_copy, algorithm_type=getattr(policy_copy, "policy_key", None)),
    )
    return {
        "window_id": int(window_id),
        "allocations": {
            int(tid): {
                "eta": np.asarray(alloc["eta"], dtype=float),
                "s_comm": np.asarray(alloc["s_comm"], dtype=float),
                "s_comp": np.asarray(alloc["s_comp"], dtype=float),
            }
            for tid, alloc in allocations.items()
        },
        "lambda_k": copy.deepcopy(getattr(policy_copy, "lambda_k", {})),
        "actual_delay_by_task": actual_delay_by_task,
        "control_solve_sec": float(time.perf_counter() - started),
        "total_bw_bps": np.asarray(total_bw_bps, dtype=float),
        "averaged_snapshots": averaged_snapshots,
        "predicted_accuracy_by_task": dict(predicted_accuracy_by_task),
        "predicted_delay_by_task": dict(predicted_delay_by_task),
    }


def _clone_allocations(allocations):
    return {
        int(tid): {
            "eta": np.asarray(alloc["eta"], dtype=float).copy(),
            "s_comm": np.asarray(alloc["s_comm"], dtype=float).copy(),
            "s_comp": np.asarray(alloc["s_comp"], dtype=float).copy(),
        }
        for tid, alloc in allocations.items()
    }


def _accuracy_summary_from_rows(rows):
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {}
    metric_column = "accuracy_true" if "accuracy_true" in frame.columns else "weighted_accuracy_true"
    grouped = frame.groupby("algorithm")[metric_column].mean().sort_values(ascending=False)
    best_algorithm = str(grouped.index[0])
    return {
        "policy_count": int(grouped.shape[0]),
        "mean_accuracy_across_policies": float(grouped.mean()),
        "best_accuracy_policy": best_algorithm,
        "best_avg_weighted_accuracy_true": float(grouped.iloc[0]),
    }


def _finite_mean(values):
    finite_values = [float(value) for value in values if not pd.isna(value)]
    if not finite_values:
        return np.nan
    return float(np.mean(np.asarray(finite_values, dtype=float)))


def _completion_rows_to_batch_timeseries_for_task(
    completion_rows,
    task_def,
    codec_name,
    dynamic_timeslot_size=None,
    algorithm_name=None,
    algorithm_type=None,
):
    task_id = int(task_def.logical_task_id)
    target_rate_hz = float(task_def.target_rate_hz)
    target_time_sec = (1.0 / target_rate_hz) if target_rate_hz > 0.0 else np.nan
    task_rows = [
        dict(row)
        for row in completion_rows
        if int(row.get("logical_task_id", -1)) == int(task_id)
    ]
    timeseries_rows = []
    for idx, row in enumerate(task_rows):
        requested_eta = [float(x) for x in row.get("requested_eta", [])]
        executed_eta = [float(x) for x in row.get("executed_eta", requested_eta)]
        pred_delay = row.get("selected_delay_est", np.nan)
        batch_row = {
            "t": int(idx),
            "logical_task_id": int(task_id),
            "series_scope": "task{}".format(int(task_id)),
            "algorithm": str(algorithm_name if algorithm_name is not None else row.get("algorithm", "")),
            "policy": str(algorithm_name if algorithm_name is not None else row.get("policy", row.get("algorithm", ""))),
            "policy_key": (algorithm_type if algorithm_type is not None else row.get("policy_key")),
            "algorithm_type": (algorithm_type if algorithm_type is not None else row.get("policy_key")),
            "codec_name": str(codec_name),
            "mu": float(row["mu"]) if row.get("mu") is not None and not pd.isna(row.get("mu")) else np.nan,
            "epsilon": float(row["epsilon"]) if row.get("epsilon") is not None and not pd.isna(row.get("epsilon")) else np.nan,
            "target_rate_hz": float(target_rate_hz) if target_rate_hz > 0.0 else np.nan,
            "target_time_sec": float(target_time_sec) if not pd.isna(target_time_sec) else np.nan,
            "accuracy_true": float(row.get("running_accuracy", np.nan)),
            "accuracy_fit": float(row["selected_accuracy_est"]) if row.get("selected_accuracy_est") is not None and not pd.isna(row.get("selected_accuracy_est")) else np.nan,
            "delay_actual_sec": float(row.get("delay_sec", np.nan)),
            "delay_pred_sec": float(pred_delay) if pred_delay is not None and not pd.isna(pred_delay) else np.nan,
            "delay_ratio_actual": (
                float(row.get("delay_sec", np.nan)) * float(target_rate_hz)
                if target_rate_hz > 0.0 and not pd.isna(row.get("delay_sec", np.nan))
                else np.nan
            ),
            "delay_ratio_pred": (
                float(pred_delay) * float(target_rate_hz)
                if target_rate_hz > 0.0 and pred_delay is not None and not pd.isna(pred_delay)
                else np.nan
            ),
            "lambda_value": float(row.get("lambda_value", np.nan)) if row.get("lambda_value") is not None else np.nan,
            "accuracy": float(row.get("running_accuracy", np.nan)),
            "delay_sec": float(row.get("delay_sec", np.nan)),
            "lambda": float(row.get("lambda_value", np.nan)) if row.get("lambda_value") is not None else np.nan,
            "sample_correct": float(row.get("sample_correct", np.nan)),
            "correct_count": int(row.get("correct_count", 0)),
            "sample_total": int(row.get("sample_total", 0)),
            "policy_version": int(row.get("policy_version", 0)),
            "window_id": int(row.get("window_id", 0)),
            "batch_idx": int(row.get("batch_idx", -1)),
            "end_to_end_delay_sec": float(row.get("end_to_end_delay_sec", np.nan)),
            "communication_delay_sec": float(row.get("communication_delay_sec", np.nan)),
            "service_delay_sec": float(row.get("service_delay_sec", np.nan)),
            "compressor_name": row.get("compressor_name"),
            "requested_eta": ",".join(_format_float4(x) for x in requested_eta),
            "executed_eta": ",".join(_format_float4(x) for x in executed_eta),
        }
        if dynamic_timeslot_size is not None:
            batch_row["dynamic_timeslot_size"] = int(dynamic_timeslot_size)
        for eta_idx, value in enumerate(requested_eta):
            batch_row["requested_eta_{}".format(eta_idx)] = float(value)
            batch_row["eta_{}".format(eta_idx)] = float(value)
        for eta_idx, value in enumerate(executed_eta):
            batch_row["executed_eta_{}".format(eta_idx)] = float(value)
        timeseries_rows.append(batch_row)
    return timeseries_rows


def _average_numeric(values):
    finite = [float(value) for value in values if value is not None and not pd.isna(value)]
    if not finite:
        return np.nan
    return float(np.mean(np.asarray(finite, dtype=float)))


def _task_weight_map(tasks):
    weights = {int(task.logical_task_id): max(float(task.weight), 0.0) for task in tasks}
    total = float(sum(weights.values()))
    if total <= 1e-12:
        if not weights:
            return {}
        uniform = 1.0 / float(len(weights))
        return {int(task_id): float(uniform) for task_id in weights.keys()}
    return {int(task_id): float(value) / total for task_id, value in weights.items()}


def _weighted_average_by_task(value_by_task, task_weights):
    valid = []
    for task_id, value in value_by_task.items():
        if value is None or pd.isna(value):
            continue
        weight = float(task_weights.get(int(task_id), 0.0))
        if weight <= 0.0:
            continue
        valid.append((weight, float(value)))
    if not valid:
        return np.nan
    denom = float(sum(weight for weight, _ in valid))
    if denom <= 1e-12:
        return np.nan
    return float(sum(weight * value for weight, value in valid) / denom)


def _weighted_average_from_rows_by_task(rows_by_task, field_name, task_weights):
    return _weighted_average_by_task(
        {
            int(task_id): row.get(field_name, np.nan)
            for task_id, row in rows_by_task.items()
        },
        task_weights,
    )


def _task_scope_rows(summary_by_scope):
    rows_by_task = {}
    for scope_key, row in summary_by_scope.items():
        if row is None:
            continue
        scope_text = str(scope_key)
        if not scope_text.startswith("task"):
            continue
        try:
            task_id = int(scope_text.replace("task", ""))
        except Exception:
            continue
        rows_by_task[int(task_id)] = dict(row)
    return rows_by_task


def _apply_weighted_non_accuracy_timeseries_fields(avg_row, rows_by_task, task_weights):
    weighted_fields = [
        "target_rate_hz",
        "target_time_sec",
        "delay_actual_sec",
        "delay_pred_sec",
        "delay_ratio_actual",
        "delay_ratio_pred",
        "lambda_value",
        "delay_sec",
        "lambda",
        "end_to_end_delay_sec",
        "communication_delay_sec",
        "service_delay_sec",
    ]
    for field_name in weighted_fields:
        avg_row[field_name] = _weighted_average_from_rows_by_task(rows_by_task, field_name, task_weights)
    for eta_idx in range(NUM_TRANSFER_POINTS):
        req_value = _weighted_average_from_rows_by_task(rows_by_task, "requested_eta_{}".format(eta_idx), task_weights)
        exec_value = _weighted_average_from_rows_by_task(rows_by_task, "executed_eta_{}".format(eta_idx), task_weights)
        avg_row["requested_eta_{}".format(eta_idx)] = req_value
        avg_row["eta_{}".format(eta_idx)] = req_value
        avg_row["executed_eta_{}".format(eta_idx)] = exec_value
    avg_row["requested_eta"] = ",".join(
        _format_float4(avg_row.get("requested_eta_{}".format(eta_idx), np.nan))
        for eta_idx in range(NUM_TRANSFER_POINTS)
    )
    avg_row["executed_eta"] = ",".join(
        _format_float4(avg_row.get("executed_eta_{}".format(eta_idx), np.nan))
        for eta_idx in range(NUM_TRANSFER_POINTS)
    )
    return avg_row


def _apply_weighted_non_accuracy_summary_fields(avg_row, rows_by_task, task_weights):
    weighted_fields = [
        "target_rate_hz",
        "target_time_sec",
        "avg_delay",
        "avg_excess_delay",
        "avg_delay_ratio",
        "avg_delay_actual_sec",
        "avg_delay_pred_sec",
        "avg_excess_delay_actual_sec",
        "avg_excess_delay_pred_sec",
        "avg_delay_ratio_actual",
        "avg_delay_ratio_pred",
        "avg_delay_sec",
        "violation_rate",
        "final_lambda",
        "lambda_input",
        "solver_elapsed_sec",
        "predicted_delay_sec",
    ]
    for field_name in weighted_fields:
        if any(field_name in row for row in rows_by_task.values()):
            avg_row[field_name] = _weighted_average_from_rows_by_task(rows_by_task, field_name, task_weights)
    return avg_row


def _combine_summary_avg_from_task_scopes(summary_by_scope, tasks, existing_avg_row=None):
    rows_by_task = _task_scope_rows(summary_by_scope)
    if not rows_by_task:
        return existing_avg_row
    avg_row = dict(existing_avg_row) if existing_avg_row is not None else dict(next(iter(rows_by_task.values())))
    return _apply_weighted_non_accuracy_summary_fields(avg_row, rows_by_task, _task_weight_map(tasks))


def _combine_batch_timeseries_avg(batch_rows_by_task, tasks):
    if not batch_rows_by_task:
        return []
    task_weights = _task_weight_map(tasks)
    ordered_task_ids = sorted(int(task_id) for task_id in batch_rows_by_task.keys())
    aligned_len = min((len(rows) for rows in batch_rows_by_task.values()), default=0)
    avg_rows = []
    for idx in range(aligned_len):
        rows_by_task = {
            int(task_id): dict(batch_rows_by_task[task_id][idx])
            for task_id in ordered_task_ids
        }
        source_rows = list(rows_by_task.values())
        if not source_rows:
            continue
        total_samples = int(sum(int(row.get("sample_total", 0)) for row in source_rows))
        total_correct = int(sum(int(row.get("correct_count", 0)) for row in source_rows))
        avg_row = {
            "t": int(idx),
            "logical_task_id": -1,
            "series_scope": "avg",
            "algorithm": str(source_rows[0].get("algorithm", "")),
            "policy": str(source_rows[0].get("policy", source_rows[0].get("algorithm", ""))),
            "policy_key": source_rows[0].get("policy_key"),
            "algorithm_type": source_rows[0].get("algorithm_type", source_rows[0].get("policy_key")),
            "codec_name": source_rows[0].get("codec_name"),
            "mu": _average_numeric(row.get("mu", np.nan) for row in source_rows),
            "epsilon": _average_numeric(row.get("epsilon", np.nan) for row in source_rows),
            "target_rate_hz": np.nan,
            "target_time_sec": np.nan,
            "accuracy_true": (
                float(total_correct) / float(total_samples)
                if total_samples > 0 else _average_numeric(row.get("accuracy_true", np.nan) for row in source_rows)
            ),
            "accuracy_fit": _average_numeric(row.get("accuracy_fit", np.nan) for row in source_rows),
            "delay_actual_sec": np.nan,
            "delay_pred_sec": np.nan,
            "delay_ratio_actual": np.nan,
            "delay_ratio_pred": np.nan,
            "lambda_value": np.nan,
            "accuracy": (
                float(total_correct) / float(total_samples)
                if total_samples > 0 else _average_numeric(row.get("accuracy", np.nan) for row in source_rows)
            ),
            "delay_sec": np.nan,
            "lambda": np.nan,
            "sample_correct": _average_numeric(row.get("sample_correct", np.nan) for row in source_rows),
            "correct_count": int(total_correct),
            "sample_total": int(total_samples),
            "policy_version": int(max(int(row.get("policy_version", 0)) for row in source_rows)),
            "window_id": int(max(int(row.get("window_id", 0)) for row in source_rows)),
            "batch_idx": int(idx),
            "end_to_end_delay_sec": np.nan,
            "communication_delay_sec": np.nan,
            "service_delay_sec": np.nan,
            "compressor_name": source_rows[0].get("compressor_name"),
            "requested_eta": "",
            "executed_eta": "",
        }
        if "dynamic_timeslot_size" in source_rows[0] and not pd.isna(source_rows[0].get("dynamic_timeslot_size")):
            avg_row["dynamic_timeslot_size"] = int(source_rows[0]["dynamic_timeslot_size"])
        avg_row = _apply_weighted_non_accuracy_timeseries_fields(avg_row, rows_by_task, task_weights)
        avg_rows.append(avg_row)
    return avg_rows


def _build_batch_timeseries_sets(
    completion_rows,
    tasks,
    codec_name,
    dynamic_timeslot_size=None,
    algorithm_name=None,
    algorithm_type=None,
):
    batch_rows_by_scope = {}
    batch_rows_by_task = {}
    for task in tasks:
        task_rows = _completion_rows_to_batch_timeseries_for_task(
            completion_rows,
            task,
            codec_name,
            dynamic_timeslot_size=dynamic_timeslot_size,
            algorithm_name=algorithm_name,
            algorithm_type=algorithm_type,
        )
        scope_key = "task{}".format(int(task.logical_task_id))
        batch_rows_by_scope[scope_key] = task_rows
        batch_rows_by_task[int(task.logical_task_id)] = task_rows
    batch_rows_by_scope["avg"] = _combine_batch_timeseries_avg(batch_rows_by_task, tasks)
    return batch_rows_by_scope


def _batch_timeseries_to_window_summaries(batch_rows, dynamic_timeslot_size, anchor_task_id):
    if not batch_rows:
        return []
    window_size = max(int(dynamic_timeslot_size or 1), 1)
    window_rows = []
    for window_idx, start in enumerate(range(0, len(batch_rows), window_size), start=1):
        chunk = list(batch_rows[start:start + window_size])
        if not chunk:
            continue
        total_samples = int(sum(int(item.get("sample_total", 0)) for item in chunk))
        total_correct = int(sum(int(item.get("correct_count", 0)) for item in chunk))
        avg_delay_sec = _finite_mean(item.get("delay_actual_sec", np.nan) for item in chunk)
        target_time_sec = float(chunk[-1].get("target_time_sec", np.nan))
        avg_accuracy = (
            float(total_correct) / float(total_samples)
            if total_samples > 0 else np.nan
        )
        first_row = dict(chunk[0])
        last_row = dict(chunk[-1])
        summary_row = {
            "window_id": int(window_idx),
            "dynamic_timeslot_size": int(window_size),
            "actual_window_size": int(len(chunk)),
            "anchor_task_id": int(anchor_task_id),
            "anchor_completion_count": int(len(chunk)),
            "sample_count": int(len(chunk)),
            "batch_count": int(len(chunk)),
            "window_total_samples": int(total_samples),
            "window_correct_samples": int(total_correct),
            "start_t_index": int(chunk[0].get("t", 0)),
            "end_t_index": int(chunk[-1].get("t", 0)),
            "start_batch_idx": int(chunk[0].get("batch_idx", -1)),
            "end_batch_idx": int(chunk[-1].get("batch_idx", -1)),
            "algorithm": str(first_row.get("algorithm", "")),
            "policy": str(first_row.get("policy", first_row.get("algorithm", ""))),
            "policy_key": first_row.get("policy_key"),
            "algorithm_type": first_row.get("algorithm_type", first_row.get("policy_key")),
            "codec_name": first_row.get("codec_name"),
            "mu": float(first_row["mu"]) if first_row.get("mu") is not None and not pd.isna(first_row.get("mu")) else np.nan,
            "epsilon": float(first_row["epsilon"]) if first_row.get("epsilon") is not None and not pd.isna(first_row.get("epsilon")) else np.nan,
            "avg_accuracy": float(avg_accuracy) if not pd.isna(avg_accuracy) else np.nan,
            "avg_delay_sec": float(avg_delay_sec) if not pd.isna(avg_delay_sec) else np.nan,
            "target_time_sec": float(target_time_sec) if not pd.isna(target_time_sec) else np.nan,
            "avg_excess_delay_sec": (
                float(avg_delay_sec - target_time_sec)
                if not pd.isna(avg_delay_sec) and not pd.isna(target_time_sec)
                else np.nan
            ),
            "lambda_input": float(last_row.get("lambda_value", np.nan)) if last_row.get("lambda_value") is not None else np.nan,
            "solver_elapsed_sec": np.nan,
            "predicted_accuracy": _finite_mean(item.get("accuracy_fit", np.nan) for item in chunk),
            "predicted_delay_sec": _finite_mean(item.get("delay_pred_sec", np.nan) for item in chunk),
            "policy_version": int(last_row.get("policy_version", 0)),
        }
        for eta_idx in range(NUM_TRANSFER_POINTS):
            eta_value = last_row.get("requested_eta_{}".format(eta_idx), np.nan)
            if not pd.isna(eta_value):
                summary_row["requested_eta_{}".format(eta_idx)] = float(eta_value)
        window_rows.append(summary_row)
    return window_rows


def _combine_window_summaries_avg(window_rows_by_scope, tasks, anchor_task_id):
    task_weights = _task_weight_map(tasks)
    task_window_rows = {
        int(task.logical_task_id): list(window_rows_by_scope.get("task{}".format(int(task.logical_task_id)), []))
        for task in tasks
    }
    max_len = max((len(rows) for rows in task_window_rows.values()), default=0)
    avg_rows = []
    for idx in range(max_len):
        rows_by_task = {
            int(task_id): dict(rows[idx])
            for task_id, rows in task_window_rows.items()
            if idx < len(rows)
        }
        if not rows_by_task:
            continue
        source_rows = list(rows_by_task.values())
        total_samples = int(sum(int(row.get("window_total_samples", 0)) for row in source_rows))
        total_correct = int(sum(int(row.get("window_correct_samples", 0)) for row in source_rows))
        first_row = dict(source_rows[0])
        avg_row = {
            "window_id": int(max(int(row.get("window_id", idx + 1)) for row in source_rows)),
            "dynamic_timeslot_size": int(first_row.get("dynamic_timeslot_size", 1)),
            "actual_window_size": int(sum(int(row.get("actual_window_size", 0)) for row in source_rows)),
            "anchor_task_id": int(anchor_task_id),
            "anchor_completion_count": int(sum(int(row.get("anchor_completion_count", 0)) for row in source_rows)),
            "sample_count": int(sum(int(row.get("sample_count", 0)) for row in source_rows)),
            "batch_count": int(sum(int(row.get("batch_count", 0)) for row in source_rows)),
            "window_total_samples": int(total_samples),
            "window_correct_samples": int(total_correct),
            "start_t_index": int(min(int(row.get("start_t_index", 0)) for row in source_rows)),
            "end_t_index": int(max(int(row.get("end_t_index", 0)) for row in source_rows)),
            "start_batch_idx": int(min(int(row.get("start_batch_idx", -1)) for row in source_rows)),
            "end_batch_idx": int(max(int(row.get("end_batch_idx", -1)) for row in source_rows)),
            "algorithm": str(first_row.get("algorithm", "")),
            "policy": str(first_row.get("policy", first_row.get("algorithm", ""))),
            "policy_key": first_row.get("policy_key"),
            "algorithm_type": first_row.get("algorithm_type", first_row.get("policy_key")),
            "codec_name": first_row.get("codec_name"),
            "mu": float(first_row["mu"]) if first_row.get("mu") is not None and not pd.isna(first_row.get("mu")) else np.nan,
            "epsilon": float(first_row["epsilon"]) if first_row.get("epsilon") is not None and not pd.isna(first_row.get("epsilon")) else np.nan,
            "avg_accuracy": (
                float(total_correct) / float(total_samples)
                if total_samples > 0 else np.nan
            ),
            "avg_delay_sec": np.nan,
            "target_time_sec": np.nan,
            "avg_excess_delay_sec": np.nan,
            "lambda_input": np.nan,
            "solver_elapsed_sec": np.nan,
            "predicted_accuracy": _average_numeric(row.get("predicted_accuracy", np.nan) for row in source_rows),
            "predicted_delay_sec": np.nan,
            "policy_version": int(max(int(row.get("policy_version", 0)) for row in source_rows)),
            "series_scope": "avg",
        }
        for eta_idx in range(NUM_TRANSFER_POINTS):
            avg_row["requested_eta_{}".format(eta_idx)] = _weighted_average_from_rows_by_task(
                rows_by_task,
                "requested_eta_{}".format(eta_idx),
                task_weights,
            )
        avg_row = _apply_weighted_non_accuracy_summary_fields(avg_row, rows_by_task, task_weights)
        avg_rows.append(avg_row)
    return avg_rows


def _build_window_summaries_sets(batch_rows_by_scope, dynamic_timeslot_size, anchor_task_id, tasks):
    window_rows_by_scope = {}
    for task in tasks:
        scope_key = "task{}".format(int(task.logical_task_id))
        rows = list(batch_rows_by_scope.get(scope_key, []))
        window_rows = _batch_timeseries_to_window_summaries(
            rows,
            dynamic_timeslot_size=dynamic_timeslot_size,
            anchor_task_id=int(task.logical_task_id),
        )
        for row in window_rows:
            row["series_scope"] = str(scope_key)
        window_rows_by_scope[str(scope_key)] = window_rows
    avg_rows = _combine_window_summaries_avg(window_rows_by_scope, tasks, anchor_task_id)
    for row in avg_rows:
        row["series_scope"] = "avg"
    window_rows_by_scope["avg"] = avg_rows
    return window_rows_by_scope


def _build_policy_summary_row(batch_rows, window_summaries):
    if not batch_rows:
        return None
    first_row = dict(batch_rows[0])
    last_row = dict(batch_rows[-1])
    last_window = dict(window_summaries[-1]) if window_summaries else {}
    target_rate_hz = float(first_row.get("target_rate_hz", np.nan))
    avg_delay_thisslot = float(last_window.get("avg_delay_sec", np.nan))
    acc_thisslot = float(last_window.get("avg_accuracy", np.nan))
    summary_row = {
        **({"dynamic_timeslot_size": int(first_row["dynamic_timeslot_size"])} if "dynamic_timeslot_size" in first_row and not pd.isna(first_row.get("dynamic_timeslot_size")) else {}),
        **({"target_ratio": float(first_row["target_ratio"])} if "target_ratio" in first_row and not pd.isna(first_row.get("target_ratio")) else {}),
        **({"target_label": str(first_row["target_label"])} if "target_label" in first_row else {}),
        **({"accuracy_estimator_mode": str(first_row["accuracy_estimator_mode"])} if "accuracy_estimator_mode" in first_row else {}),
        "algorithm": str(first_row.get("algorithm", "")),
        "policy": str(first_row.get("policy", first_row.get("algorithm", ""))),
        "policy_key": first_row.get("policy_key"),
        "algorithm_type": first_row.get("algorithm_type", first_row.get("policy_key")),
        "mu": float(first_row["mu"]) if first_row.get("mu") is not None and not pd.isna(first_row.get("mu")) else np.nan,
        "epsilon": float(first_row["epsilon"]) if first_row.get("epsilon") is not None and not pd.isna(first_row.get("epsilon")) else np.nan,
        "target_rate_hz": float(first_row.get("target_rate_hz", np.nan)),
        "target_time_sec": float(first_row.get("target_time_sec", np.nan)),
        "window_id": int(last_window.get("window_id", -1)) if last_window else np.nan,
        "sample_count_thisslot": int(last_window.get("sample_count", 0)) if last_window else 0,
        "batch_count_thisslot": int(last_window.get("batch_count", 0)) if last_window else 0,
        "acc_all_samples": float(last_row.get("accuracy_true", np.nan)),
        "acc_thisslot": float(acc_thisslot) if not pd.isna(acc_thisslot) else np.nan,
        "avg_utility": float(acc_thisslot) if not pd.isna(acc_thisslot) else np.nan,
        "avg_acc": float(acc_thisslot) if not pd.isna(acc_thisslot) else np.nan,
        "avg_accuracy": float(acc_thisslot) if not pd.isna(acc_thisslot) else np.nan,
        "avg_delay": float(avg_delay_thisslot) if not pd.isna(avg_delay_thisslot) else np.nan,
        "avg_excess_delay": float(last_window.get("avg_excess_delay_sec", np.nan)) if last_window else np.nan,
        "avg_delay_ratio": (
            float(avg_delay_thisslot) * float(target_rate_hz)
            if not pd.isna(avg_delay_thisslot) and not pd.isna(target_rate_hz)
            else np.nan
        ),
        "lambda_input": float(last_window.get("lambda_input", np.nan)) if last_window else np.nan,
        "solver_elapsed_sec": float(last_window.get("solver_elapsed_sec", np.nan)) if last_window else np.nan,
        "predicted_accuracy": float(last_window.get("predicted_accuracy", np.nan)) if last_window else np.nan,
        "predicted_delay_sec": float(last_window.get("predicted_delay_sec", np.nan)) if last_window else np.nan,
        "avg_accuracy_fit": _finite_mean(item.get("accuracy_fit", np.nan) for item in batch_rows),
        "avg_accuracy_true": _finite_mean(item.get("accuracy_true", np.nan) for item in batch_rows),
        "avg_delay_actual_sec": _finite_mean(item.get("delay_actual_sec", np.nan) for item in batch_rows),
        "avg_delay_pred_sec": _finite_mean(item.get("delay_pred_sec", np.nan) for item in batch_rows),
        "avg_excess_delay_actual_sec": _finite_mean(
            (
                float(item.get("delay_actual_sec", np.nan)) - float(item.get("target_time_sec", np.nan))
                if not pd.isna(item.get("delay_actual_sec", np.nan)) and not pd.isna(item.get("target_time_sec", np.nan))
                else np.nan
            )
            for item in batch_rows
        ),
        "avg_excess_delay_pred_sec": _finite_mean(
            (
                float(item.get("delay_pred_sec", np.nan)) - float(item.get("target_time_sec", np.nan))
                if not pd.isna(item.get("delay_pred_sec", np.nan)) and not pd.isna(item.get("target_time_sec", np.nan))
                else np.nan
            )
            for item in batch_rows
        ),
        "avg_delay_ratio_actual": _finite_mean(item.get("delay_ratio_actual", np.nan) for item in batch_rows),
        "avg_delay_ratio_pred": _finite_mean(item.get("delay_ratio_pred", np.nan) for item in batch_rows),
        "num_batches": int(len(batch_rows)),
        "num_control_steps": int(len(window_summaries)),
    }
    for key, value in first_row.items():
        if str(key).startswith("target_ratio_task") and not pd.isna(value):
            summary_row[str(key)] = float(value)
    return summary_row


def _build_batch_policy_summary_row(batch_rows):
    if not batch_rows:
        return None
    first_row = dict(batch_rows[0])
    target_rate_hz = float(first_row.get("target_rate_hz", np.nan))
    delay_arr = np.asarray([item.get("delay_actual_sec", np.nan) for item in batch_rows], dtype=float)
    pred_delay_arr = np.asarray([item.get("delay_pred_sec", np.nan) for item in batch_rows], dtype=float)
    acc_arr = np.asarray([item.get("accuracy_true", np.nan) for item in batch_rows], dtype=float)
    pred_acc_arr = np.asarray([item.get("accuracy_fit", np.nan) for item in batch_rows], dtype=float)
    target_time_sec = float(first_row.get("target_time_sec", np.nan))
    excess_delay_arr = delay_arr - target_time_sec if not pd.isna(target_time_sec) else np.full_like(delay_arr, np.nan)
    violation_rate = (
        float(np.mean(delay_arr > target_time_sec))
        if delay_arr.size > 0 and not pd.isna(target_time_sec)
        else 0.0
    )
    summary_row = {
        **({"dynamic_timeslot_size": int(first_row["dynamic_timeslot_size"])} if "dynamic_timeslot_size" in first_row and not pd.isna(first_row.get("dynamic_timeslot_size")) else {}),
        **({"target_ratio": float(first_row["target_ratio"])} if "target_ratio" in first_row and not pd.isna(first_row.get("target_ratio")) else {}),
        **({"target_label": str(first_row["target_label"])} if "target_label" in first_row else {}),
        **({"accuracy_estimator_mode": str(first_row["accuracy_estimator_mode"])} if "accuracy_estimator_mode" in first_row else {}),
        "algorithm": str(first_row.get("algorithm", "")),
        "policy": str(first_row.get("policy", first_row.get("algorithm", ""))),
        "policy_key": first_row.get("policy_key"),
        "algorithm_type": first_row.get("algorithm_type", first_row.get("policy_key")),
        "mu": float(first_row["mu"]) if first_row.get("mu") is not None and not pd.isna(first_row.get("mu")) else np.nan,
        "epsilon": float(first_row["epsilon"]) if first_row.get("epsilon") is not None and not pd.isna(first_row.get("epsilon")) else np.nan,
        "target_rate_hz": float(target_rate_hz) if not pd.isna(target_rate_hz) else np.nan,
        "target_time_sec": float(target_time_sec) if not pd.isna(target_time_sec) else np.nan,
        "avg_utility": _finite_mean(acc_arr),
        "avg_acc": _finite_mean(acc_arr),
        "avg_delay": _finite_mean(delay_arr),
        "avg_excess_delay": _finite_mean(excess_delay_arr),
        "avg_delay_ratio": _finite_mean(item.get("delay_ratio_actual", np.nan) for item in batch_rows),
        "avg_accuracy_fit": _finite_mean(pred_acc_arr),
        "avg_accuracy_true": _finite_mean(acc_arr),
        "avg_delay_actual_sec": _finite_mean(delay_arr),
        "avg_delay_pred_sec": _finite_mean(pred_delay_arr),
        "avg_delay_ratio_actual": _finite_mean(item.get("delay_ratio_actual", np.nan) for item in batch_rows),
        "avg_delay_ratio_pred": _finite_mean(item.get("delay_ratio_pred", np.nan) for item in batch_rows),
        "avg_accuracy": _finite_mean(acc_arr),
        "avg_delay_sec": _finite_mean(delay_arr),
        "violation_rate": float(violation_rate),
        "final_lambda": float(batch_rows[-1].get("lambda_value", np.nan)) if batch_rows[-1].get("lambda_value") is not None else np.nan,
        "num_batches": int(len(batch_rows)),
    }
    for key, value in first_row.items():
        if str(key).startswith("target_ratio_task") and not pd.isna(value):
            summary_row[str(key)] = float(value)
    return summary_row


def _build_weighted_avg_policy_summaries(summary_by_scope, tasks):
    existing_avg_row = summary_by_scope.get("avg")
    combined_avg_row = _combine_summary_avg_from_task_scopes(summary_by_scope, tasks, existing_avg_row=existing_avg_row)
    updated = dict(summary_by_scope)
    updated["avg"] = combined_avg_row
    return updated


def _aggregate_policy_summary_rows(summary_rows):
    if not summary_rows:
        return {
            "policy_count": 0,
            "avg_accuracy_over_policies": np.nan,
            "best_accuracy_policy": None,
            "best_avg_accuracy": np.nan,
        }
    frame = pd.DataFrame(summary_rows)
    if frame.empty or "avg_accuracy" not in frame.columns:
        return {
            "policy_count": 0,
            "avg_accuracy_over_policies": np.nan,
            "best_accuracy_policy": None,
            "best_avg_accuracy": np.nan,
        }
    summary = {
        "policy_count": int(frame.shape[0]),
        "avg_accuracy_over_policies": float(frame["avg_accuracy"].mean()),
        "best_accuracy_policy": None,
        "best_avg_accuracy": np.nan,
    }
    if not frame["avg_accuracy"].isna().all():
        best_idx = frame["avg_accuracy"].idxmax()
        summary["best_accuracy_policy"] = str(frame.loc[best_idx, "algorithm"])
        summary["best_avg_accuracy"] = float(frame.loc[best_idx, "avg_accuracy"])
    return summary


def _summarize_policy_rows(rows):
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    summary_rows = []
    group_cols = ["algorithm"]
    if "target_label" in frame.columns:
        group_cols = ["target_label", "algorithm"]
    if "dynamic_timeslot_size" in frame.columns:
        group_cols = ["dynamic_timeslot_size"] + list(group_cols)
    for group_key, group in frame.groupby(group_cols, sort=False):
        dynamic_timeslot_size = None
        target_label = None
        algorithm = None
        if isinstance(group_key, tuple):
            values = list(group_key)
            if "dynamic_timeslot_size" in group_cols:
                dynamic_timeslot_size = values.pop(0)
            if "target_label" in group_cols:
                target_label = values.pop(0)
            algorithm = values.pop(0) if values else None
        else:
            algorithm = group_key
        summary_rows.append(
            {
                **({"dynamic_timeslot_size": int(dynamic_timeslot_size)} if dynamic_timeslot_size is not None else {}),
                **({"target_label": str(target_label)} if target_label is not None else {}),
                **(
                    {"accuracy_estimator_mode": str(group["accuracy_estimator_mode"].iloc[0])}
                    if "accuracy_estimator_mode" in group.columns else {}
                ),
                "algorithm": str(algorithm),
                **({"algorithm_type": str(group["algorithm_type"].iloc[0])} if "algorithm_type" in group.columns else {}),
                **({"mu": float(group["mu"].iloc[0])} if "mu" in group.columns else {}),
                **({"epsilon": float(group["epsilon"].iloc[0])} if "epsilon" in group.columns else {}),
                "avg_utility": float(group["accuracy_true"].mean()) if "accuracy_true" in group.columns else float(group["weighted_utility_true"].mean()),
                "avg_acc": float(group["accuracy_true"].mean()) if "accuracy_true" in group.columns else float(group["weighted_accuracy_true"].mean()),
                "avg_delay": float(group["delay_actual_sec"].mean()) if "delay_actual_sec" in group.columns else float(group["avg_delay_actual_sec"].mean()),
                "avg_excess_delay": float((group["delay_actual_sec"] - group["target_time_sec"]).mean()) if "delay_actual_sec" in group.columns and "target_time_sec" in group.columns else float(group["avg_excess_delay_actual_sec"].mean()),
                "avg_delay_ratio": float(group["delay_ratio_actual"].mean()) if "delay_ratio_actual" in group.columns else float(group["avg_delay_ratio_actual"].mean()),
                "avg_accuracy_fit": float(group["accuracy_fit"].mean()) if "accuracy_fit" in group.columns else np.nan,
                "avg_accuracy_true": float(group["accuracy_true"].mean()) if "accuracy_true" in group.columns else np.nan,
                "avg_weighted_accuracy_fit": float(group["weighted_accuracy_fit"].mean()) if "weighted_accuracy_fit" in group.columns else np.nan,
                "avg_weighted_accuracy_true": float(group["weighted_accuracy_true"].mean()),
                "avg_weighted_utility_fit": float(group["weighted_utility_fit"].mean()) if "weighted_utility_fit" in group.columns else np.nan,
                "avg_weighted_utility_true": float(group["weighted_utility_true"].mean()) if "weighted_utility_true" in group.columns else np.nan,
                "avg_delay_actual_sec": float(group["delay_actual_sec"].mean()) if "delay_actual_sec" in group.columns else float(group["avg_delay_actual_sec"].mean()),
                "avg_delay_pred_sec": float(group["delay_pred_sec"].mean()) if "delay_pred_sec" in group.columns else (float(group["avg_delay_pred_sec"].mean()) if "avg_delay_pred_sec" in group.columns else np.nan),
                "avg_excess_delay_actual_sec": float(group["avg_excess_delay_actual_sec"].mean()),
                "avg_excess_delay_pred_sec": float(group["avg_excess_delay_pred_sec"].mean()) if "avg_excess_delay_pred_sec" in group.columns else np.nan,
                "avg_delay_ratio_actual": float(group["delay_ratio_actual"].mean()) if "delay_ratio_actual" in group.columns else float(group["avg_delay_ratio_actual"].mean()),
                "avg_delay_ratio_pred": float(group["delay_ratio_pred"].mean()) if "delay_ratio_pred" in group.columns else (float(group["avg_delay_ratio_pred"].mean()) if "avg_delay_ratio_pred" in group.columns else np.nan),
                "num_control_steps": int(group.shape[0]),
            }
        )
    return summary_rows


def _target_time_by_task(tasks):
    values = {}
    for task in tasks:
        rate = float(task.target_rate_hz)
        values[int(task.logical_task_id)] = (1.0 / rate) if rate > 0.0 else None
    return values


def _format_target_dir_name(target_ratio):
    if isinstance(target_ratio, dict):
        parts = []
        for task_id in sorted(int(k) for k in target_ratio.keys()):
            ratio_value = float(target_ratio[int(task_id)])
            ratio_text = format(ratio_value, ".12g").rstrip("0").rstrip(".")
            parts.append("task{}_{}".format(int(task_id), _sanitize_tag(ratio_text)))
        return "ratio_{}".format("_".join(parts))
    ratio_value = float(target_ratio)
    ratio_text = format(ratio_value, ".12g").rstrip("0").rstrip(".")
    return "ratio_{}".format(_sanitize_tag(ratio_text))


def _dynamic_timeslot_output_dir(base_dir, dynamic_timeslot_size):
    return os.path.join(base_dir, "dynamic_timeslot_{}".format(int(dynamic_timeslot_size)))


def _annotate_rows_for_target(rows, target_ratio, accuracy_estimator_mode=None, anchor_task_id=None):
    target_label = _format_target_dir_name(target_ratio)
    annotated = []
    for row in rows:
        item = dict(row)
        if isinstance(target_ratio, dict):
            if anchor_task_id is not None and int(anchor_task_id) in target_ratio:
                item["target_ratio"] = float(target_ratio[int(anchor_task_id)])
            for task_id, ratio_value in sorted(target_ratio.items()):
                item["target_ratio_task{}".format(int(task_id))] = float(ratio_value)
        else:
            item["target_ratio"] = float(target_ratio)
        item["target_label"] = str(target_label)
        if accuracy_estimator_mode is not None:
            item["accuracy_estimator_mode"] = str(accuracy_estimator_mode)
        annotated.append(item)
    return annotated


def _write_plot_context(
    output_dir,
    tasks,
    codec_name,
    channel_limit_mbps,
    target_time_ratio,
    accuracy_estimator_mode="fitting_model",
    stein_sigma=None,
    stein_N=None,
    dynamic_timeslot_size=None,
    max_dynamic_timeslot_count=None,
    anchor_task_id=None,
):
    task_entries = []
    for task in tasks:
        target_time_sec = (1.0 / float(task.target_rate_hz)) if float(task.target_rate_hz) > 0.0 else None
        task_entries.append(
            {
                "task_id": int(task.logical_task_id),
                "model": task.model,
                "dataset": task.dataset,
                "batch_size": int(task.batch_size),
                "w_k": float(task.weight),
                "target_rate_hz": float(task.target_rate_hz),
                "target_time_sec": (float(target_time_sec) if target_time_sec is not None else None),
                "scenario_tx_limit_mbps": float(channel_limit_mbps),
            }
        )
    context = {
        "experiment": "online",
        "platform": "jetson",
        "model": "resnet_flan_t5_multi_task",
        "dataset": "mixed",
        "codec_name": codec_name,
        "accuracy_estimator_mode": str(accuracy_estimator_mode),
        "batch_size": "mixed",
        "channel_limit_mbps": float(channel_limit_mbps),
        "per_link_initial_bandwidth_mbps": float(channel_limit_mbps),
        "channel_limit_semantics": "per_link",
        "channel_speed_tag": _format_tx_limit_tag(channel_limit_mbps),
        "delay_definition": "bottleneck_max_stage",
        "tasks": task_entries,
    }
    if isinstance(target_time_ratio, dict):
        context["target_time_ratio_by_task"] = {
            str(int(task_id)): float(ratio_value) for task_id, ratio_value in sorted(target_time_ratio.items())
        }
    elif target_time_ratio is not None:
        context["target_time_ratio"] = float(target_time_ratio)
    if anchor_task_id is not None:
        anchor_task_id = int(anchor_task_id)
        context["anchor_task_id"] = int(anchor_task_id)
        anchor_entry = next((item for item in task_entries if int(item["task_id"]) == int(anchor_task_id)), None)
        if anchor_entry is not None:
            context["target_rate_hz"] = float(anchor_entry["target_rate_hz"])
            context["target_time_sec"] = (
                float(anchor_entry["target_time_sec"]) if anchor_entry.get("target_time_sec") is not None else None
            )
            if isinstance(target_time_ratio, dict) and int(anchor_task_id) in target_time_ratio:
                context["target_time_ratio"] = float(target_time_ratio[int(anchor_task_id)])
    if dynamic_timeslot_size is not None:
        context["dynamic_timeslot_size"] = int(dynamic_timeslot_size)
        context["dynamic_timeslot_mode"] = "anchor_task_min_count"
    if max_dynamic_timeslot_count is not None:
        context["max_dynamic_timeslot_count"] = int(max_dynamic_timeslot_count)
    if str(accuracy_estimator_mode) == "stein_estimator":
        context["stein_sigma"] = float(stein_sigma) if stein_sigma is not None else None
        context["stein_N"] = int(stein_N) if stein_N is not None else None
    with open(os.path.join(output_dir, "plot_context.json"), "w", encoding="utf-8") as handle:
        json.dump(context, handle, indent=2)
    return context


def _compute_control_record(
    t_index,
    algorithm_name,
    algorithm_type,
    policy,
    tasks,
    allocations,
    latest_snapshots,
    current_total_bw_bps,
    aggregate_stats=None,
    control_solve_sec=None,
    window_id=None,
    dynamic_timeslot_size=None,
    actual_window_size=None,
    anchor_task_id=None,
    predicted_accuracy_by_task=None,
    predicted_delay_by_task=None,
):
    weighted_accuracy_true = 0.0
    weight_sum = sum(float(task.weight) for task in tasks)
    avg_delay_actual_sec = float(np.mean([latest_snapshots[int(task.logical_task_id)].total_delay_sec for task in tasks]))
    if predicted_accuracy_by_task is None:
        predicted_accuracy_by_task = {}
    if predicted_delay_by_task is None:
        predicted_delay_by_task = {}
    weighted_accuracy_fit_values = []
    pred_delay_values = []
    pred_excess_values = []
    pred_ratio_values = []
    avg_excess_delay_actual_sec = float(
        np.mean(
            [
                latest_snapshots[int(task.logical_task_id)].total_delay_sec - (1.0 / float(task.target_rate_hz))
                if float(task.target_rate_hz) > 0.0 else np.nan
                for task in tasks
            ]
        )
    )
    avg_delay_ratio_actual = float(
        np.mean(
            [
                latest_snapshots[int(task.logical_task_id)].total_delay_sec * float(task.target_rate_hz)
                for task in tasks
            ]
        )
    )
    anchor_task = None
    if tasks:
        if anchor_task_id is None:
            anchor_task = tasks[0]
        else:
            for task in tasks:
                if int(task.logical_task_id) == int(anchor_task_id):
                    anchor_task = task
                    break
        if anchor_task is None:
            anchor_task = tasks[0]
    record = {
        "t": int(t_index),
        "algorithm": algorithm_name,
        "policy": algorithm_name,
        "policy_key": algorithm_type,
        "algorithm_type": algorithm_type,
        "mu": float(policy.mu) if policy.mu is not None else np.nan,
        "control_solve_sec": (float(control_solve_sec) if control_solve_sec is not None else np.nan),
        "weighted_accuracy_fit": np.nan,
        "weighted_accuracy_true": 0.0,
        "weighted_utility_fit": np.nan,
        "weighted_utility_true": 0.0,
        "avg_delay_actual_sec": avg_delay_actual_sec,
        "avg_delay_pred_sec": np.nan,
        "avg_excess_delay_actual_sec": avg_excess_delay_actual_sec,
        "avg_excess_delay_pred_sec": np.nan,
        "avg_delay_ratio_actual": avg_delay_ratio_actual,
        "avg_delay_ratio_pred": np.nan,
        "window_id": (int(window_id) if window_id is not None else np.nan),
        "dynamic_timeslot_size": (int(dynamic_timeslot_size) if dynamic_timeslot_size is not None else np.nan),
        "actual_window_size": (int(actual_window_size) if actual_window_size is not None else np.nan),
        "anchor_task_id": (int(anchor_task.logical_task_id) if anchor_task is not None else np.nan),
    }
    if aggregate_stats is not None:
        record["completion_total"] = int(
            sum(int(aggregate_stats[int(task.logical_task_id)]["seen"]) for task in tasks)
        )
    for task in tasks:
        tid = int(task.logical_task_id)
        snap = latest_snapshots[tid]
        alloc = allocations[tid]
        weighted_accuracy_true += float(task.weight) * float(snap.running_accuracy)
        eta = np.asarray(alloc["eta"], dtype=float)
        s_comm = np.asarray(alloc["s_comm"], dtype=float)
        s_comp = np.asarray(alloc["s_comp"], dtype=float)
        predicted_accuracy = float(predicted_accuracy_by_task.get(int(tid), np.nan))
        predicted_delay = float(predicted_delay_by_task.get(int(tid), np.nan))
        if not pd.isna(predicted_accuracy):
            weighted_accuracy_fit_values.append(float(task.weight) * predicted_accuracy)
        if not pd.isna(predicted_delay):
            predicted_excess = (
                predicted_delay - (1.0 / float(task.target_rate_hz))
                if float(task.target_rate_hz) > 0.0 else np.nan
            )
            predicted_ratio = predicted_delay * float(task.target_rate_hz) if float(task.target_rate_hz) > 0.0 else np.nan
            pred_delay_values.append(predicted_delay)
            pred_excess_values.append(predicted_excess)
            pred_ratio_values.append(predicted_ratio)
        else:
            predicted_excess = np.nan
            predicted_ratio = np.nan
        if aggregate_stats is not None:
            record["completion_task{}".format(tid)] = int(aggregate_stats[tid]["seen"])
        record["target_rate_hz_task{}".format(tid)] = float(task.target_rate_hz)
        record["accuracy_fit_task{}".format(tid)] = predicted_accuracy
        record["accuracy_true_task{}".format(tid)] = float(snap.running_accuracy)
        record["delay_actual_sec_task{}".format(tid)] = float(snap.total_delay_sec)
        record["end_to_end_delay_sec_task{}".format(tid)] = float(snap.end_to_end_delay_sec)
        record["delay_pred_sec_task{}".format(tid)] = predicted_delay
        record["excess_delay_sec_task{}".format(tid)] = (
            float(snap.total_delay_sec) - (1.0 / float(task.target_rate_hz))
            if float(task.target_rate_hz) > 0.0 else np.nan
        )
        record["excess_delay_pred_sec_task{}".format(tid)] = predicted_excess
        record["delay_ratio_actual_task{}".format(tid)] = float(snap.total_delay_sec) * float(task.target_rate_hz)
        record["delay_ratio_pred_task{}".format(tid)] = predicted_ratio
        record["service_delay_sec_task{}".format(tid)] = float(snap.service_delay_sec)
        record["communication_delay_sec_task{}".format(tid)] = float(snap.communication_delay_sec)
        record["miscellaneous_delay_sec_task{}".format(tid)] = float(snap.miscellaneous_delay_sec)
        record["lambda_task{}".format(tid)] = float(getattr(policy, "lambda_k", {}).get(tid, 0.0))
        record["w_k_task{}".format(tid)] = float(task.weight)
        record["sample_correct_task{}".format(tid)] = float(snap.sample_correct)
        for link_idx, value in enumerate(eta):
            record["eta_task{}_{}".format(tid, link_idx)] = float(value)
        for link_idx, value in enumerate(s_comm):
            record["s_comm_task{}_{}".format(tid, link_idx)] = float(value)
        for node_idx, value in enumerate(s_comp):
            record["s_comp_task{}_{}".format(tid, node_idx)] = float(value)
    if weighted_accuracy_fit_values:
        record["weighted_accuracy_fit"] = float(np.sum(np.asarray(weighted_accuracy_fit_values, dtype=float)) / max(weight_sum, 1e-12))
    record["weighted_accuracy_true"] = float(weighted_accuracy_true / max(weight_sum, 1e-12))
    record["weighted_utility_fit"] = float(record["weighted_accuracy_fit"]) if not pd.isna(record["weighted_accuracy_fit"]) else np.nan
    record["weighted_utility_true"] = float(record["weighted_accuracy_true"])
    if pred_delay_values:
        record["avg_delay_pred_sec"] = float(np.mean(np.asarray(pred_delay_values, dtype=float)))
    if pred_excess_values:
        record["avg_excess_delay_pred_sec"] = float(np.mean(np.asarray(pred_excess_values, dtype=float)))
    if pred_ratio_values:
        record["avg_delay_ratio_pred"] = float(np.mean(np.asarray(pred_ratio_values, dtype=float)))
    if anchor_task is not None:
        anchor_tid = int(anchor_task.logical_task_id)
        anchor_snap = latest_snapshots[anchor_tid]
        anchor_alloc = allocations[anchor_tid]
        anchor_eta = np.asarray(anchor_alloc["eta"], dtype=float)
        anchor_target_rate_hz = float(anchor_task.target_rate_hz)
        anchor_target_time_sec = (1.0 / anchor_target_rate_hz) if anchor_target_rate_hz > 0.0 else np.nan
        record["target_rate_hz"] = anchor_target_rate_hz
        record["target_time_sec"] = anchor_target_time_sec
        record["accuracy_fit"] = float(record.get("accuracy_fit_task{}".format(anchor_tid), np.nan))
        record["accuracy_true"] = float(anchor_snap.running_accuracy)
        record["delay_actual_sec"] = float(anchor_snap.total_delay_sec)
        record["delay_pred_sec"] = float(record.get("delay_pred_sec_task{}".format(anchor_tid), np.nan))
        record["delay_ratio_actual"] = float(anchor_snap.total_delay_sec) * anchor_target_rate_hz
        anchor_pred_delay = float(record.get("delay_pred_sec", np.nan))
        record["delay_ratio_pred"] = float(anchor_pred_delay) * anchor_target_rate_hz
        record["lambda_value"] = float(getattr(policy, "lambda_k", {}).get(anchor_tid, 0.0))
        record["requested_eta"] = ",".join(_format_float4(value) for value in anchor_eta)
        record["executed_eta"] = ",".join(_format_float4(value) for value in anchor_eta)
        for idx, value in enumerate(anchor_eta):
            record["requested_eta_{}".format(idx)] = float(value)
            record["executed_eta_{}".format(idx)] = float(value)
            record["eta_{}".format(idx)] = float(value)
            record["executed_eta_{}".format(idx)] = float(value)
    return record


def _maybe_render_plots(trial_dir, plot_context):
    try:
        render_multi_trial_plots(trial_dir, plot_context=plot_context)
    except Exception:
        logging.exception("Plot rendering failed for %s", trial_dir)


def _retarget_control_records(rows, tasks, algorithm_name=None, algorithm_type=None, anchor_task_id=None):
    patched = []
    task_by_id = {int(task.logical_task_id): task for task in tasks}
    for row in rows:
        item = dict(row)
        if algorithm_name is not None:
            item["algorithm"] = str(algorithm_name)
            item["policy"] = str(algorithm_name)
        if algorithm_type is not None:
            item["algorithm_type"] = str(algorithm_type)
            item["policy_key"] = str(algorithm_type)
        delay_ratio_values = []
        excess_delay_values = []
        delay_ratio_pred_values = []
        excess_delay_pred_values = []
        for task in tasks:
            tid = int(task.logical_task_id)
            target_rate = float(task.target_rate_hz)
            delay_sec = float(item.get("delay_actual_sec_task{}".format(tid), 0.0))
            delay_pred_sec = float(item.get("delay_pred_sec_task{}".format(tid), np.nan))
            item["target_rate_hz_task{}".format(tid)] = target_rate
            item["excess_delay_sec_task{}".format(tid)] = (
                delay_sec - (1.0 / target_rate) if target_rate > 0.0 else np.nan
            )
            item["excess_delay_pred_sec_task{}".format(tid)] = (
                delay_pred_sec - (1.0 / target_rate) if target_rate > 0.0 else np.nan
            )
            item["delay_ratio_actual_task{}".format(tid)] = delay_sec * target_rate
            item["delay_ratio_pred_task{}".format(tid)] = delay_pred_sec * target_rate if target_rate > 0.0 else np.nan
            excess_delay_values.append(item["excess_delay_sec_task{}".format(tid)])
            delay_ratio_values.append(delay_sec * target_rate)
            excess_delay_pred_values.append(item["excess_delay_pred_sec_task{}".format(tid)])
            delay_ratio_pred_values.append(item["delay_ratio_pred_task{}".format(tid)])
        if excess_delay_values:
            item["avg_excess_delay_actual_sec"] = float(np.mean(np.asarray(excess_delay_values, dtype=float)))
        if delay_ratio_values:
            item["avg_delay_ratio_actual"] = float(np.mean(np.asarray(delay_ratio_values, dtype=float)))
        if excess_delay_pred_values:
            item["avg_excess_delay_pred_sec"] = float(np.mean(np.asarray(excess_delay_pred_values, dtype=float)))
        if delay_ratio_pred_values:
            item["avg_delay_ratio_pred"] = float(np.mean(np.asarray(delay_ratio_pred_values, dtype=float)))
        if "weighted_accuracy_fit" in item:
            item["weighted_utility_fit"] = float(item["weighted_accuracy_fit"])
        if "weighted_accuracy_true" in item:
            item["weighted_utility_true"] = float(item["weighted_accuracy_true"])
        anchor_tid = None
        if anchor_task_id is not None:
            anchor_tid = int(anchor_task_id)
        elif "anchor_task_id" in item:
            try:
                anchor_tid = int(item["anchor_task_id"])
            except Exception:
                anchor_tid = None
        if anchor_tid is None and tasks:
            anchor_tid = int(tasks[0].logical_task_id)
        anchor_task = task_by_id.get(int(anchor_tid)) if anchor_tid is not None else None
        if anchor_task is not None:
            target_rate = float(anchor_task.target_rate_hz)
            delay_actual = float(item.get("delay_actual_sec_task{}".format(anchor_tid), np.nan))
            delay_pred = float(item.get("delay_pred_sec_task{}".format(anchor_tid), np.nan))
            item["anchor_task_id"] = int(anchor_tid)
            item["target_rate_hz"] = target_rate
            item["target_time_sec"] = (1.0 / target_rate) if target_rate > 0.0 else np.nan
            item["accuracy_true"] = float(item.get("accuracy_true_task{}".format(anchor_tid), np.nan))
            item["accuracy_fit"] = float(item.get("accuracy_fit_task{}".format(anchor_tid), np.nan))
            item["delay_actual_sec"] = delay_actual
            item["delay_pred_sec"] = delay_pred
            item["delay_ratio_actual"] = delay_actual * target_rate if target_rate > 0.0 else np.nan
            item["delay_ratio_pred"] = delay_pred * target_rate if target_rate > 0.0 else np.nan
            item["lambda_value"] = float(item.get("lambda_task{}".format(anchor_tid), np.nan))
            eta_values = []
            for link_idx in range(NUM_TRANSFER_POINTS):
                eta_value = float(item.get("eta_task{}_{}".format(anchor_tid, link_idx), np.nan))
                item["requested_eta_{}".format(link_idx)] = eta_value
                item["executed_eta_{}".format(link_idx)] = eta_value
                item["eta_{}".format(link_idx)] = eta_value
                eta_values.append(eta_value)
            item["requested_eta"] = ",".join(_format_float4(value) for value in eta_values)
            item["executed_eta"] = ",".join(_format_float4(value) for value in eta_values)
        patched.append(item)
    return patched


def _process_completion(
    result_payload,
    pending,
    inflight_by_task,
    aggregate_stats,
    latest_snapshots,
    shared_bandwidth,
):
    request_id = result_payload["request_id"]
    meta = _finalize_request(request_id, result_payload, pending)
    if meta is None:
        return None
    logical_task_id = int(meta["logical_task_id"])
    inflight_by_task[logical_task_id] = max(0, inflight_by_task[logical_task_id] - 1)
    if meta["task_family"] == "resnet":
        predictions = result_payload["predictions"]
        labels = meta["labels"]
        pred_tensor = torch.as_tensor(predictions, dtype=torch.long)
        labels_cpu = labels.detach().cpu().to(torch.long)
        correct_count = int((pred_tensor == labels_cpu).to(torch.long).sum().item())
        sample_total = int(labels_cpu.numel())
        sample_correct = float(correct_count) / float(sample_total) if sample_total > 0 else 0.0
    else:
        correct_count = int(int(int(result_payload["predicted_label"]) == int(meta["label"])))
        sample_total = 1
        sample_correct = float(correct_count)
    aggregate_stats[logical_task_id]["seen"] += int(sample_total)
    aggregate_stats[logical_task_id]["correct"] += float(correct_count)
    aggregate_stats[logical_task_id]["last_correct"] = sample_correct
    aggregate_stats[logical_task_id]["completions"] += 1
    snapshot = _build_snapshot_from_completion(meta, aggregate_stats)
    latest_snapshots[logical_task_id] = snapshot
    shared_bandwidth.observe(logical_task_id, snapshot.tp_bandwidth_bps)
    return {
        "logical_task_id": logical_task_id,
        "snapshot": snapshot,
        "meta": meta,
        "correct_count": int(correct_count),
        "sample_total": int(sample_total),
        "sample_correct": float(sample_correct),
    }


def _run_stage_round_robin(
    transport,
    sender,
    tasks,
    runtime_map,
    items_by_task,
    initial_allocations,
    shared_bandwidth,
    max_inflight_per_task,
    device,
    policy=None,
    anchor_task_id=None,
    output_dir=None,
    plot_context=None,
    algorithm_name=None,
    algorithm_type=None,
    channel_estimator=None,
    terminate_on_task_id=None,
    loop_non_terminating_tasks=False,
    record_every_completion=False,
    freeze_bandwidth_during_stage=False,
    fixed_stage_total_bw_bps=None,
    tx_limit_mbps_for_warning=None,
    dynamic_timeslot_size=None,
    max_dynamic_timeslot_count=None,
    max_total_batches=None,
    target_completion_count_override=None,
):
    tasks_by_id = {int(task.logical_task_id): task for task in tasks}
    task_order = [int(task.logical_task_id) for task in tasks]
    rr_index = 0
    pending = {}
    dispatch_futures = {}
    inflight_by_task = {tid: 0 for tid in task_order}
    aggregate_stats = {
        tid: {
            "seen": 0,
            "correct": 0.0,
            "last_correct": 0.0,
            "completions": 0,
            "delay_ratio_sum": 0.0,
            "delay_ratio_count": 0,
        }
        for tid in task_order
    }
    latest_snapshots = {}
    current_allocations = _clone_allocations(initial_allocations)
    if fixed_stage_total_bw_bps is not None:
        total_bw_bps = np.asarray(fixed_stage_total_bw_bps, dtype=float).copy()
    else:
        total_bw_bps = _current_total_bandwidth_bps(channel_estimator, shared_bandwidth)
    _warn_if_per_link_bandwidth_low(
        total_bw_bps,
        tx_limit_mbps=tx_limit_mbps_for_warning,
        context_label="{}:stage_start".format(algorithm_name or "warmup"),
    )
    fixed_channel_policy = _policy_uses_fixed_channel(policy=policy, algorithm_type=algorithm_type) or bool(
        freeze_bandwidth_during_stage
    ) or bool(
        fixed_stage_total_bw_bps is not None
    )
    window_bandwidth_control = bool(
        policy is not None
        and anchor_task_id is not None
        and dynamic_timeslot_size is not None
        and not fixed_channel_policy
    )
    fixed_total_bw_bps = (
        np.asarray(fixed_stage_total_bw_bps, dtype=float).copy()
        if fixed_stage_total_bw_bps is not None
        else np.asarray(total_bw_bps, dtype=float).copy()
    )
    control_records = []
    completion_rows = []
    control_step = 0
    total_tp_stats_history = []
    terminated_task_id = None
    termination_reason = None
    completion_window_queues = {
        int(tid): deque()
        for tid in task_order
    }
    solver_job = None
    window_summaries = []
    window_summary_by_id = {}
    next_window_id = 1
    next_policy_version = 1
    anchor_dispatch_count = 0
    dispatch_horizon_closed = False
    base_items_by_task = {int(tid): list(items_by_task[tid]) for tid in task_order}
    effective_max_dynamic_timeslot_count = (
        None if target_completion_count_override is not None else max_dynamic_timeslot_count
    )
    effective_max_total_batches = (
        None if target_completion_count_override is not None else max_total_batches
    )
    target_completion_count = (
        None if target_completion_count_override is None else max(1, int(target_completion_count_override))
    )
    if terminate_on_task_id is not None:
        if target_completion_count is not None:
            pass
        elif effective_max_total_batches is not None:
            target_completion_count = max(1, int(effective_max_total_batches))
        elif effective_max_dynamic_timeslot_count is None:
            target_completion_count = int(len(base_items_by_task[int(terminate_on_task_id)]))
    current_control_state = {
        "policy_version": 0,
        "window_id": 0,
        "allocations": _clone_allocations(current_allocations),
        "lambda_k": (
            {int(k): float(v) for k, v in getattr(policy, "lambda_k", {}).items()}
            if hasattr(policy, "lambda_k") else {}
        ),
        "control_solve_sec": np.nan,
        "total_bw_bps": np.asarray(total_bw_bps, dtype=float).copy(),
        "predicted_accuracy_by_task": {},
        "predicted_delay_by_task": {},
    }

    logging.info(
        "[A][%s] Stage start: tasks=%s pending_items=%s max_inflight_per_task=%s total_bw_bps=%s terminate_on_task_id=%s target_completion_count=%s loop_non_terminating_tasks=%s",
        algorithm_name or "warmup",
        task_order,
        {int(tid): len(items_by_task[tid]) for tid in task_order},
        int(max_inflight_per_task),
        [float(x) for x in np.asarray(total_bw_bps, dtype=float)],
        (None if terminate_on_task_id is None else int(terminate_on_task_id)),
        target_completion_count,
        bool(loop_non_terminating_tasks),
    )
    if fixed_channel_policy:
        logging.info(
            "[A][%s] Fixed-channel baseline: freezing total_bw_bps=%s for entire stage",
            algorithm_name or "warmup",
            [float(x) for x in np.asarray(fixed_total_bw_bps, dtype=float)],
        )
    if policy is not None:
        logging.info(
            "[A][%s] Initial allocations: %s",
            algorithm_name,
            {
                int(tid): {
                    "eta": [float(x) for x in np.asarray(initial_allocations[tid]["eta"], dtype=float)],
                    "s_comm": [float(x) for x in np.asarray(initial_allocations[tid]["s_comm"], dtype=float)],
                    "s_comp": [float(x) for x in np.asarray(initial_allocations[tid]["s_comp"], dtype=float)],
                }
                for tid in task_order
            },
        )

    def _anchor_buffer_count():
        if anchor_task_id is None:
            return int(sum(len(queue_items) for queue_items in completion_window_queues.values()))
        return int(len(completion_window_queues.get(int(anchor_task_id), ())))

    def _flatten_completion_window_queues():
        snapshot_records = []
        for tid in task_order:
            snapshot_records.extend(list(completion_window_queues[int(tid)]))
        return snapshot_records

    def _should_stop_new_control_updates():
        if effective_max_dynamic_timeslot_count is None:
            return False
        return int(len(window_summaries)) >= int(effective_max_dynamic_timeslot_count)

    def _dispatch_allowed():
        if terminate_on_task_id is None:
            return True
        if target_completion_count_override is not None:
            return int(anchor_dispatch_count) < int(target_completion_count)
        if effective_max_total_batches is not None and int(anchor_dispatch_count) >= int(effective_max_total_batches):
            return False
        if effective_max_dynamic_timeslot_count is None:
            return int(anchor_dispatch_count) < int(len(base_items_by_task[int(terminate_on_task_id)]))
        return int(len(window_summaries)) < int(effective_max_dynamic_timeslot_count)

    def _close_dispatch_horizon(reason):
        nonlocal dispatch_horizon_closed, terminated_task_id, termination_reason
        if dispatch_horizon_closed:
            return
        dispatch_horizon_closed = True
        if termination_reason is None:
            termination_reason = str(reason)
        for tid in task_order:
            items_by_task[tid].clear()
        if terminated_task_id is None and terminate_on_task_id is not None:
            terminated_task_id = int(terminate_on_task_id)
        logging.info(
            "[A][%s] Dispatch horizon closed: reason=%s anchor_task=%s anchor_dispatched=%s windows=%s max_total_batches=%s max_dynamic_timeslot_count=%s",
            algorithm_name or "warmup",
            str(reason),
            (None if terminate_on_task_id is None else int(terminate_on_task_id)),
            int(anchor_dispatch_count),
            int(len(window_summaries)),
            (None if effective_max_total_batches is None else int(effective_max_total_batches)),
            (None if effective_max_dynamic_timeslot_count is None else int(effective_max_dynamic_timeslot_count)),
        )

    def _try_start_control_solver(solver_executor):
        nonlocal solver_job, completion_window_queues, next_window_id
        if policy is None or anchor_task_id is None or dynamic_timeslot_size is None:
            return
        if solver_job is not None:
            return
        if _should_stop_new_control_updates():
            return
        if len(latest_snapshots) < len(tasks):
            return
        if _anchor_buffer_count() < max(1, int(dynamic_timeslot_size)):
            return
        snapshot_records = _flatten_completion_window_queues()
        snapshot_records_by_task = {
            int(tid): deque(list(completion_window_queues[int(tid)]))
            for tid in task_order
        }
        completion_window_queues = {
            int(tid): deque()
            for tid in task_order
        }
        averaged_snapshots = _aggregate_control_window_snapshots(snapshot_records, tasks, latest_snapshots)
        if len(averaged_snapshots) < len(tasks):
            completion_window_queues = {
                int(tid): deque(list(snapshot_records_by_task[int(tid)]) + list(completion_window_queues[int(tid)]))
                for tid in task_order
            }
            return
        window_total_bw_bps = _aggregate_control_window_total_bandwidth_bps(
            snapshot_records,
            tasks,
            latest_snapshots,
        )
        window_id = int(next_window_id)
        next_window_id += 1
        anchor_completion_count = int(
            sum(1 for item in snapshot_records if int(item["logical_task_id"]) == int(anchor_task_id))
        )
        summary_row = {
            "window_id": int(window_id),
            "dynamic_timeslot_size": int(dynamic_timeslot_size),
            "actual_window_size": int(len(snapshot_records)),
            "anchor_task_id": int(anchor_task_id),
            "anchor_completion_count": int(anchor_completion_count),
            "solver_elapsed_sec": np.nan,
        }
        anchor_snap = averaged_snapshots.get(int(anchor_task_id))
        anchor_task = tasks_by_id.get(int(anchor_task_id))
        if anchor_snap is not None and anchor_task is not None:
            anchor_records = [
                item for item in snapshot_records if int(item["logical_task_id"]) == int(anchor_task_id)
            ]
            anchor_total_samples = int(sum(int(item.get("sample_total", 0)) for item in anchor_records))
            anchor_total_correct = int(sum(int(item.get("correct_count", 0)) for item in anchor_records))
            summary_row["sample_count"] = int(anchor_completion_count)
            summary_row["batch_count"] = int(anchor_completion_count)
            summary_row["window_total_samples"] = int(anchor_total_samples)
            summary_row["window_correct_samples"] = int(anchor_total_correct)
            summary_row["avg_accuracy"] = (
                float(anchor_total_correct) / float(anchor_total_samples)
                if anchor_total_samples > 0 else np.nan
            )
            summary_row["avg_delay_sec"] = float(anchor_snap.total_delay_sec)
            summary_row["target_time_sec"] = (
                (1.0 / float(anchor_task.target_rate_hz)) if float(anchor_task.target_rate_hz) > 0.0 else np.nan
            )
        for task in tasks:
            tid = int(task.logical_task_id)
            records_for_task = [item for item in snapshot_records if int(item["logical_task_id"]) == tid]
            snap = averaged_snapshots[tid]
            task_total_samples = int(sum(int(item.get("sample_total", 0)) for item in records_for_task))
            task_total_correct = int(sum(int(item.get("correct_count", 0)) for item in records_for_task))
            summary_row["completion_count_task{}".format(tid)] = int(len(records_for_task))
            summary_row["sample_count_task{}".format(tid)] = int(len(records_for_task))
            summary_row["batch_count_task{}".format(tid)] = int(len(records_for_task))
            summary_row["window_total_samples_task{}".format(tid)] = int(task_total_samples)
            summary_row["window_correct_samples_task{}".format(tid)] = int(task_total_correct)
            summary_row["avg_accuracy_task{}".format(tid)] = (
                float(task_total_correct) / float(task_total_samples)
                if task_total_samples > 0 else np.nan
            )
            summary_row["avg_delay_sec_task{}".format(tid)] = float(snap.total_delay_sec)
            summary_row["avg_end_to_end_delay_sec_task{}".format(tid)] = float(snap.end_to_end_delay_sec)
            for link_idx, value in enumerate(np.asarray(snap.a_ref, dtype=float)):
                summary_row["avg_a_task{}_{}".format(tid, link_idx)] = float(value)
            for link_idx, value in enumerate(np.asarray(snap.tp_bandwidth_bps, dtype=float)):
                summary_row["avg_c_task{}_{}".format(tid, link_idx)] = float(value)
            for node_idx, value in enumerate(np.asarray(snap.tau_list, dtype=float)):
                summary_row["avg_tau_task{}_{}".format(tid, node_idx)] = float(value)
        window_summaries.append(summary_row)
        window_summary_by_id[int(window_id)] = summary_row
        if fixed_channel_policy:
            bw_snapshot = np.asarray(fixed_total_bw_bps, dtype=float).copy()
        elif channel_estimator is not None:
            channel_estimator.observe_task(_tp_stats_from_total_bandwidth_bps(window_total_bw_bps))
            bw_snapshot = _predict_total_bandwidth_bps_from_estimator(channel_estimator, window_total_bw_bps)
        else:
            bw_snapshot = np.asarray(window_total_bw_bps, dtype=float).copy()
        summary_row["window_total_bw_link0_mbps"] = float(bw_snapshot[0]) / 1e6 if bw_snapshot.size > 0 else np.nan
        summary_row["window_total_bw_link1_mbps"] = float(bw_snapshot[1]) / 1e6 if bw_snapshot.size > 1 else np.nan
        summary_row["window_total_bw_link2_mbps"] = float(bw_snapshot[2]) / 1e6 if bw_snapshot.size > 2 else np.nan
        solver_job = solver_executor.submit(
            _solve_multi_policy_snapshot,
            policy,
            averaged_snapshots,
            bw_snapshot,
            tasks_by_id,
            int(window_id),
        )

    def _maybe_collect_control_solver_result():
        nonlocal solver_job, current_allocations, current_control_state, control_step, next_policy_version
        if solver_job is None or not solver_job.done():
            return
        solved = solver_job.result()
        current_allocations = _clone_allocations(solved["allocations"])
        predicted_accuracy_by_task = dict(solved.get("predicted_accuracy_by_task", {}))
        predicted_delay_by_task = dict(solved.get("predicted_delay_by_task", {}))
        if hasattr(policy, "lambda_k") and isinstance(solved.get("lambda_k"), dict):
            policy.lambda_k = {int(k): float(v) for k, v in solved["lambda_k"].items()}
        current_control_state = {
            "policy_version": int(next_policy_version),
            "window_id": int(solved["window_id"]),
            "allocations": _clone_allocations(current_allocations),
            "lambda_k": {int(k): float(v) for k, v in solved.get("lambda_k", {}).items()},
            "control_solve_sec": float(solved["control_solve_sec"]),
            "total_bw_bps": np.asarray(solved["total_bw_bps"], dtype=float).copy(),
            "predicted_accuracy_by_task": dict(predicted_accuracy_by_task),
            "predicted_delay_by_task": dict(predicted_delay_by_task),
        }
        summary_row = window_summary_by_id.get(int(solved["window_id"]))
        if summary_row is not None:
            summary_row["solver_elapsed_sec"] = float(solved["control_solve_sec"])
            summary_row["policy_version"] = int(next_policy_version)
            summary_row["lambda_input"] = float(solved.get("lambda_k", {}).get(int(anchor_task_id), 0.0))
            summary_row["predicted_accuracy"] = float(predicted_accuracy_by_task.get(int(anchor_task_id), np.nan))
            summary_row["predicted_delay_sec"] = float(predicted_delay_by_task.get(int(anchor_task_id), np.nan))
            for task in tasks:
                tid = int(task.logical_task_id)
                alloc = current_allocations[tid]
                for link_idx, value in enumerate(np.asarray(alloc["eta"], dtype=float)):
                    summary_row["eta_task{}_{}".format(tid, link_idx)] = float(value)
                    if int(tid) == int(anchor_task_id):
                        summary_row["requested_eta_{}".format(link_idx)] = float(value)
                for link_idx, value in enumerate(np.asarray(alloc["s_comm"], dtype=float)):
                    summary_row["s_comm_task{}_{}".format(tid, link_idx)] = float(value)
                for node_idx, value in enumerate(np.asarray(alloc["s_comp"], dtype=float)):
                    summary_row["s_comp_task{}_{}".format(tid, node_idx)] = float(value)
        control_records.append(
            _compute_control_record(
                control_step,
                algorithm_name or str(getattr(policy, "policy_name", "policy")),
                algorithm_type or str(getattr(policy, "policy_key", "policy")),
                policy,
                tasks,
                current_allocations,
                solved["averaged_snapshots"],
                solved["total_bw_bps"],
                aggregate_stats=aggregate_stats,
                control_solve_sec=float(solved["control_solve_sec"]),
                window_id=int(solved["window_id"]),
                dynamic_timeslot_size=(None if dynamic_timeslot_size is None else int(dynamic_timeslot_size)),
                actual_window_size=(
                    None if summary_row is None else int(summary_row.get("actual_window_size", 0))
                ),
                anchor_task_id=anchor_task_id,
                predicted_accuracy_by_task=predicted_accuracy_by_task,
                predicted_delay_by_task=predicted_delay_by_task,
            )
        )
        logging.info(
            "[A][%s] Control update #%s window=%s solve_sec=%.4f policy_v=%s lambda=%s",
            algorithm_name or "policy",
            int(control_step),
            int(solved["window_id"]),
            float(solved["control_solve_sec"]),
            int(next_policy_version),
            {int(k): float(v) for k, v in getattr(policy, "lambda_k", {}).items()},
        )
        control_step += 1
        next_policy_version += 1
        solver_job = None

    dispatch_workers = max(len(task_order) * int(max_inflight_per_task), 1)
    with ThreadPoolExecutor(max_workers=dispatch_workers) as dispatch_executor, ThreadPoolExecutor(max_workers=1) as solver_executor:
        while any(items_by_task[tid] for tid in task_order) or pending or dispatch_futures:
            _maybe_collect_control_solver_result()
            _try_start_control_solver(solver_executor)
            if _should_stop_new_control_updates():
                _close_dispatch_horizon("max_dynamic_timeslot_count_reached")
            if (not dispatch_horizon_closed) and (not _dispatch_allowed()):
                _close_dispatch_horizon("anchor_dispatch_limit_reached")
            for future, future_meta in list(dispatch_futures.items()):
                if not future.done():
                    continue
                dispatch_futures.pop(future, None)
                logical_task_id = int(future_meta["logical_task_id"])
                try:
                    meta = future.result()
                except Exception:
                    inflight_by_task[logical_task_id] = max(0, inflight_by_task[logical_task_id] - 1)
                    logging.exception(
                        "[A][%s] Dispatch worker failed request=%s task=%s batch=%s",
                        algorithm_name or "warmup",
                        future_meta["request_id"],
                        logical_task_id,
                        int(future_meta["batch_idx"]),
                    )
                    continue
                request_id = str(meta["request_id"])
                meta["policy_version"] = int(future_meta["policy_version"])
                meta["window_id"] = int(future_meta["window_id"])
                meta["control_solve_sec"] = float(future_meta["control_solve_sec"])
                meta["total_bw_bps"] = [float(x) for x in np.asarray(future_meta["total_bw_bps"], dtype=float)]
                meta["s_comm"] = [float(x) for x in np.asarray(future_meta["s_comm"], dtype=float)]
                meta["s_comp"] = [float(x) for x in np.asarray(future_meta["s_comp"], dtype=float)]
                meta["selected_accuracy_est"] = float(future_meta["selected_accuracy_est"])
                meta["selected_delay_est"] = float(future_meta["selected_delay_est"])
                meta["lambda_value"] = float(future_meta["lambda_value"])
                pending[request_id] = meta
                logging.info(
                    "[A][%s] Dispatch task=%s batch=%s eta=%s s_comm=%s total_bw_bps=%s policy_v=%s window=%s",
                    algorithm_name or "warmup",
                    logical_task_id,
                    int(future_meta["batch_idx"]),
                    [float(x) for x in np.asarray(future_meta["eta"], dtype=float)],
                    [float(x) for x in np.asarray(future_meta["s_comm"], dtype=float)],
                    [float(x) for x in np.asarray(future_meta["total_bw_bps"], dtype=float)],
                    int(future_meta["policy_version"]),
                    int(future_meta["window_id"]),
                )

            while terminated_task_id is None:
                if dispatch_horizon_closed or _should_stop_new_control_updates() or (not _dispatch_allowed()):
                    break
                candidate_tid = None
                for _ in range(len(task_order)):
                    tid = task_order[rr_index % len(task_order)]
                    rr_index += 1
                    if (
                        terminate_on_task_id is not None
                        and int(tid) != int(terminate_on_task_id)
                        and not items_by_task[int(terminate_on_task_id)]
                        and inflight_by_task[int(terminate_on_task_id)] > 0
                    ):
                        continue
                    if (
                        terminate_on_task_id is not None
                        and not items_by_task[tid]
                        and base_items_by_task.get(int(tid))
                    ):
                        should_refill = False
                        if int(tid) == int(terminate_on_task_id):
                            should_refill = bool(
                                (
                                    target_completion_count_override is not None
                                    or effective_max_total_batches is not None
                                    or effective_max_dynamic_timeslot_count is not None
                                )
                                and _dispatch_allowed()
                            )
                        else:
                            should_refill = bool(loop_non_terminating_tasks and _dispatch_allowed())
                        if should_refill:
                            items_by_task[tid] = deque(list(base_items_by_task[int(tid)]))
                    if items_by_task[tid] and inflight_by_task[tid] < int(max_inflight_per_task):
                        candidate_tid = tid
                        break
                if candidate_tid is None:
                    break
                batch = items_by_task[candidate_tid].popleft()
                effective_total_bw_bps = (
                    np.asarray(fixed_total_bw_bps, dtype=float).copy()
                    if fixed_channel_policy
                    else np.asarray(current_control_state["total_bw_bps"], dtype=float).copy()
                )
                alloc = copy.deepcopy(current_control_state["allocations"][candidate_tid])
                alloc["total_bw_bps"] = effective_total_bw_bps.copy()
                request_id = str(uuid.uuid4())
                inflight_by_task[candidate_tid] += 1
                future_meta = {
                    "request_id": request_id,
                    "logical_task_id": int(candidate_tid),
                    "batch_idx": int(batch.get("batch_idx", -1)),
                    "eta": np.asarray(alloc["eta"], dtype=float).copy(),
                    "s_comm": np.asarray(alloc["s_comm"], dtype=float).copy(),
                    "s_comp": np.asarray(alloc["s_comp"], dtype=float).copy(),
                    "total_bw_bps": effective_total_bw_bps.copy(),
                    "policy_version": int(current_control_state["policy_version"]),
                    "window_id": int(current_control_state["window_id"]),
                    "control_solve_sec": float(current_control_state["control_solve_sec"]),
                    "selected_accuracy_est": float(current_control_state.get("predicted_accuracy_by_task", {}).get(int(candidate_tid), np.nan)),
                    "selected_delay_est": float(current_control_state.get("predicted_delay_by_task", {}).get(int(candidate_tid), np.nan)),
                    "lambda_value": float(current_control_state.get("lambda_k", {}).get(int(candidate_tid), 0.0)),
                }
                if terminate_on_task_id is not None and int(candidate_tid) == int(terminate_on_task_id):
                    anchor_dispatch_count += 1
                dispatch_futures[
                    dispatch_executor.submit(
                        _dispatch_async_request,
                        sender,
                        runtime_map,
                        tasks_by_id,
                        batch,
                        candidate_tid,
                        alloc,
                        device,
                        request_id,
                    )
                ] = future_meta

            if terminated_task_id is not None and not pending and not dispatch_futures:
                break
            if not pending and not dispatch_futures and not any(items_by_task[tid] for tid in task_order):
                break
            if not pending:
                time.sleep(0.01)
                continue
            try:
                envelope = transport.receive(msg_type=MessageType.FINAL_RESULT, source="D", timeout=0.25)
            except TimeoutError:
                continue
            completion = _process_completion(
                envelope["payload"],
                pending,
                inflight_by_task,
                aggregate_stats,
                latest_snapshots,
                shared_bandwidth,
            )
            if completion is None:
                logging.warning("[A][%s] Received FINAL_RESULT for unknown request_id=%s", algorithm_name or "warmup", envelope["payload"].get("request_id"))
                continue
            measured_channel_mbps = [
                float(value) / 1e6 for value in np.asarray(completion["snapshot"].tp_bandwidth_bps, dtype=float)
            ]
            alloc_total_bw_bps = np.asarray(completion["meta"].get("total_bw_bps", []), dtype=float)
            alloc_s_comm = np.asarray(completion["meta"].get("s_comm", []), dtype=float)
            if alloc_total_bw_bps.size > 0 and alloc_s_comm.size > 0:
                alloc_link_mbps = [
                    float(value) / 1e6
                    for value in np.asarray(alloc_total_bw_bps * alloc_s_comm, dtype=float)
                ]
            else:
                alloc_link_mbps = []
            logging.info(
                "[A][%s] Completion task=%s delay=%.4fs service=%.4fs comm=%.4fs channel_mbps=%s alloc_link_mbps=%s policy_v=%s window=%s",
                algorithm_name or "warmup",
                int(completion["logical_task_id"]),
                float(completion["snapshot"].total_delay_sec),
                float(completion["snapshot"].service_delay_sec),
                float(completion["snapshot"].communication_delay_sec),
                [round(float(x), 4) for x in measured_channel_mbps],
                [round(float(x), 4) for x in alloc_link_mbps],
                int(completion["meta"].get("policy_version", 0)),
                int(completion["meta"].get("window_id", 0)),
            )
            task_id = int(completion["logical_task_id"])
            task = tasks_by_id[task_id]
            delay_ratio_actual = (
                float(completion["snapshot"].total_delay_sec) * float(task.target_rate_hz)
                if float(task.target_rate_hz) > 0.0
                else np.nan
            )
            if not pd.isna(delay_ratio_actual):
                aggregate_stats[task_id]["delay_ratio_sum"] += float(delay_ratio_actual)
                aggregate_stats[task_id]["delay_ratio_count"] += 1
            running_acc = (
                float(aggregate_stats[task_id]["correct"]) / float(aggregate_stats[task_id]["seen"])
                if int(aggregate_stats[task_id]["seen"]) > 0
                else np.nan
            )
            running_delay_ratio = (
                float(aggregate_stats[task_id]["delay_ratio_sum"]) / float(aggregate_stats[task_id]["delay_ratio_count"])
                if int(aggregate_stats[task_id]["delay_ratio_count"]) > 0
                else np.nan
            )
            logging.info(
                "[A][%s] Running summary task=%s family=%s acc=%.4f delay_ratio=%.4f",
                algorithm_name or "warmup",
                task_id,
                str(completion["meta"].get("task_family", "")),
                float(running_acc) if not pd.isna(running_acc) else np.nan,
                float(running_delay_ratio) if not pd.isna(running_delay_ratio) else np.nan,
            )
            if (
                policy is not None
                and tx_limit_mbps_for_warning is not None
                and len(latest_snapshots) == len(tasks)
            ):
                observed_total_bw_bps = np.asarray(shared_bandwidth.total_bandwidth_bps(), dtype=float)
                breach_threshold_bps = float(tx_limit_mbps_for_warning) * 1.5 * 1e6
                if observed_total_bw_bps.size > 0 and np.any(observed_total_bw_bps > breach_threshold_bps):
                    breached_links = [
                        int(idx)
                        for idx, value in enumerate(observed_total_bw_bps)
                        if float(value) > breach_threshold_bps
                    ]
                    logging.warning(
                        "[A][%s] Channel collapse detected after completion task=%s: total_link_mbps=%s threshold_mbps=%.4f breached_links=%s; stopping current policy",
                        algorithm_name or "policy",
                        int(completion["logical_task_id"]),
                        [round(float(x) / 1e6, 4) for x in observed_total_bw_bps],
                        float(breach_threshold_bps) / 1e6,
                        breached_links,
                    )
                    if terminated_task_id is None and terminate_on_task_id is None:
                        terminated_task_id = int(completion["logical_task_id"])
                    _close_dispatch_horizon("channel_collapse_over_1p5x_limit")
            completion_rows.append(
                {
                    "logical_task_id": int(completion["logical_task_id"]),
                    "algorithm": str(algorithm_name or ""),
                    "policy": str(algorithm_name or ""),
                    "policy_key": (algorithm_type or str(getattr(policy, "policy_key", ""))),
                    "mu": (
                        float(getattr(policy, "mu", np.nan))
                        if policy is not None and getattr(policy, "mu", None) is not None
                        else np.nan
                    ),
                    "epsilon": (
                        float(getattr(policy, "epsilon", np.nan))
                        if policy is not None and getattr(policy, "epsilon", None) is not None
                        else np.nan
                    ),
                    "batch_idx": int(completion["meta"].get("batch_idx", -1)),
                    "delay_sec": float(completion["snapshot"].total_delay_sec),
                    "end_to_end_delay_sec": float(completion["snapshot"].end_to_end_delay_sec),
                    "service_delay_sec": float(completion["snapshot"].service_delay_sec),
                    "communication_delay_sec": float(completion["snapshot"].communication_delay_sec),
                    "running_accuracy": float(completion["snapshot"].running_accuracy),
                    "sample_correct": float(completion["sample_correct"]),
                    "correct_count": int(completion["correct_count"]),
                    "sample_total": int(completion["sample_total"]),
                    "policy_version": int(completion["meta"].get("policy_version", 0)),
                    "window_id": int(completion["meta"].get("window_id", 0)),
                    "selected_accuracy_est": float(completion["meta"].get("selected_accuracy_est", np.nan)),
                    "selected_delay_est": float(completion["meta"].get("selected_delay_est", np.nan)),
                    "lambda_value": float(completion["meta"].get("lambda_value", np.nan)),
                    "compressor_name": completion["meta"].get("compressor_name"),
                    "requested_eta": list(completion["meta"].get("requested_eta", [])),
                    "executed_eta": list(completion["meta"].get("executed_eta", [])),
                }
            )
            if policy is not None and anchor_task_id is not None and dynamic_timeslot_size is not None:
                completion_window_queues[int(completion["logical_task_id"])].append(
                    {
                        "logical_task_id": int(completion["logical_task_id"]),
                        "snapshot": completion["snapshot"],
                        "correct_count": int(completion["correct_count"]),
                        "sample_total": int(completion["sample_total"]),
                        "policy_version": int(completion["meta"].get("policy_version", 0)),
                    }
                )
            if terminated_task_id is None:
                if (
                    terminate_on_task_id is not None
                    and target_completion_count is not None
                    and int(aggregate_stats[int(terminate_on_task_id)]["completions"]) >= int(target_completion_count)
                    and inflight_by_task[int(terminate_on_task_id)] == 0
                ):
                    terminated_task_id = int(terminate_on_task_id)
                    logging.info(
                        "[A][%s] Termination triggered: anchor/slower task=%s reached target completion count=%s; draining remaining in-flight requests before ending stage",
                        algorithm_name or "warmup",
                        int(terminated_task_id),
                        int(target_completion_count),
                    )
                else:
                    for tid in task_order:
                        if (not items_by_task[tid]) and inflight_by_task[tid] == 0 and not loop_non_terminating_tasks:
                            terminated_task_id = int(tid)
                            logging.info(
                                "[A][%s] Termination triggered: task=%s has completed all items; draining remaining in-flight requests before ending stage",
                                algorithm_name or "warmup",
                                int(terminated_task_id),
                            )
                            break
            if len(latest_snapshots) == len(tasks):
                total_tp_stats = shared_bandwidth.total_tp_stats()
                total_tp_stats_history.append(total_tp_stats)
                if window_bandwidth_control:
                    total_bw_bps = np.asarray(current_control_state["total_bw_bps"], dtype=float).copy()
                elif fixed_channel_policy:
                    total_bw_bps = fixed_total_bw_bps.copy()
                elif channel_estimator is not None:
                    channel_estimator.observe_task(total_tp_stats)
                    total_bw_bps = _current_total_bandwidth_bps(channel_estimator, shared_bandwidth)
                else:
                    total_bw_bps = shared_bandwidth.total_bandwidth_bps()
                if not fixed_channel_policy:
                    _warn_if_per_link_bandwidth_low(
                        total_bw_bps,
                        tx_limit_mbps=tx_limit_mbps_for_warning,
                        context_label="{}:control_step_{}".format(algorithm_name or "warmup", int(control_step)),
                    )
                if policy is not None and record_every_completion:
                    control_records.append(
                        _compute_control_record(
                            control_step,
                            algorithm_name or str(getattr(policy, "policy_name", "policy")),
                            algorithm_type or str(getattr(policy, "policy_key", "policy")),
                            policy,
                            tasks,
                            current_control_state["allocations"],
                            latest_snapshots,
                            current_control_state["total_bw_bps"],
                            aggregate_stats=aggregate_stats,
                            control_solve_sec=np.nan,
                            anchor_task_id=anchor_task_id,
                            predicted_accuracy_by_task=current_control_state.get("predicted_accuracy_by_task", {}),
                            predicted_delay_by_task=current_control_state.get("predicted_delay_by_task", {}),
                        )
                    )
                    control_step += 1
                _maybe_collect_control_solver_result()
                _try_start_control_solver(solver_executor)
        while solver_job is not None or (
            policy is not None
            and anchor_task_id is not None
            and dynamic_timeslot_size is not None
            and _anchor_buffer_count() >= max(1, int(dynamic_timeslot_size))
        ):
            _maybe_collect_control_solver_result()
            _try_start_control_solver(solver_executor)
            if solver_job is None and _anchor_buffer_count() < max(1, int(dynamic_timeslot_size)):
                break
            time.sleep(0.01)
    return {
        "latest_snapshots": latest_snapshots,
        "control_records": control_records,
        "completion_rows": completion_rows,
        "aggregate_stats": aggregate_stats,
        "final_allocations": current_control_state["allocations"],
        "total_tp_stats_history": total_tp_stats_history,
        "terminated_task_id": terminated_task_id,
        "termination_reason": termination_reason,
        "window_summaries": window_summaries,
    }


def _run_worker_b_task(sender, runtime_map, envelope, device):
    payload = envelope["payload"]
    runtime = runtime_map[int(payload["logical_task_id"])]
    family = payload["task_family"]
    logging.info(
        "[B] Received request=%s task=%s family=%s batch=%s eta=%s",
        payload.get("request_id"),
        int(payload["logical_task_id"]),
        family,
        int(payload.get("batch_idx", -1)),
        payload.get("feature_k_values"),
    )
    if family == "resnet":
        t_restore = time.perf_counter()
        hidden = restore_activation_payload(payload["activation_comp"], device)
        restore_sec = time.perf_counter() - t_restore
        t_comp = time.perf_counter()
        with torch.no_grad():
            hidden_out = runtime["partition"](hidden)
        compute_sec = time.perf_counter() - t_comp
        t_prepare = time.perf_counter()
        hidden_comp, stats = build_activation_payload(
            hidden_out,
            compression_param=payload["compression_params_list"][1],
            compressor_name=payload["compressor_name"],
            feature_k_value=payload["feature_k_values"][1],
        )
        outgoing = dict(payload)
        outgoing["activation_comp"] = hidden_comp
        outgoing["tp_original_bytes"] = list(payload["tp_original_bytes"])
        outgoing["tp_original_bytes"][1] = stats["original_bytes"]
        outgoing["tp_compressed_bytes"] = list(payload["tp_compressed_bytes"])
        outgoing["tp_compressed_bytes"][1] = stats["compressed_bytes"]
        prepare_sec = time.perf_counter() - t_prepare
    else:
        t_restore = time.perf_counter()
        hidden = restore_activation_payload(payload["activation_comp"], device)
        position_bias = payload.get("position_bias")
        if position_bias is not None:
            position_bias = position_bias.to(device)
        attention_mask = payload.get("encoder_attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        restore_sec = time.perf_counter() - t_restore
        t_comp = time.perf_counter()
        with torch.no_grad():
            hidden_out, next_position_bias = runtime["partition"](
                hidden,
                attention_mask=attention_mask,
                position_bias=position_bias,
            )
        compute_sec = time.perf_counter() - t_comp
        t_prepare = time.perf_counter()
        hidden_comp, stats = build_activation_payload(
            hidden_out,
            compression_param=payload["compression_params_list"][1],
            compressor_name=payload["compressor_name"],
            feature_k_value=payload["feature_k_values"][1],
        )
        outgoing = dict(payload)
        outgoing["activation_comp"] = hidden_comp
        outgoing["position_bias"] = None if next_position_bias is None else next_position_bias.detach().cpu()
        outgoing["tp_original_bytes"] = list(payload["tp_original_bytes"])
        outgoing["tp_original_bytes"][1] = stats["original_bytes"]
        outgoing["tp_compressed_bytes"] = list(payload["tp_compressed_bytes"])
        outgoing["tp_compressed_bytes"][1] = stats["compressed_bytes"]
        prepare_sec = time.perf_counter() - t_prepare
    outgoing["worker_stats_chain"] = dict(payload.get("worker_stats_chain", {}))
    outgoing["worker_stats_chain"]["A"] = _receiver_side_tp_stats(
        envelope,
        "A",
        profile=payload.get("worker_stats_chain", {}).get("A", {}).get("profile"),
        request_id=payload["request_id"],
        logical_task_id=payload["logical_task_id"],
    )
    outgoing["worker_stats_chain"]["B"] = {
        "request_id": payload["request_id"],
        "logical_task_id": int(payload["logical_task_id"]),
        "node_id": "B",
        "profile": _worker_profile("B", compute_sec, restore_sec, prepare_sec, 0, 0.0),
    }
    handshake_meta = _build_handshake_meta(
        request_id=payload["request_id"],
        logical_task_id=int(payload["logical_task_id"]),
        link_idx=1,
        expected_payload_bytes=len(pickle.dumps(outgoing, protocol=pickle.HIGHEST_PROTOCOL)),
        stage_msg_type=MessageType.ENCODER_OUTPUT,
        task_family=family,
        src_node="B",
        dst_node="C",
    )
    bytes_sent, send_sec = sender.send(
        "C",
        MessageType.ENCODER_OUTPUT,
        outgoing,
        logical_task_id=int(payload["logical_task_id"]),
        link_idx=1,
        rate_bps=float(payload["link_rate_limits_bps"][1]),
        handshake_meta=handshake_meta,
    )
    logging.info(
        "[B] Forwarded request=%s task=%s -> C bytes=%s send_sec=%.4f",
        payload.get("request_id"),
        int(payload["logical_task_id"]),
        int(bytes_sent),
        float(send_sec),
    )


def _run_worker_c_task(sender, runtime_map, envelope, device):
    payload = envelope["payload"]
    runtime = runtime_map[int(payload["logical_task_id"])]
    family = payload["task_family"]
    logging.info(
        "[C] Received request=%s task=%s family=%s batch=%s eta=%s",
        payload.get("request_id"),
        int(payload["logical_task_id"]),
        family,
        int(payload.get("batch_idx", -1)),
        payload.get("feature_k_values"),
    )
    upstream_stats = _receiver_side_tp_stats(
        envelope,
        "B",
        profile=payload.get("worker_stats_chain", {}).get("B", {}).get("profile"),
        request_id=payload["request_id"],
        logical_task_id=payload["logical_task_id"],
    )
    if family == "resnet":
        t_restore = time.perf_counter()
        hidden = restore_activation_payload(payload["activation_comp"], device)
        restore_sec = time.perf_counter() - t_restore
        t_comp = time.perf_counter()
        with torch.no_grad():
            hidden_out = runtime["partition"](hidden)
        compute_sec = time.perf_counter() - t_comp
        t_prepare = time.perf_counter()
        hidden_comp, stats = build_activation_payload(
            hidden_out,
            compression_param=payload["compression_params_list"][2],
            compressor_name=payload["compressor_name"],
            feature_k_value=payload["feature_k_values"][2],
        )
        outgoing = dict(payload)
        outgoing["activation_comp"] = hidden_comp
        outgoing["tp_original_bytes"] = list(payload["tp_original_bytes"])
        outgoing["tp_original_bytes"][2] = stats["original_bytes"]
        outgoing["tp_compressed_bytes"] = list(payload["tp_compressed_bytes"])
        outgoing["tp_compressed_bytes"][2] = stats["compressed_bytes"]
        prepare_sec = time.perf_counter() - t_prepare
    else:
        t_restore = time.perf_counter()
        hidden = restore_activation_payload(payload["activation_comp"], device)
        position_bias = payload.get("position_bias")
        if position_bias is not None:
            position_bias = position_bias.to(device)
        attention_mask = payload.get("encoder_attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        restore_sec = time.perf_counter() - t_restore
        t_comp = time.perf_counter()
        with torch.no_grad():
            hidden_out, _ = runtime["partition"](
                hidden,
                attention_mask=attention_mask,
                position_bias=position_bias,
            )
        compute_sec = time.perf_counter() - t_comp
        t_prepare = time.perf_counter()
        hidden_comp, stats = build_activation_payload(
            hidden_out,
            compression_param=payload["compression_params_list"][2],
            compressor_name=payload["compressor_name"],
            feature_k_value=payload["feature_k_values"][2],
        )
        outgoing = dict(payload)
        outgoing["activation_comp"] = hidden_comp
        outgoing["position_bias"] = None
        outgoing["tp_original_bytes"] = list(payload["tp_original_bytes"])
        outgoing["tp_original_bytes"][2] = stats["original_bytes"]
        outgoing["tp_compressed_bytes"] = list(payload["tp_compressed_bytes"])
        outgoing["tp_compressed_bytes"][2] = stats["compressed_bytes"]
        prepare_sec = time.perf_counter() - t_prepare
    outgoing["worker_stats_chain"] = dict(payload.get("worker_stats_chain", {}))
    outgoing["worker_stats_chain"]["B"] = upstream_stats
    outgoing["worker_stats_chain"]["C"] = {
        "request_id": payload["request_id"],
        "logical_task_id": int(payload["logical_task_id"]),
        "node_id": "C",
        "profile": _worker_profile("C", compute_sec, restore_sec, prepare_sec, 0, 0.0),
    }
    handshake_meta = _build_handshake_meta(
        request_id=payload["request_id"],
        logical_task_id=int(payload["logical_task_id"]),
        link_idx=2,
        expected_payload_bytes=len(pickle.dumps(outgoing, protocol=pickle.HIGHEST_PROTOCOL)),
        stage_msg_type=MessageType.DECODER_STEP,
        task_family=family,
        src_node="C",
        dst_node="D",
    )
    bytes_sent, send_sec = sender.send(
        "D",
        MessageType.DECODER_STEP,
        outgoing,
        logical_task_id=int(payload["logical_task_id"]),
        link_idx=2,
        rate_bps=float(payload["link_rate_limits_bps"][2]),
        handshake_meta=handshake_meta,
    )
    logging.info(
        "[C] Forwarded request=%s task=%s -> D bytes=%s send_sec=%.4f",
        payload.get("request_id"),
        int(payload["logical_task_id"]),
        int(bytes_sent),
        float(send_sec),
    )


def _run_worker_d_task(sender, runtime_map, envelope, device):
    payload = envelope["payload"]
    runtime = runtime_map[int(payload["logical_task_id"])]
    upstream_stats = _receiver_side_tp_stats(
        envelope,
        "C",
        profile=payload.get("worker_stats_chain", {}).get("C", {}).get("profile"),
        request_id=payload["request_id"],
        logical_task_id=payload["logical_task_id"],
    )
    logging.info(
        "[D] Received request=%s task=%s family=%s batch=%s eta=%s",
        payload.get("request_id"),
        int(payload["logical_task_id"]),
        payload["task_family"],
        int(payload.get("batch_idx", -1)),
        payload.get("feature_k_values"),
    )
    if payload["task_family"] == "resnet":
        t_restore = time.perf_counter()
        hidden = restore_activation_payload(payload["activation_comp"], device)
        restore_sec = time.perf_counter() - t_restore
        t_comp = time.perf_counter()
        with torch.no_grad():
            logits = runtime["partition"](hidden)
        compute_sec = time.perf_counter() - t_comp
        t_prepare = time.perf_counter()
        result = {
            "request_id": payload["request_id"],
            "logical_task_id": int(payload["logical_task_id"]),
            "task_family": "resnet",
            "batch_idx": int(payload["batch_idx"]),
            "requested_eta": list(payload["requested_eta"]),
            "executed_eta": list(payload["feature_k_values"]),
            "compressor_name": payload["compressor_name"],
            "compression_params_list": list(payload["compression_params_list"]),
            "mapping_info": payload.get("mapping_info"),
            "predictions": torch.argmax(logits, dim=1).detach().cpu().tolist(),
            "tp_original_bytes": list(payload["tp_original_bytes"]),
            "tp_compressed_bytes": list(payload["tp_compressed_bytes"]),
            "worker_stats_chain": dict(payload.get("worker_stats_chain", {})),
        }
        prepare_sec = time.perf_counter() - t_prepare
    else:
        t_restore = time.perf_counter()
        encoder_hidden = restore_activation_payload(payload["activation_comp"], device)
        attention_mask = payload.get("encoder_attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        decoder_input_ids = torch.tensor([[int(payload["decoder_start_token_id"])]], dtype=torch.long, device=device)
        restore_sec = time.perf_counter() - t_restore
        t_comp = time.perf_counter()
        with torch.no_grad():
            logits, _, _ = runtime["partition"](
                decoder_input_ids,
                encoder_hidden,
                payload["request_id"],
                encoder_attention_mask=attention_mask,
                position_bias=None,
                encoder_decoder_position_bias=None,
            )
        compute_sec = time.perf_counter() - t_comp
        if logits.dim() == 3:
            token_logits = logits[:, -1, :]
        elif logits.dim() == 2:
            token_logits = logits
        else:
            raise RuntimeError("Unexpected logits shape at node D: {}".format(tuple(logits.shape)))
        positive_id = int(payload["positive_id"])
        negative_id = int(payload["negative_id"])
        positive_logit = float(token_logits[0, positive_id].item())
        negative_logit = float(token_logits[0, negative_id].item())
        predicted_label = 1 if positive_logit >= negative_logit else 0
        t_prepare = time.perf_counter()
        result = {
            "request_id": payload["request_id"],
            "logical_task_id": int(payload["logical_task_id"]),
            "task_family": "flan_t5",
            "batch_idx": int(payload["batch_idx"]),
            "sample_id": payload["sample_id"],
            "requested_eta": list(payload["requested_eta"]),
            "executed_eta": list(payload["feature_k_values"]),
            "compressor_name": payload["compressor_name"],
            "compression_params_list": list(payload["compression_params_list"]),
            "mapping_info": payload.get("mapping_info"),
            "predicted_label": int(predicted_label),
            "positive_logit": positive_logit,
            "negative_logit": negative_logit,
            "tp_original_bytes": list(payload["tp_original_bytes"]),
            "tp_compressed_bytes": list(payload["tp_compressed_bytes"]),
            "worker_stats_chain": dict(payload.get("worker_stats_chain", {})),
        }
        prepare_sec = time.perf_counter() - t_prepare
        runtime["partition"].clear_cache(payload["request_id"])
    result["worker_stats_chain"]["C"] = upstream_stats
    result["worker_stats_chain"]["D"] = {
        "request_id": payload["request_id"],
        "logical_task_id": int(payload["logical_task_id"]),
        "node_id": "D",
        "profile": _worker_profile("D", compute_sec, restore_sec, prepare_sec, 0, 0.0),
    }
    sender.send("A", MessageType.FINAL_RESULT, result)
    logging.info(
        "[D] Completed request=%s task=%s family=%s",
        payload.get("request_id"),
        int(payload["logical_task_id"]),
        payload["task_family"],
    )


def _worker_loop(node_id, transport, sender, runtime_map, device, worker_threads):
    msg_type = {
        "B": MessageType.TASK_INPUT,
        "C": MessageType.ENCODER_OUTPUT,
        "D": MessageType.DECODER_STEP,
    }[node_id]
    handler = {
        "B": _run_worker_b_task,
        "C": _run_worker_c_task,
        "D": _run_worker_d_task,
    }[node_id]
    task_ids = sorted(int(task_id) for task_id in runtime_map.keys())
    if int(worker_threads) != 1:
        logging.info(
            "[%s] worker_threads=%s ignored; using one dedicated worker thread per logical task=%s",
            node_id,
            int(worker_threads),
            task_ids,
        )

    sentinel = object()
    task_queues = {
        int(task_id): queue.Queue()
        for task_id in task_ids
    }
    worker_threads_by_task = {}

    def _consume_task_queue(task_id):
        task_queue = task_queues[int(task_id)]
        while True:
            item = task_queue.get()
            try:
                if item is sentinel:
                    break
                handler(sender, runtime_map, item, device)
            except Exception:
                payload = {}
                try:
                    payload = dict(item.get("payload", {})) if isinstance(item, dict) else {}
                except Exception:
                    payload = {}
                logging.exception(
                    "[%s] Dedicated worker thread failed task=%s request=%s batch=%s",
                    node_id,
                    int(task_id),
                    payload.get("request_id"),
                    payload.get("batch_idx"),
                )
            finally:
                task_queue.task_done()

    for task_id in task_ids:
        thread = threading.Thread(
            target=_consume_task_queue,
            args=(int(task_id),),
            daemon=True,
            name="{}_task_{}".format(node_id, int(task_id)),
        )
        thread.start()
        worker_threads_by_task[int(task_id)] = thread

    try:
        while True:
            envelope = transport.receive(msg_type=msg_type, source=None)
            payload = envelope["payload"]
            if payload.get("cmd") == "shutdown":
                break
            logical_task_id = int(payload["logical_task_id"])
            task_queue = task_queues.get(int(logical_task_id))
            if task_queue is None:
                logging.warning(
                    "[%s] Received envelope for unknown logical_task_id=%s msg_type=%s request=%s",
                    node_id,
                    int(logical_task_id),
                    int(msg_type),
                    payload.get("request_id"),
                )
                continue
            task_queue.put(envelope)
    finally:
        for task_id in task_ids:
            task_queues[int(task_id)].put(sentinel)
        for task_id in task_ids:
            worker_threads_by_task[int(task_id)].join(timeout=2.0)


def run_worker(
    node_id,
    runtime_config,
    config_base_dir,
    device=DEFAULT_DEVICE,
    transport_backend=TRANSPORT_BACKEND,
    tx_bucket_capacity_bytes=DEFAULT_TX_BUCKET_CAPACITY_BYTES,
    worker_threads=DEFAULT_WORKER_THREADS,
):
    if node_id == "A":
        raise ValueError("run_worker only supports B/C/D")
    tasks = _build_worker_task_defs(runtime_config)
    runtime_map = _build_runtime_map(tasks, node_id=node_id, device=device)
    transport = _create_multitask_transport(
        node_id=node_id,
        transport_backend=transport_backend,
        tx_bucket_capacity_bytes=tx_bucket_capacity_bytes,
    )
    transport.start()
    if node_id == "B":
        transport.wait_for_peers(["A", "C"], timeout=TRANSPORT_READY_TIMEOUT_SEC)
    elif node_id == "C":
        transport.wait_for_peers(["A", "B", "D"], timeout=TRANSPORT_READY_TIMEOUT_SEC)
    else:
        transport.wait_for_peers(["A", "C"], timeout=TRANSPORT_READY_TIMEOUT_SEC)
    sender = ThreadSafeTransportSender(transport)
    logging.info("[%s] Worker ready for %d logical tasks", node_id, len(runtime_map))
    try:
        _worker_loop(node_id, transport, sender, runtime_map, device, worker_threads)
    finally:
        transport.close()


def _send_shutdown(transport):
    for peer_id, msg_type in [("B", MessageType.TASK_INPUT), ("C", MessageType.ENCODER_OUTPUT), ("D", MessageType.DECODER_STEP)]:
        try:
            transport.send(peer_id, msg_type, {"cmd": "shutdown"})
        except Exception:
            continue


def _create_multitask_transport(
    node_id,
    transport_backend,
    tx_bucket_capacity_bytes,
):
    backend = str(transport_backend).strip().lower()
    if backend != "http_dynamic":
        raise ValueError(
            "jetson_multi_task_pipeline_async.py only supports transport_backend='http_dynamic', got '{}'".format(
                transport_backend
            )
        )
    return create_transport_dynamic(
        node_id,
        NODE_IPS,
        NODE_PORTS,
        tx_bucket_capacity_bytes=tx_bucket_capacity_bytes,
    )


def _default_algorithms(mu_values):
    algorithms = []
    for mu in _order_mu_values_for_multitask(mu_values):
        algorithms.append(
            {
                "name": "No-CSI (Alg2) (mu={})".format(mu),
                "type": "no_csi_multi",
                "mu": float(mu),
                "epsilon": DEFAULT_EPSILON,
                "J": DEFAULT_J,
            }
        )
    algorithms.extend(
        [
            {"name": "No-CSI Baseline: moving average", "type": "certainty_equivalence_multi_baseline"},
            {"name": "Baseline: no compression", "type": "no_compression_multi_baseline"},
            {"name": "Baseline: max compression", "type": "max_compression_multi_baseline"},
            {"name": "Baseline: equal share", "type": "equal_share_multi_baseline"},
        ]
    )
    for mu in _order_mu_values_for_multitask(mu_values):
        algorithms.append(
            {
                "name": "Baseline: decoupled equal split (mu={})".format(mu),
                "type": "decoupled_equal_split_multi_baseline",
                "mu": float(mu),
                "epsilon": DEFAULT_EPSILON,
            }
        )
        algorithms.append(
            {
                "name": "Baseline: queue proportional (mu={})".format(mu),
                "type": "queue_proportional_multi_baseline",
                "mu": float(mu),
                "epsilon": DEFAULT_EPSILON,
            }
        )
    return algorithms


def _expand_algorithms(manifest, mu_values):
    if "algorithms" not in manifest or not manifest["algorithms"]:
        return _default_algorithms(mu_values)
    ordered_mu_values = _order_mu_values_for_multitask(mu_values)
    expanded = []
    for algo_cfg in manifest["algorithms"]:
        if str(algo_cfg.get("type", "")).lower() != "no_csi_multi":
            expanded.append(copy.deepcopy(algo_cfg))
            continue
        if "mu" in algo_cfg:
            expanded.append(copy.deepcopy(algo_cfg))
            continue
        for mu in ordered_mu_values:
            item = copy.deepcopy(algo_cfg)
            item["mu"] = float(mu)
            name = str(item.get("name", "No-CSI (Alg2)"))
            item["name"] = "{} (mu={})".format(name, mu) if "mu=" not in name else name
            expanded.append(item)
    return expanded


def _run_codec_experiment(
    transport,
    sender,
    runtime_config,
    config_base_dir,
    codec_name,
    accuracy_estimator_mode,
    device,
    output_root,
    tx_limit_mbps,
    max_inflight_per_task,
    channel_predictor_mode,
    channel_model_type,
    channel_window_size,
    channel_probe_steps,
    stein_sigma,
    stein_N,
    stein_fast_max_batches,
    stein_fast_max_samples,
):
    tasks = _build_task_defs(
        config_base_dir,
        runtime_config,
        codec_name=codec_name,
        device=device,
        accuracy_estimator_mode=accuracy_estimator_mode,
        stein_sigma=stein_sigma,
        stein_N=stein_N,
        stein_fast_max_batches=stein_fast_max_batches,
        stein_fast_max_samples=stein_fast_max_samples,
    )
    runtime_map = _build_runtime_map(tasks, node_id="A", device=device)
    task_ids = [int(task.logical_task_id) for task in tasks]
    tasks_by_id = {int(task.logical_task_id): task for task in tasks}
    codec_dir = os.path.join(output_root, _sanitize_tag(codec_name))
    estimator_dir = os.path.join(codec_dir, _sanitize_tag(accuracy_estimator_mode))
    os.makedirs(estimator_dir, exist_ok=True)
    initial_per_link_bps = _per_link_bandwidth_bps(tx_limit_mbps)
    configured_total_bw_bps = np.full(NUM_TRANSFER_POINTS, float(initial_per_link_bps), dtype=float)

    probe_task_id = int(task_ids[0])
    probe_task = tasks_by_id[probe_task_id]
    probe_runtime_map = {probe_task_id: runtime_map[probe_task_id]}
    probe_shared_bandwidth = SharedBandwidthState(
        task_ids=[probe_task_id],
        num_links=NUM_TRANSFER_POINTS,
        initial_total_bps=initial_per_link_bps,
    )
    probe_allocations = {
        probe_task_id: {
            "eta": np.ones(NUM_TRANSFER_POINTS, dtype=float),
            "s_comm": np.ones(NUM_TRANSFER_POINTS, dtype=float),
            "s_comp": np.ones(NUM_NODES, dtype=float),
        }
    }
    probe_items = {
        probe_task_id: deque(list(tasks_by_id[probe_task_id].warmup_items[: int(channel_probe_steps)]))
    }
    logging.info(
        "[A][%s] Channel probe start: task=%s steps=%s",
        "{}|{}".format(codec_name, accuracy_estimator_mode),
        probe_task_id,
        int(len(probe_items[probe_task_id])),
    )
    logging.info(
        "[A][%s] Channel probe config: fixed_per_link_bw_bps=%s max_inflight_per_task=%s",
        "{}|{}".format(codec_name, accuracy_estimator_mode),
        [float(x) for x in np.asarray(probe_shared_bandwidth.total_bandwidth_bps(), dtype=float)],
        4,
    )
    channel_probe_result = _run_stage_round_robin(
        transport,
        sender,
        [probe_task],
        probe_runtime_map,
        probe_items,
        probe_allocations,
        probe_shared_bandwidth,
        max_inflight_per_task=4,
        device=device,
        policy=None,
        algorithm_name="Channel probe",
        algorithm_type="channel_probe",
        terminate_on_task_id=probe_task_id,
        loop_non_terminating_tasks=False,
        record_every_completion=False,
        freeze_bandwidth_during_stage=True,
        tx_limit_mbps_for_warning=tx_limit_mbps,
    )
    channel_probe_history = list(channel_probe_result.get("total_tp_stats_history", []))
    _save_rows(
        os.path.join(estimator_dir, "channel_probe_tp_stats.csv"),
        _tp_stats_rows_from_history(channel_probe_history),
    )
    probe_mean_total_bw_bps = _probe_mean_total_bandwidth_bps(
        channel_probe_history,
        fallback_total_bps=initial_per_link_bps,
    )
    _warn_if_per_link_bandwidth_low(
        probe_mean_total_bw_bps,
        tx_limit_mbps=tx_limit_mbps,
        context_label="{}|{}:channel_probe_mean".format(codec_name, accuracy_estimator_mode),
    )
    if channel_probe_history:
        logging.info(
            "[A][%s] Channel probe complete: mean_per_link_bw_bps=%s from %s task1-only samples",
            "{}|{}".format(codec_name, accuracy_estimator_mode),
            [float(x) for x in np.asarray(probe_mean_total_bw_bps, dtype=float)],
            int(len(channel_probe_history)),
        )
    else:
        logging.warning(
            "[A][%s] Channel probe produced no observations; falling back to fixed initial bandwidth=%s",
            "{}|{}".format(codec_name, accuracy_estimator_mode),
            [float(x) for x in np.asarray(probe_mean_total_bw_bps, dtype=float)],
        )
    logging.info(
        "[A][%s] Policy initialization bandwidth source: using configured per-link limit=%s for warmup and adaptive policy initialization",
        "{}|{}".format(codec_name, accuracy_estimator_mode),
        [float(x) for x in np.asarray(configured_total_bw_bps, dtype=float)],
    )

    shared_bandwidth = SharedBandwidthState(
        task_ids=task_ids,
        num_links=NUM_TRANSFER_POINTS,
        initial_total_bps=initial_per_link_bps,
    )
    warmup_allocations = {
        tid: {
            "eta": np.ones(NUM_TRANSFER_POINTS, dtype=float),
            "s_comm": np.full(NUM_TRANSFER_POINTS, 1.0 / max(len(task_ids), 1), dtype=float),
            "s_comp": np.full(NUM_NODES, 1.0 / max(len(task_ids), 1), dtype=float),
        }
        for tid in task_ids
    }
    warmup_policy = NoCompressionMultiTaskBaseline(tasks)
    warmup_items = {tid: deque(tasks_by_id[tid].warmup_items) for tid in task_ids}
    warmup_result = _run_stage_round_robin(
        transport,
        sender,
        tasks,
        runtime_map,
        warmup_items,
        warmup_allocations,
        shared_bandwidth,
        max_inflight_per_task=max_inflight_per_task,
        device=device,
        policy=warmup_policy,
        algorithm_name="Warmup: no compression",
        algorithm_type="no_compression_multi_baseline",
        terminate_on_task_id=None,
        loop_non_terminating_tasks=False,
        record_every_completion=True,
        channel_estimator=None,
        fixed_stage_total_bw_bps=configured_total_bw_bps,
        tx_limit_mbps_for_warning=tx_limit_mbps,
    )
    warmup_completion_rows = warmup_result["completion_rows"]
    warmup_delay_by_task = {}
    for tid in task_ids:
        delays = [row["delay_sec"] for row in warmup_completion_rows if int(row["logical_task_id"]) == tid]
        warmup_delay_by_task[tid] = float(np.median(delays)) if delays else 0.0
    anchor_task_id = max(task_ids, key=lambda tid: warmup_delay_by_task.get(tid, 0.0))
    logging.info(
        "[A][%s] Warmup complete. anchor_task_id=%s warmup_delay_by_task=%s",
        "{}|{}".format(codec_name, accuracy_estimator_mode),
        anchor_task_id,
        {int(k): _format_float4(v) for k, v in warmup_delay_by_task.items()},
    )
    logging.info(
        "[A][%s] Warmup semantics: full independent no-compression run completed before adaptive policies",
        "{}|{}".format(codec_name, accuracy_estimator_mode),
    )
    logging.info(
        "[A][%s] Initial shared bandwidth semantics: per_link_limit=%.4f Mbps on each of %s links",
        "{}|{}".format(codec_name, accuracy_estimator_mode),
        float(tx_limit_mbps),
        int(NUM_TRANSFER_POINTS),
    )

    baseline_target_time_by_task = {
        int(task.logical_task_id): float(warmup_delay_by_task.get(int(task.logical_task_id), 0.0))
        for task in tasks
    }
    dynamic_target_rate_by_task = {}
    for task in tasks:
        target_time_sec = float(baseline_target_time_by_task[int(task.logical_task_id)])
        task.target_rate_hz = (1.0 / target_time_sec) if target_time_sec > 0.0 else float(task.target_rate_hz)
        dynamic_target_rate_by_task[int(task.logical_task_id)] = float(task.target_rate_hz)
    logging.info(
        "[A][%s] Baseline target times from no-compression medians (sec)=%s -> target rates=%s",
        "{}|{}".format(codec_name, accuracy_estimator_mode),
        {int(k): _format_float4(v) for k, v in baseline_target_time_by_task.items()},
        {int(k): _format_float4(v) for k, v in dynamic_target_rate_by_task.items()},
    )
    warmup_policy_rows = _retarget_control_records(
        warmup_result.get("control_records", []),
        tasks,
        algorithm_name="Baseline: no compression",
        algorithm_type="no_compression_multi_baseline",
        anchor_task_id=anchor_task_id,
    )
    if str(channel_predictor_mode).strip().lower() != "online":
        logging.info(
            "[A][%s] Overriding channel_predictor_mode=%s -> online for multi-task experiment",
            "{}|{}".format(codec_name, accuracy_estimator_mode),
            channel_predictor_mode,
        )

    mu_values = _parse_mu_values(runtime_config.get("mu_values", DEFAULT_MU_VALUES))
    all_algorithms = _expand_algorithms(runtime_config, mu_values)
    skipped_online_algorithm_types = {
        "no_compression_multi_baseline",
        "csi_aware_multi",
        "proportional_multi_baseline",
        "strict_priority_multi_baseline",
    }
    runnable_algorithms = [
        cfg for cfg in all_algorithms
        if str(cfg["type"]).lower() not in skipped_online_algorithm_types
    ]
    all_rows = []
    logging.info(
        "[A][%s] Using configured per-link bandwidth as c_hat basis for adaptive policies: %s",
        "{}|{}".format(codec_name, accuracy_estimator_mode),
        [float(x) for x in np.asarray(configured_total_bw_bps, dtype=float)],
    )
    target_ratio_groups = _resolve_target_ratio_groups(runtime_config, tasks)
    dynamic_timeslot_sizes = _parse_dynamic_timeslot_sizes(
        runtime_config.get("dynamic_timeslot_sizes", DEFAULT_DYNAMIC_TIMESLOT_SIZES)
    )
    max_dynamic_timeslot_count = runtime_config.get("max_dynamic_timeslot_count", DEFAULT_MAX_DYNAMIC_TIMESLOT_COUNT)
    max_total_batches = runtime_config.get("max_total_batches")
    policy_summaries = []
    lastslot_policy_summaries = []
    policy_summaries_by_scope = {}
    lastslot_policy_summaries_by_scope = {}
    for ratio_by_task in target_ratio_groups:
        per_target_rate_by_task = {}
        for task in tasks:
            warmup_delay = float(warmup_delay_by_task.get(int(task.logical_task_id), 0.0))
            target_ratio = float(ratio_by_task[int(task.logical_task_id)])
            target_time_sec = warmup_delay * float(target_ratio) if warmup_delay > 0.0 else 0.0
            task.target_rate_hz = (1.0 / target_time_sec) if target_time_sec > 0.0 else float(task.target_rate_hz)
            per_target_rate_by_task[int(task.logical_task_id)] = float(task.target_rate_hz)
        target_label = _format_target_dir_name(ratio_by_task)
        target_dir = os.path.join(estimator_dir, target_label)
        os.makedirs(target_dir, exist_ok=True)
        logging.info(
            "[A][%s] Target group ratios=%s -> target rates=%s output_dir=%s",
            "{}|{}".format(codec_name, accuracy_estimator_mode),
            {int(k): _format_float4(v) for k, v in sorted(ratio_by_task.items())},
            {int(k): _format_float4(v) for k, v in per_target_rate_by_task.items()},
            target_dir,
        )
        for dynamic_timeslot_size in dynamic_timeslot_sizes:
            dynamic_dir = _dynamic_timeslot_output_dir(target_dir, dynamic_timeslot_size)
            os.makedirs(dynamic_dir, exist_ok=True)
            target_plot_context = _write_plot_context(
                dynamic_dir,
                tasks,
                codec_name,
                tx_limit_mbps,
                ratio_by_task,
                accuracy_estimator_mode=accuracy_estimator_mode,
                stein_sigma=stein_sigma,
                stein_N=stein_N,
                dynamic_timeslot_size=int(dynamic_timeslot_size),
                max_dynamic_timeslot_count=max_dynamic_timeslot_count,
                anchor_task_id=anchor_task_id,
            )

            baseline_rows_by_scope = {
                scope_key: _annotate_rows_for_target(
                    rows,
                    ratio_by_task,
                    accuracy_estimator_mode=accuracy_estimator_mode,
                    anchor_task_id=anchor_task_id,
                )
                for scope_key, rows in _build_batch_timeseries_sets(
                    warmup_completion_rows,
                    tasks,
                    codec_name=codec_name,
                    dynamic_timeslot_size=int(dynamic_timeslot_size),
                    algorithm_name="Baseline: no compression",
                    algorithm_type="no_compression_multi_baseline",
                ).items()
            }
            baseline_window_rows_by_scope = {
                scope_key: _annotate_rows_for_target(
                    rows,
                    ratio_by_task,
                    accuracy_estimator_mode=accuracy_estimator_mode,
                    anchor_task_id=anchor_task_id,
                )
                for scope_key, rows in _build_window_summaries_sets(
                    baseline_rows_by_scope,
                    dynamic_timeslot_size=int(dynamic_timeslot_size),
                    anchor_task_id=int(anchor_task_id),
                    tasks=tasks,
                ).items()
            }
            baseline_rows_for_target = list(baseline_rows_by_scope.get("avg", []))
            baseline_window_rows = list(baseline_window_rows_by_scope.get("avg", []))
            for row in baseline_rows_for_target:
                row["dynamic_timeslot_size"] = int(dynamic_timeslot_size)
            baseline_batch_summary_by_scope = {
                scope_key: _build_batch_policy_summary_row(rows)
                for scope_key, rows in baseline_rows_by_scope.items()
            }
            baseline_batch_summary_by_scope = _build_weighted_avg_policy_summaries(
                baseline_batch_summary_by_scope,
                tasks,
            )
            baseline_lastslot_summary_by_scope = {
                scope_key: _build_policy_summary_row(
                    baseline_rows_by_scope.get(scope_key, []),
                    baseline_window_rows_by_scope.get(scope_key, []),
                )
                for scope_key in baseline_rows_by_scope.keys()
            }
            baseline_lastslot_summary_by_scope = _build_weighted_avg_policy_summaries(
                baseline_lastslot_summary_by_scope,
                tasks,
            )
            warmup_tag = _sanitize_tag("Baseline: no compression")
            _save_rows(os.path.join(dynamic_dir, "timeseries_{}.csv".format(warmup_tag)), baseline_rows_for_target)
            _save_rows(os.path.join(dynamic_dir, "window_summary_{}.csv".format(warmup_tag)), baseline_window_rows)
            for scope_key, rows in baseline_rows_by_scope.items():
                _save_rows(os.path.join(dynamic_dir, "timeseries_{}_{}.csv".format(warmup_tag, scope_key)), rows)
            for scope_key, rows in baseline_window_rows_by_scope.items():
                _save_rows(os.path.join(dynamic_dir, "window_summary_{}_{}.csv".format(warmup_tag, scope_key)), rows)
            dynamic_rows = list(baseline_rows_for_target)
            target_summary = [baseline_batch_summary_by_scope["avg"]] if baseline_batch_summary_by_scope.get("avg") is not None else []
            target_lastslot_summary = [baseline_lastslot_summary_by_scope["avg"]] if baseline_lastslot_summary_by_scope.get("avg") is not None else []
            target_summary_by_scope = {
                scope_key: ([summary_row] if summary_row is not None else [])
                for scope_key, summary_row in baseline_batch_summary_by_scope.items()
            }
            target_lastslot_summary_by_scope = {
                scope_key: ([summary_row] if summary_row is not None else [])
                for scope_key, summary_row in baseline_lastslot_summary_by_scope.items()
            }

            for algo_cfg in runnable_algorithms:
                algo_type = str(algo_cfg["type"])
                policy = create_policy(algo_cfg, tasks)
                algo_name = str(algo_cfg.get("name", policy.policy_name))
                logging.info(
                    "[A][%s][%s][dyn=%d] Policy start",
                    "{}|{}".format(codec_name, accuracy_estimator_mode),
                    algo_name,
                    int(dynamic_timeslot_size),
                )
                shared_bandwidth_policy = SharedBandwidthState(
                    task_ids=task_ids,
                    num_links=NUM_TRANSFER_POINTS,
                    initial_total_bps=initial_per_link_bps,
                )
                stage_channel_estimator = None
                stage_fixed_total_bw_bps = None
                if _policy_uses_fixed_channel(policy=policy, algorithm_type=algo_type):
                    stage_fixed_total_bw_bps = configured_total_bw_bps.copy()
                elif _policy_uses_estimator(policy=policy, algorithm_type=algo_type):
                    stage_channel_estimator = build_channel_estimator(
                        device=device,
                        model_type=channel_model_type,
                        update_mode=channel_predictor_mode,
                        window_size=channel_window_size,
                        default_bandwidth_bps=initial_per_link_bps,
                    )
                t_solve = time.perf_counter()
                initial_allocations = policy.current_allocations(
                    warmup_result["latest_snapshots"],
                    (
                        _current_total_bandwidth_bps(stage_channel_estimator, shared_bandwidth_policy)
                        if stage_channel_estimator is not None else
                        configured_total_bw_bps
                    ),
                )
                initial_control_solve_sec = time.perf_counter() - t_solve
                non_mu_policy_completion_limit = (
                    int(DEFAULT_NON_MU_POLICY_COMPLETION_LIMIT)
                    if not _policy_uses_mu(policy=policy, algorithm_type=algo_type)
                    else None
                )
                logging.info(
                    "[A][%s][%s][dyn=%d] Initial allocation solve time: %.4fs",
                    "{}|{}".format(codec_name, accuracy_estimator_mode),
                    algo_name,
                    int(dynamic_timeslot_size),
                    float(initial_control_solve_sec),
                )
                experiment_items = {tid: deque(tasks_by_id[tid].experiment_items) for tid in task_ids}
                trial_result = _run_stage_round_robin(
                    transport,
                    sender,
                    tasks,
                    runtime_map,
                    experiment_items,
                    initial_allocations,
                    shared_bandwidth_policy,
                    max_inflight_per_task=max_inflight_per_task,
                    device=device,
                    policy=policy,
                    anchor_task_id=anchor_task_id,
                    output_dir=dynamic_dir,
                    plot_context=target_plot_context,
                    algorithm_name=algo_name,
                    algorithm_type=algo_type,
                    channel_estimator=stage_channel_estimator,
                    terminate_on_task_id=anchor_task_id,
                    loop_non_terminating_tasks=True,
                    fixed_stage_total_bw_bps=stage_fixed_total_bw_bps,
                    tx_limit_mbps_for_warning=tx_limit_mbps,
                    dynamic_timeslot_size=int(dynamic_timeslot_size),
                    max_dynamic_timeslot_count=max_dynamic_timeslot_count,
                    max_total_batches=max_total_batches,
                    target_completion_count_override=non_mu_policy_completion_limit,
                )
                target_rows_by_scope = {
                    scope_key: _annotate_rows_for_target(
                        rows,
                        ratio_by_task,
                        accuracy_estimator_mode=accuracy_estimator_mode,
                        anchor_task_id=anchor_task_id,
                    )
                    for scope_key, rows in _build_batch_timeseries_sets(
                        trial_result["completion_rows"],
                        tasks,
                        codec_name=codec_name,
                        dynamic_timeslot_size=int(dynamic_timeslot_size),
                        algorithm_name=algo_name,
                        algorithm_type=algo_type,
                    ).items()
                }
                window_rows_by_scope = {
                    scope_key: _annotate_rows_for_target(
                        rows,
                        ratio_by_task,
                        accuracy_estimator_mode=accuracy_estimator_mode,
                        anchor_task_id=anchor_task_id,
                    )
                    for scope_key, rows in _build_window_summaries_sets(
                        target_rows_by_scope,
                        dynamic_timeslot_size=int(dynamic_timeslot_size),
                        anchor_task_id=int(anchor_task_id),
                        tasks=tasks,
                    ).items()
                }
                target_rows = list(target_rows_by_scope.get("avg", []))
                window_rows = list(window_rows_by_scope.get("avg", []))
                for row in target_rows:
                    row["dynamic_timeslot_size"] = int(dynamic_timeslot_size)
                algo_tag = _sanitize_tag(algo_name)
                _save_rows(os.path.join(dynamic_dir, "timeseries_{}.csv".format(algo_tag)), target_rows)
                _save_rows(os.path.join(dynamic_dir, "window_summary_{}.csv".format(algo_tag)), window_rows)
                for scope_key, rows in target_rows_by_scope.items():
                    _save_rows(os.path.join(dynamic_dir, "timeseries_{}_{}.csv".format(algo_tag, scope_key)), rows)
                for scope_key, rows in window_rows_by_scope.items():
                    _save_rows(os.path.join(dynamic_dir, "window_summary_{}_{}.csv".format(algo_tag, scope_key)), rows)
                dynamic_rows.extend(target_rows)
                batch_summary_by_scope = {
                    scope_key: _build_batch_policy_summary_row(rows)
                    for scope_key, rows in target_rows_by_scope.items()
                }
                batch_summary_by_scope = _build_weighted_avg_policy_summaries(
                    batch_summary_by_scope,
                    tasks,
                )
                lastslot_summary_by_scope = {
                    scope_key: _build_policy_summary_row(
                        target_rows_by_scope.get(scope_key, []),
                        window_rows_by_scope.get(scope_key, []),
                    )
                    for scope_key in target_rows_by_scope.keys()
                }
                lastslot_summary_by_scope = _build_weighted_avg_policy_summaries(
                    lastslot_summary_by_scope,
                    tasks,
                )
                if batch_summary_by_scope.get("avg") is not None:
                    target_summary.append(batch_summary_by_scope["avg"])
                if lastslot_summary_by_scope.get("avg") is not None:
                    target_lastslot_summary.append(lastslot_summary_by_scope["avg"])
                for scope_key, summary_row in batch_summary_by_scope.items():
                    if summary_row is not None:
                        target_summary_by_scope.setdefault(scope_key, []).append(summary_row)
                for scope_key, summary_row in lastslot_summary_by_scope.items():
                    if summary_row is not None:
                        target_lastslot_summary_by_scope.setdefault(scope_key, []).append(summary_row)
                logging.info(
                    "[A][%s][%s][dyn=%d] Policy finished. terminated_task_id=%s termination_reason=%s",
                    "{}|{}".format(codec_name, accuracy_estimator_mode),
                    algo_name,
                    int(dynamic_timeslot_size),
                    trial_result.get("terminated_task_id"),
                    trial_result.get("termination_reason"),
                )

            _save_rows(os.path.join(dynamic_dir, "policy_summary.csv"), target_summary)
            _save_rows(os.path.join(dynamic_dir, "summary_lasttimeslot.csv"), target_lastslot_summary)
            for scope_key, rows in target_summary_by_scope.items():
                _save_rows(os.path.join(dynamic_dir, "policy_summary_{}.csv".format(scope_key)), rows)
            for scope_key, rows in target_lastslot_summary_by_scope.items():
                _save_rows(os.path.join(dynamic_dir, "summary_lasttimeslot_{}.csv".format(scope_key)), rows)
            target_run_summary = {
                "codec_name": codec_name,
                "accuracy_estimator_mode": str(accuracy_estimator_mode),
                "target_ratio": float(ratio_by_task[int(anchor_task_id)]),
                "target_ratio_by_task": {int(k): float(v) for k, v in sorted(ratio_by_task.items())},
                "dynamic_timeslot_size": int(dynamic_timeslot_size),
                "max_dynamic_timeslot_count": (
                    None if max_dynamic_timeslot_count is None else int(max_dynamic_timeslot_count)
                ),
                "max_total_batches": (None if max_total_batches is None else int(max_total_batches)),
                "target_time_sec_by_task": _target_time_by_task(tasks),
                "target_rate_by_task_hz": {int(task.logical_task_id): float(task.target_rate_hz) for task in tasks},
                "probe_mean_bandwidth_mbps_per_link": [float(x) / 1e6 for x in np.asarray(probe_mean_total_bw_bps, dtype=float)],
                "warmup_delay_by_task_sec": warmup_delay_by_task,
                "policy_summary": target_summary,
                "summary_lasttimeslot": target_lastslot_summary,
                "accuracy_summary": _aggregate_policy_summary_rows(target_summary),
            }
            with open(os.path.join(dynamic_dir, "run_summary.json"), "w", encoding="utf-8") as handle:
                json.dump(target_run_summary, handle, indent=2)
            policy_summaries.extend(target_summary)
            lastslot_policy_summaries.extend(target_lastslot_summary)
            for scope_key, rows in target_summary_by_scope.items():
                policy_summaries_by_scope.setdefault(scope_key, []).extend(rows)
            for scope_key, rows in target_lastslot_summary_by_scope.items():
                lastslot_policy_summaries_by_scope.setdefault(scope_key, []).extend(rows)
            all_rows.extend(dynamic_rows)
            _maybe_render_plots(dynamic_dir, target_plot_context)

    policy_summary = policy_summaries if policy_summaries else _summarize_policy_rows(all_rows)
    _save_rows(os.path.join(estimator_dir, "policy_summary.csv"), policy_summary)
    _save_rows(os.path.join(estimator_dir, "summary_lasttimeslot.csv"), lastslot_policy_summaries)
    for scope_key, rows in policy_summaries_by_scope.items():
        _save_rows(os.path.join(estimator_dir, "policy_summary_{}.csv".format(scope_key)), rows)
    for scope_key, rows in lastslot_policy_summaries_by_scope.items():
        _save_rows(os.path.join(estimator_dir, "summary_lasttimeslot_{}.csv".format(scope_key)), rows)
    run_summary = {
        "codec_name": codec_name,
        "accuracy_estimator_mode": str(accuracy_estimator_mode),
        "anchor_task_id": int(anchor_task_id),
        "channel_probe_task_id": int(probe_task_id),
        "channel_probe_steps": int(len(channel_probe_history)),
        "probe_mean_bandwidth_mbps_per_link": [float(x) / 1e6 for x in np.asarray(probe_mean_total_bw_bps, dtype=float)],
        "warmup_delay_by_task_sec": warmup_delay_by_task,
        "baseline_target_time_sec_by_task": baseline_target_time_by_task,
        "baseline_target_rate_by_task_hz": dynamic_target_rate_by_task,
        "target_ratio_groups": [
            {int(k): float(v) for k, v in sorted(group.items())} for group in target_ratio_groups
        ],
        "dynamic_timeslot_sizes": [int(x) for x in dynamic_timeslot_sizes],
        "max_dynamic_timeslot_count": (
            None if max_dynamic_timeslot_count is None else int(max_dynamic_timeslot_count)
        ),
        "max_total_batches": (None if max_total_batches is None else int(max_total_batches)),
        "tx_limit_mbps": float(tx_limit_mbps),
        "initial_per_link_bandwidth_mbps": float(tx_limit_mbps),
        "channel_limit_semantics": "per_link",
        "last_terminated_task_id": (
            trial_result.get("terminated_task_id") if 'trial_result' in locals() else None
        ),
        "policy_summary": policy_summary,
        "summary_lasttimeslot": lastslot_policy_summaries,
        "accuracy_summary": _aggregate_policy_summary_rows(policy_summary),
    }
    with open(os.path.join(estimator_dir, "run_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2)
    logging.info(
        "[A][%s] Codec-scoped cleanup start: releasing tasks/runtime/probe/warmup state before next codec or estimator mode",
        "{}|{}".format(codec_name, accuracy_estimator_mode),
    )
    del tasks
    del runtime_map
    del task_ids
    del tasks_by_id
    del probe_task
    del probe_runtime_map
    del probe_shared_bandwidth
    del probe_allocations
    del probe_items
    del channel_probe_result
    del channel_probe_history
    del shared_bandwidth
    del warmup_allocations
    del warmup_policy
    del warmup_items
    del warmup_result
    del warmup_completion_rows
    del warmup_delay_by_task
    del baseline_target_time_by_task
    del dynamic_target_rate_by_task
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return run_summary


def run_node_a(
    runtime_config,
    config_base_dir,
    device=DEFAULT_DEVICE,
    base_output_dir=DEFAULT_OUTPUT_DIR,
    transport_backend=TRANSPORT_BACKEND,
    tx_limit_mbps=DEFAULT_TX_LIMIT_MBPS,
    tx_bucket_capacity_bytes=DEFAULT_TX_BUCKET_CAPACITY_BYTES,
    max_inflight_per_task=DEFAULT_MAX_INFLIGHT_PER_TASK,
    channel_predictor_mode=DEFAULT_CHANNEL_UPDATE_MODE,
    channel_model_type=DEFAULT_CHANNEL_MODEL_TYPE,
    channel_window_size=DEFAULT_CHANNEL_WINDOW_SIZE,
    channel_probe_steps=DEFAULT_CHANNEL_PROBE_STEPS,
    stein_sigma=DEFAULT_STEIN_SIGMA,
    stein_N=DEFAULT_STEIN_N,
    stein_fast_max_batches=DEFAULT_STEIN_FAST_MAX_BATCHES,
    stein_fast_max_samples=DEFAULT_STEIN_FAST_MAX_SAMPLES,
):
    output_root = _make_run_output_dir("A", base_output_dir, device, tx_limit_mbps)
    transport = _create_multitask_transport(
        node_id="A",
        transport_backend=transport_backend,
        tx_bucket_capacity_bytes=tx_bucket_capacity_bytes,
    )
    transport.start()
    transport.wait_for_peers(["B", "C", "D"], timeout=TRANSPORT_READY_TIMEOUT_SEC)
    sender = ThreadSafeTransportSender(transport)
    codec_names = list(runtime_config.get("codec_names", DEFAULT_CODEC_NAMES))
    accuracy_estimator_modes = _parse_accuracy_estimator_modes(
        runtime_config.get("accuracy_estimator_modes", DEFAULT_ACCURACY_ESTIMATOR_MODES)
    )
    run_summary = {
        "experiment": "online",
        "device": device,
        "tx_limit_mbps": float(tx_limit_mbps),
        "channel_limit_semantics": "per_link",
        "codec_runs": [],
    }
    try:
        for codec_name in codec_names:
            for accuracy_estimator_mode in accuracy_estimator_modes:
                codec_summary = _run_codec_experiment(
                    transport,
                    sender,
                    runtime_config,
                    config_base_dir,
                    codec_name=str(codec_name),
                    accuracy_estimator_mode=str(accuracy_estimator_mode),
                    device=device,
                    output_root=output_root,
                    tx_limit_mbps=tx_limit_mbps,
                    max_inflight_per_task=max_inflight_per_task,
                    channel_predictor_mode=channel_predictor_mode,
                    channel_model_type=channel_model_type,
                    channel_window_size=channel_window_size,
                    channel_probe_steps=channel_probe_steps,
                    stein_sigma=stein_sigma,
                    stein_N=stein_N,
                    stein_fast_max_batches=stein_fast_max_batches,
                    stein_fast_max_samples=stein_fast_max_samples,
                )
                run_summary["codec_runs"].append(codec_summary)
    finally:
        _send_shutdown(transport)
        transport.close()
    with open(os.path.join(output_root, "run_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2)
    return output_root


def _build_parser():
    parser = argparse.ArgumentParser(description="Event-driven async mixed ResNet+Flan-T5 multi-task pipeline")
    parser.add_argument("--node", choices=NODE_ORDER, required=True)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--base_output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--compressor_profiles", default="topk,quantization,llmint8")
    parser.add_argument("--accuracy_estimator_modes", default=DEFAULT_ACCURACY_ESTIMATOR_MODES)
    parser.add_argument(
        "--target_ratios",
        default=None,
        help="Optional global fallback target ratio list for all tasks. Ratios are relative to each task's warmup no-compression delay.",
    )
    parser.add_argument(
        "--target_task1_ratios",
        default="0.75",
        help="Target ratio list for task 1. Ratios are relative to task 1 warmup no-compression delay. Default: 0.75",
    )
    parser.add_argument(
        "--target_task2_ratios",
        default="0.9",
        help="Target ratio list for task 2. Ratios are relative to task 2 warmup no-compression delay. Default: 0.9",
    )
    parser.add_argument("--target_time_ratios", dest="target_time_ratios_legacy", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--resnet_target_time_ratios", dest="resnet_target_time_ratios_legacy", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--flan_target_time_ratios", dest="flan_target_time_ratios_legacy", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dynamic_timeslot_sizes", default="5,10,25")
    parser.add_argument("--max_dynamic_timeslot_count", type=int, default=DEFAULT_MAX_DYNAMIC_TIMESLOT_COUNT)
    parser.add_argument("--max_total_batches", type=int, default=None)
    parser.add_argument("--mu_values", default="0.1,0.01")
    parser.add_argument("--warmup_samples_per_task", type=int, default=DEFAULT_WARMUP_SAMPLES_PER_TASK)
    parser.add_argument("--tx_limit_mbps", type=float, default=DEFAULT_TX_LIMIT_MBPS)
    parser.add_argument("--initial_total_bandwidth_mbps", dest="tx_limit_mbps", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--stein_sigma", type=float, default=DEFAULT_STEIN_SIGMA)
    parser.add_argument("--stein_N", type=int, default=DEFAULT_STEIN_N)
    parser.add_argument("--stein_fast_max_batches", type=int, default=DEFAULT_STEIN_FAST_MAX_BATCHES)
    parser.add_argument("--stein_fast_max_samples", type=int, default=DEFAULT_STEIN_FAST_MAX_SAMPLES)
    parser.add_argument("--transport_backend", default=TRANSPORT_BACKEND)
    parser.add_argument("--channel_predictor_mode", default=DEFAULT_CHANNEL_UPDATE_MODE, choices=["warmup", "online"])
    parser.add_argument("--channel_model_type", default=DEFAULT_CHANNEL_MODEL_TYPE, choices=SUPPORTED_CHANNEL_MODEL_TYPES)
    parser.add_argument("--channel_window_size", type=int, default=DEFAULT_CHANNEL_WINDOW_SIZE)
    parser.add_argument("--channel_probe_steps", type=int, default=DEFAULT_CHANNEL_PROBE_STEPS)
    parser.add_argument("--tx_bucket_capacity_kb", type=int, default=DEFAULT_TX_BUCKET_CAPACITY_BYTES // 1024)
    parser.add_argument("--max_inflight_per_task", type=int, default=DEFAULT_MAX_INFLIGHT_PER_TASK)
    parser.add_argument("--worker_threads", type=int, default=DEFAULT_WORKER_THREADS)
    parser.add_argument("--resnet_target_rate_hz", dest="resnet_target_rate_hz_legacy", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--resnet_max_items", type=int, default=DEFAULT_RESNET_MAX_ITEMS)
    parser.add_argument("--resnet_weight", type=float, default=DEFAULT_RESNET_WEIGHT)
    parser.add_argument("--resnet_batch_size", type=int, default=DEFAULT_RESNET_BATCH_SIZE)
    parser.add_argument("--resnet_checkpoint_path", default=RESNET_CHECKPOINT_PATH)
    parser.add_argument("--resnet_data_root", default=RESNET_DEFAULT_DATA_ROOT)
    parser.add_argument("--resnet_download_data", action="store_true", default=RESNET_DEFAULT_DOWNLOAD_DATA)
    parser.add_argument("--resnet_topk_estimator_path", default=DEFAULT_RESNET_TOPK_ESTIMATOR_PATH)
    parser.add_argument("--resnet_quantization_estimator_path", default=DEFAULT_RESNET_QUANTIZATION_ESTIMATOR_PATH)
    parser.add_argument("--resnet_llmint8_estimator_path", default=DEFAULT_RESNET_LLMINT8_ESTIMATOR_PATH)
    parser.add_argument("--resnet_llmint8_mapping_path", default=DEFAULT_RESNET_LLMINT8_MAPPING_PATH)
    parser.add_argument("--resnet_llmint8_outlier_precision", default=DEFAULT_RESNET_OUTLIER_PRECISION)
    parser.add_argument("--resnet_llmint8_regular_precision", default=DEFAULT_RESNET_REGULAR_PRECISION)
    parser.add_argument("--flan_target_rate_hz", dest="flan_target_rate_hz_legacy", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--flan_max_items", type=int, default=DEFAULT_FLAN_MAX_ITEMS)
    parser.add_argument("--flan_weight", type=float, default=DEFAULT_FLAN_WEIGHT)
    parser.add_argument("--flan_batch_size", type=int, default=DEFAULT_FLAN_BATCH_SIZE)
    parser.add_argument("--flan_model_name", default=FLAN_DEFAULT_MODEL_NAME)
    parser.add_argument("--flan_dataset_path", default=DEFAULT_FLAN_DATASET_PATH)
    parser.add_argument("--flan_split", default=FLAN_DEFAULT_SPLIT)
    parser.add_argument("--flan_max_input_length", type=int, default=FLAN_DEFAULT_MAX_INPUT_LENGTH)
    parser.add_argument("--flan_prompt_template", default=FLAN_DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--flan_positive_token", default=FLAN_DEFAULT_POSITIVE_TOKEN)
    parser.add_argument("--flan_negative_token", default=FLAN_DEFAULT_NEGATIVE_TOKEN)
    parser.add_argument("--flan_topk_estimator_path", default=DEFAULT_FLAN_TOPK_ESTIMATOR_PATH)
    parser.add_argument("--flan_quantization_estimator_path", default=DEFAULT_FLAN_QUANTIZATION_ESTIMATOR_PATH)
    parser.add_argument("--flan_llmint8_estimator_path", default=DEFAULT_FLAN_LLMINT8_ESTIMATOR_PATH)
    parser.add_argument("--flan_llmint8_mapping_path", default=DEFAULT_FLAN_LLMINT8_MAPPING_PATH)
    parser.add_argument("--flan_llmint8_outlier_precision", default=DEFAULT_FLAN_OUTLIER_PRECISION)
    parser.add_argument("--flan_llmint8_regular_precision", default=DEFAULT_FLAN_REGULAR_PRECISION)
    parser.add_argument("--log_level", default="INFO")
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    runtime_config = _build_inline_config(args)
    config_base_dir = os.getcwd()
    tx_bucket_capacity_bytes = int(args.tx_bucket_capacity_kb) * 1024
    if args.node == "A":
        run_node_a(
            runtime_config=runtime_config,
            config_base_dir=config_base_dir,
            device=args.device,
            base_output_dir=args.base_output_dir,
            transport_backend=args.transport_backend,
            tx_limit_mbps=float(args.tx_limit_mbps),
            tx_bucket_capacity_bytes=tx_bucket_capacity_bytes,
            max_inflight_per_task=int(args.max_inflight_per_task),
            channel_predictor_mode=args.channel_predictor_mode,
            channel_model_type=args.channel_model_type,
            channel_window_size=int(args.channel_window_size),
            channel_probe_steps=int(args.channel_probe_steps),
            stein_sigma=float(args.stein_sigma),
            stein_N=int(args.stein_N),
            stein_fast_max_batches=int(args.stein_fast_max_batches),
            stein_fast_max_samples=int(args.stein_fast_max_samples),
        )
        return
    run_worker(
        node_id=args.node,
        runtime_config=runtime_config,
        config_base_dir=config_base_dir,
        device=args.device,
        transport_backend=args.transport_backend,
        tx_bucket_capacity_bytes=tx_bucket_capacity_bytes,
        worker_threads=int(args.worker_threads),
    )


if __name__ == "__main__":
    main()
