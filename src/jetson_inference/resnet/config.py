"""argument normalization and experiment configuration assembly."""

import datetime
import json
import os
import pandas as pd
import torch

from collections import OrderedDict


from jetson_inference.paths import PROJECT_ROOT

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

def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)

def _accuracy_from_predictions(predictions, labels):
    pred_tensor = torch.as_tensor(predictions, dtype=torch.long)
    labels = labels.detach().cpu().to(torch.long)
    correct = int((pred_tensor == labels).sum().item())
    total = int(labels.numel())
    return correct, total, (float(correct) / float(total) if total > 0 else 0.0)

def _dynamic_timeslot_output_dir(base_dir, dynamic_timeslot_size):
    path = os.path.join(base_dir, "dynamic_timeslot_{}".format(int(dynamic_timeslot_size)))
    os.makedirs(path, exist_ok=True)
    return path
