"""argument normalization and experiment configuration assembly."""

import copy
import datetime
import json
import os
import pandas as pd

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
