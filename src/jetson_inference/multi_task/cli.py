"""argument parser and main dispatch; defaults are unchanged."""

import argparse
import logging
import os

from jetson_inference.common.channel_estimator import SUPPORTED_CHANNEL_MODEL_TYPES
from jetson_inference.flan_t5.config import (
    DEFAULT_MAX_INPUT_LENGTH as FLAN_DEFAULT_MAX_INPUT_LENGTH,
    DEFAULT_MODEL_NAME as FLAN_DEFAULT_MODEL_NAME,
    DEFAULT_NEGATIVE_TOKEN as FLAN_DEFAULT_NEGATIVE_TOKEN,
    DEFAULT_POSITIVE_TOKEN as FLAN_DEFAULT_POSITIVE_TOKEN,
    DEFAULT_PROMPT_TEMPLATE as FLAN_DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_SPLIT as FLAN_DEFAULT_SPLIT,
)
from jetson_inference.multi_task.config import (
    DEFAULT_ACCURACY_ESTIMATOR_MODES,
    DEFAULT_CHANNEL_MODEL_TYPE,
    DEFAULT_CHANNEL_PROBE_STEPS,
    DEFAULT_CHANNEL_UPDATE_MODE,
    DEFAULT_CHANNEL_WINDOW_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_FLAN_BATCH_SIZE,
    DEFAULT_FLAN_DATASET_PATH,
    DEFAULT_FLAN_LLMINT8_ESTIMATOR_PATH,
    DEFAULT_FLAN_LLMINT8_MAPPING_PATH,
    DEFAULT_FLAN_MAX_ITEMS,
    DEFAULT_FLAN_OUTLIER_PRECISION,
    DEFAULT_FLAN_QUANTIZATION_ESTIMATOR_PATH,
    DEFAULT_FLAN_REGULAR_PRECISION,
    DEFAULT_FLAN_TOPK_ESTIMATOR_PATH,
    DEFAULT_FLAN_WEIGHT,
    DEFAULT_MAX_DYNAMIC_TIMESLOT_COUNT,
    DEFAULT_MAX_INFLIGHT_PER_TASK,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RESNET_BATCH_SIZE,
    DEFAULT_RESNET_LLMINT8_ESTIMATOR_PATH,
    DEFAULT_RESNET_LLMINT8_MAPPING_PATH,
    DEFAULT_RESNET_MAX_ITEMS,
    DEFAULT_RESNET_OUTLIER_PRECISION,
    DEFAULT_RESNET_QUANTIZATION_ESTIMATOR_PATH,
    DEFAULT_RESNET_REGULAR_PRECISION,
    DEFAULT_RESNET_TOPK_ESTIMATOR_PATH,
    DEFAULT_RESNET_WEIGHT,
    DEFAULT_STEIN_FAST_MAX_BATCHES,
    DEFAULT_STEIN_FAST_MAX_SAMPLES,
    DEFAULT_STEIN_N,
    DEFAULT_STEIN_SIGMA,
    DEFAULT_TX_BUCKET_CAPACITY_BYTES,
    DEFAULT_TX_LIMIT_MBPS,
    DEFAULT_WARMUP_SAMPLES_PER_TASK,
    DEFAULT_WORKER_THREADS,
    NODE_ORDER,
    TRANSPORT_BACKEND,
    _build_inline_config,
)
from jetson_inference.multi_task.experiment import run_node_a
from jetson_inference.multi_task.workers import run_worker
from jetson_inference.resnet.config import (
    DEFAULT_DATA_ROOT as RESNET_DEFAULT_DATA_ROOT,
    DEFAULT_DOWNLOAD_DATA as RESNET_DEFAULT_DOWNLOAD_DATA,
    MODEL_CHECKPOINT as RESNET_CHECKPOINT_PATH,
)


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
