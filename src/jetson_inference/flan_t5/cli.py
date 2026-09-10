"""argument parser and main dispatch; defaults are unchanged."""

import argparse
import logging

from jetson_inference.common.channel_estimator import SUPPORTED_CHANNEL_MODEL_TYPES
from jetson_inference.flan_t5.config import (
    DEFAULT_ACCURACY_ESTIMATOR_MODES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHANNEL_MODEL_TYPE,
    DEFAULT_CHANNEL_UPDATE_MODE,
    DEFAULT_CHANNEL_WINDOW_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_EPSILON,
    DEFAULT_LOCAL_DATASET_PATH,
    DEFAULT_MAX_DYNAMIC_TIMESLOT_COUNT,
    DEFAULT_MAX_INPUT_LENGTH,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_MODEL_NAME,
    DEFAULT_MU_VALUES,
    DEFAULT_NEGATIVE_TOKEN,
    DEFAULT_POSITIVE_TOKEN,
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_SPLIT,
    DEFAULT_STEIN_FAST_MAX_SAMPLES,
    DEFAULT_STEIN_N,
    DEFAULT_STEIN_SIGMA,
    DEFAULT_T0_RATIO,
    DEFAULT_TX_BUCKET_CAPACITY_BYTES,
    DEFAULT_TX_BUCKET_CAPACITY_KB,
    DEFAULT_TX_LIMIT_MBPS,
    DEFAULT_WARMUP_STEPS,
    NODE_ORDER,
    TRANSPORT_BACKEND,
    _resolve_run_config,
)
from jetson_inference.flan_t5.experiment import run_node_a
from jetson_inference.flan_t5.workers import run_worker


def _build_parser():
    parser = argparse.ArgumentParser(description="Standalone 4-node Jetson Flan-T5 SST-2 pipeline with single-task style online policies.")
    parser.add_argument("--node", choices=NODE_ORDER, required=True)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset_path", default=DEFAULT_LOCAL_DATASET_PATH)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--prompt_template", default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--positive_token", default=DEFAULT_POSITIVE_TOKEN)
    parser.add_argument("--negative_token", default=DEFAULT_NEGATIVE_TOKEN)
    parser.add_argument("--max_input_length", type=int, default=DEFAULT_MAX_INPUT_LENGTH)
    parser.add_argument("--warmup_steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--t0_ratio", type=float, default=DEFAULT_T0_RATIO)
    parser.add_argument("--mu_values", default=",".join("{:.6g}".format(item) for item in DEFAULT_MU_VALUES))
    parser.add_argument("--dynamic_timeslot_sizes", default="5,10,25")
    parser.add_argument("--max_dynamic_timeslot_count", type=int, default=DEFAULT_MAX_DYNAMIC_TIMESLOT_COUNT)
    parser.add_argument("--max_total_batches", type=int, default=None)
    parser.add_argument("--accuracy_estimator_modes", default=DEFAULT_ACCURACY_ESTIMATOR_MODES)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--stein_sigma", type=float, default=DEFAULT_STEIN_SIGMA)
    parser.add_argument("--stein_N", type=int, default=DEFAULT_STEIN_N)
    parser.add_argument("--stein_fast_max_samples", type=int, default=DEFAULT_STEIN_FAST_MAX_SAMPLES)
    parser.add_argument("--channel_predictor_mode", default=DEFAULT_CHANNEL_UPDATE_MODE, choices=["warmup", "online"])
    parser.add_argument("--channel_model_type", default=DEFAULT_CHANNEL_MODEL_TYPE, choices=SUPPORTED_CHANNEL_MODEL_TYPES)
    parser.add_argument("--channel_window_size", type=int, default=DEFAULT_CHANNEL_WINDOW_SIZE)
    parser.add_argument("--transport_backend", default=TRANSPORT_BACKEND)
    parser.add_argument("--tx_limit_mbps", type=float, default=DEFAULT_TX_LIMIT_MBPS)
    parser.add_argument("--tx_bucket_capacity_bytes", type=int, default=DEFAULT_TX_BUCKET_CAPACITY_BYTES)
    parser.add_argument("--tx_bucket_capacity_kb", type=float, default=DEFAULT_TX_BUCKET_CAPACITY_KB)
    parser.add_argument("--compressor_profiles", default="topk,quantization,llmint8_fp16_int8")
    parser.add_argument("--log_level", default="INFO")
    return parser

def main():
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if args.tx_bucket_capacity_kb is not None:
        args.tx_bucket_capacity_bytes = int(float(args.tx_bucket_capacity_kb) * 1024.0)
    cfg = _resolve_run_config(args)

    if args.node == "A":
        run_node_a(
            device=cfg["device"],
            model_name=cfg["model_name"],
            dataset_path=cfg["dataset_path"],
            split=cfg["split"],
            batch_size=cfg["batch_size"],
            max_samples=cfg["max_samples"],
            prompt_template=cfg["prompt_template"],
            positive_token=cfg["positive_token"],
            negative_token=cfg["negative_token"],
            max_input_length=cfg["max_input_length"],
            warmup_steps=cfg["warmup_steps"],
            t0_ratio=args.t0_ratio,
            t0_ratios=cfg["t0_ratios"],
            mu_values_arg=cfg["mu_values"],
            dynamic_timeslot_sizes_arg=cfg["dynamic_timeslot_sizes"],
            max_dynamic_timeslot_count=cfg["max_dynamic_timeslot_count"],
            max_total_batches=cfg["max_total_batches"],
            accuracy_estimator_modes_arg=cfg["accuracy_estimator_modes"],
            epsilon=args.epsilon,
            stein_sigma=cfg["stein_sigma"],
            stein_N=cfg["stein_N"],
            stein_fast_max_samples=cfg["stein_fast_max_samples"],
            channel_predictor_mode=cfg["channel_predictor_mode"],
            channel_model_type=cfg["channel_model_type"],
            channel_window_size=cfg["channel_window_size"],
            transport_backend=cfg["transport_backend"],
            tx_limit_mbps=cfg["tx_limit_mbps"],
            tx_bucket_capacity_bytes=cfg["tx_bucket_capacity_bytes"],
            compressor_profiles_arg=cfg["compressor_profiles"],
            base_output_dir=cfg["base_output_dir"],
            profile_overrides=cfg["profile_overrides"],
        )
    else:
        run_worker(
            node_id=args.node,
            device=cfg["device"],
            model_name=cfg["model_name"],
            max_input_length=cfg["max_input_length"],
            transport_backend=cfg["transport_backend"],
            tx_limit_mbps=cfg["tx_limit_mbps"],
            tx_bucket_capacity_bytes=cfg["tx_bucket_capacity_bytes"],
        )
