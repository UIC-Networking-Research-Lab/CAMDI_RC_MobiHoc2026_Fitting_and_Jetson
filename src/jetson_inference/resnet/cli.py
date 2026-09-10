"""argument parser and main dispatch; defaults are unchanged."""

import argparse
import logging

from jetson_inference.common.channel_estimator import SUPPORTED_CHANNEL_MODEL_TYPES
from jetson_inference.resnet.config import (
    DEFAULT_ACCURACY_ESTIMATOR_MODES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHANNEL_MODEL_TYPE,
    DEFAULT_CHANNEL_UPDATE_MODE,
    DEFAULT_CHANNEL_WINDOW_SIZE,
    DEFAULT_DATA_ROOT,
    DEFAULT_EPSILON,
    DEFAULT_MAX_BATCHES,
    DEFAULT_MAX_DYNAMIC_TIMESLOT_COUNT,
    DEFAULT_STEIN_FAST_MAX_BATCHES,
    DEFAULT_STEIN_N,
    DEFAULT_STEIN_SIGMA,
    DEFAULT_T0_RATIO,
    DEFAULT_TX_BUCKET_CAPACITY_KB,
    DEFAULT_TX_LIMIT_MBPS,
    DEFAULT_WARMUP_STEPS,
    MODEL_CHECKPOINT,
    NODE_ORDER,
    TRANSPORT_BACKEND,
)
from jetson_inference.resnet.experiment import run_node_a
from jetson_inference.resnet.model import (
    ResNet56PartitionFactory,
    print_partition_plan,
)
from jetson_inference.resnet.workers import run_worker


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
