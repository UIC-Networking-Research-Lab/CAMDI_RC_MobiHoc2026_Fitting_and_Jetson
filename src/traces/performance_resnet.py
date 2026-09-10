"""Collect identity-transport ResNet timing traces and local codec overhead."""

import argparse

import datetime

import json

import logging

import os

import time

import uuid

import numpy as np

import pandas as pd

import torch

from jetson_inference.common.compressors import get_compressor

from .performance_resnet_support import (
    ASYNC_MAX_INFLIGHT_TASKS,
    ASYNC_QUEUE_POLL_SEC,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATA_ROOT,
    DEFAULT_DOWNLOAD_DATA,
    DEFAULT_MAX_BATCHES,
    DEFAULT_TX_BUCKET_CAPACITY_BYTES,
    DEFAULT_TX_LIMIT_MBPS,
    MODEL_CHECKPOINT,
    MODEL_TAG,
    NODE_IPS,
    NODE_ORDER,
    NODE_PORTS,
    NUM_PARTITIONS,
    NUM_TRANSFER_POINTS,
    TRANSPORT_BACKEND,
    TRANSPORT_READY_TIMEOUT_SEC,
    MessageType,
    ResNet56PartitionFactory,
    _accuracy_from_predictions,
    _collect_result_payload,
    _drain_worker_stats,
    _flatten_dict,
    _profile_from_partition,
    _save_rows,
    _worker_profile,
    build_activation_payload,
    create_transport,
    load_cifar10_batches,
    restore_activation_payload,
)

BASE_OUTPUT_DIR = "scenario_trace_outputs"

TRACE_DATASET_TAG = "cifar10"

DEFAULT_SIMULATED_CODECS = "topk,quantization,llmint8"

def _timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def _safe_rate(original_bytes, elapsed_sec):
    if original_bytes <= 0 or elapsed_sec <= 0:
        return 0.0
    return float(original_bytes) / float(elapsed_sec)

def _parse_sim_param(raw_value, default_value):
    if raw_value is None or str(raw_value).strip() == "":
        return default_value
    text = str(raw_value).strip()
    if text.startswith("[") or text.startswith("{") or text.startswith("\""):
        return json.loads(text)
    try:
        return float(text)
    except ValueError:
        return json.loads(text)

def _parse_simulated_codecs(text):
    allowed = {"topk", "quantization", "llmint8"}
    names = []
    for item in str(text).split(","):
        name = item.strip().lower()
        if not name:
            continue
        if name not in allowed:
            raise ValueError(
                "Unsupported codec '{}' in --simulated_codecs. Choose from {}".format(
                    name,
                    sorted(allowed),
                )
            )
        if name not in names:
            names.append(name)
    if not names:
        raise ValueError("At least one codec must be provided in --simulated_codecs.")
    return names

def _build_simulation_specs(args):
    all_specs = {
        "topk": {
            "compressor_name": "topk",
            "compression_param": _parse_sim_param(args.topk_param, 0.5),
        },
        "quantization": {
            "compressor_name": "quantization",
            "compression_param": _parse_sim_param(args.quantization_param, 0.25),
        },
        "llmint8": {
            "compressor_name": "llmint8",
            "compression_param": _parse_sim_param(args.llmint8_param, [0.5, "fp16", "int4"]),
        },
    }
    specs = {
        codec_name: dict(all_specs[codec_name])
        for codec_name in _parse_simulated_codecs(args.simulated_codecs)
    }
    for spec in specs.values():
        get_compressor(spec["compressor_name"])
        spec["feature_k_value"] = 1.0 if isinstance(spec["compression_param"], list) else float(spec["compression_param"])
    return specs

def _make_output_dir(base_output_dir, scenario_name, node_id, tx_limit_mbps, batch_size):
    dirname = "{}_{}_{}_{}".format(
        "{}_bs{}".format(str(scenario_name), int(batch_size)),
        _format_bandwidth_tag(tx_limit_mbps),
        str(node_id),
        _timestamp(),
    )
    path = os.path.join(base_output_dir, dirname)
    os.makedirs(path, exist_ok=True)
    return path

def _format_bandwidth_tag(tx_limit_mbps):
    if tx_limit_mbps is None:
        return "unlimited"
    value = float(tx_limit_mbps)
    if value.is_integer():
        return "{}Mbps".format(int(value))
    return "{}Mbps".format(str(value).replace(".", "p"))

def _scenario_data_filename(args, codec_name, kind):
    bandwidth_tag = _format_bandwidth_tag(args.tx_limit_mbps)
    model_tag = str(MODEL_TAG).replace("-", "_")
    return "{}_{}_bs{}_{}_{}_{}.json".format(
        model_tag,
        TRACE_DATASET_TAG,
        int(args.batch_size),
        bandwidth_tag,
        kind,
        str(codec_name),
    )

def _simulate_sender_compress_profile(tensor, sim_spec):
    started = time.perf_counter()
    _, stats = build_activation_payload(
        tensor,
        compression_param=sim_spec["compression_param"],
        compressor_name=sim_spec["compressor_name"],
        feature_k_value=sim_spec["feature_k_value"],
    )
    elapsed = time.perf_counter() - started
    return {
        "compress_sec": float(elapsed),
        "compressed_bytes": int(stats["compressed_bytes"]),
        "original_bytes": int(stats["original_bytes"]),
        "compression_ratio": float(stats["compression_ratio"]),
    }

def _simulate_receiver_decompress_profile(tensor, sim_spec, device):
    payload, stats = build_activation_payload(
        tensor,
        compression_param=sim_spec["compression_param"],
        compressor_name=sim_spec["compressor_name"],
        feature_k_value=sim_spec["feature_k_value"],
    )
    started = time.perf_counter()
    _ = restore_activation_payload(payload, device)
    elapsed = time.perf_counter() - started
    return {
        "decompress_sec": float(elapsed),
        "compressed_bytes": int(stats["compressed_bytes"]),
        "original_bytes": int(stats["original_bytes"]),
        "compression_ratio": float(stats["compression_ratio"]),
    }

def _codec_adjusted_tau(compute_sec, decompress_sec=0.0, compress_sec=0.0):
    return float(compute_sec) + float(decompress_sec) + float(compress_sec)

def _dispatch_async_task_collect(transport, partition, batch, task_id, device, sim_specs):
    images = batch["images"].to(device)
    t_comp = time.perf_counter()
    with torch.no_grad():
        hidden = partition(images)
    compute_sec = time.perf_counter() - t_comp
    sim_profiles_a = {}
    for codec_name, sim_spec in sim_specs.items():
        sim_encode = _simulate_sender_compress_profile(hidden, sim_spec)
        sim_profiles_a[codec_name] = {
            "compress_sec": float(sim_encode["compress_sec"]),
            "decompress_sec": 0.0,
            "codec_adjusted_tau_sec": _codec_adjusted_tau(compute_sec, 0.0, sim_encode["compress_sec"]),
            "simulated_output_compressed_bytes": int(sim_encode["compressed_bytes"]),
            "simulated_output_original_bytes": int(sim_encode["original_bytes"]),
            "simulated_output_compression_ratio": float(sim_encode["compression_ratio"]),
        }

    t_prepare = time.perf_counter()
    hidden_comp, comp_stats = build_activation_payload(
        hidden,
        compression_param=None,
        compressor_name="identity",
        feature_k_value=1.0,
    )
    outgoing = {
        "task_id": task_id,
        "batch_idx": batch["batch_idx"],
        "compressor_name": "identity",
        "compression_params_list": [None, None, None],
        "feature_k_values": [1.0, 1.0, 1.0],
        "candidate_id": "identity_1_1_1",
        "outlier_values": [],
        "activation_comp": hidden_comp,
        "tp_original_bytes": [comp_stats["original_bytes"], 0, 0],
        "sim_specs": sim_specs,
    }
    prepare_sec = time.perf_counter() - t_prepare

    bytes_sent, send_sec = transport.send("B", MessageType.TASK_INPUT, outgoing)
    a_profile = _worker_profile("A", compute_sec, 0.0, prepare_sec, bytes_sent, send_sec)
    return {
        "task_id": task_id,
        "batch_idx": batch["batch_idx"],
        "labels": batch["labels"].clone(),
        "started_at": time.perf_counter(),
        "tp_original_bytes": list(outgoing["tp_original_bytes"]),
        "tp_stats": {"A": {"bytes": int(bytes_sent), "elapsed": float(send_sec)}},
        "profiles": {"A": a_profile},
        "sim_profiles": {"A": sim_profiles_a},
        "worker_stats": {},
    }

def _run_worker_b_collect(transport, partition, device):
    while True:
        envelope = transport.receive(msg_type=MessageType.TASK_INPUT, source=None)
        payload = envelope["payload"]

        t_restore = time.perf_counter()
        hidden_in = restore_activation_payload(payload["activation_comp"], device)
        restore_sec = time.perf_counter() - t_restore
        sim_profiles = {}
        sim_specs = payload["sim_specs"]
        hidden, compute_sec = _profile_from_partition(partition, hidden_in)
        for codec_name, sim_spec in sim_specs.items():
            sim_decode = _simulate_receiver_decompress_profile(hidden_in, sim_spec, device)
            sim_encode = _simulate_sender_compress_profile(hidden, sim_spec)
            sim_profiles[codec_name] = {
                "compress_sec": float(sim_encode["compress_sec"]),
                "decompress_sec": float(sim_decode["decompress_sec"]),
                "codec_adjusted_tau_sec": _codec_adjusted_tau(compute_sec, sim_decode["decompress_sec"], sim_encode["compress_sec"]),
                "simulated_input_compressed_bytes": int(sim_decode["compressed_bytes"]),
                "simulated_output_compressed_bytes": int(sim_encode["compressed_bytes"]),
                "simulated_input_original_bytes": int(sim_decode["original_bytes"]),
                "simulated_output_original_bytes": int(sim_encode["original_bytes"]),
                "simulated_input_compression_ratio": float(sim_decode["compression_ratio"]),
                "simulated_output_compression_ratio": float(sim_encode["compression_ratio"]),
            }

        t_prepare = time.perf_counter()
        hidden_comp, stats = build_activation_payload(hidden, None, "identity", 1.0)
        outgoing = dict(payload)
        outgoing["activation_comp"] = hidden_comp
        outgoing["tp_original_bytes"][1] = stats["original_bytes"]
        prepare_sec = time.perf_counter() - t_prepare

        bytes_sent, send_sec = transport.send("C", MessageType.ENCODER_OUTPUT, outgoing)
        transport.send("A", MessageType.WORKER_STATS, {
            "task_id": payload["task_id"],
            "node_id": "B",
            "tp_index": 1,
            "tp_stats": {"bytes": int(bytes_sent), "elapsed": float(send_sec)},
            "profile": _worker_profile("B", compute_sec, restore_sec, prepare_sec, bytes_sent, send_sec),
            "sim_profiles": sim_profiles,
        })

def _run_worker_c_collect(transport, partition, device):
    while True:
        envelope = transport.receive(msg_type=MessageType.ENCODER_OUTPUT, source=None)
        payload = envelope["payload"]

        t_restore = time.perf_counter()
        hidden_in = restore_activation_payload(payload["activation_comp"], device)
        restore_sec = time.perf_counter() - t_restore
        sim_profiles = {}
        sim_specs = payload["sim_specs"]
        hidden, compute_sec = _profile_from_partition(partition, hidden_in)
        for codec_name, sim_spec in sim_specs.items():
            sim_decode = _simulate_receiver_decompress_profile(hidden_in, sim_spec, device)
            sim_encode = _simulate_sender_compress_profile(hidden, sim_spec)
            sim_profiles[codec_name] = {
                "compress_sec": float(sim_encode["compress_sec"]),
                "decompress_sec": float(sim_decode["decompress_sec"]),
                "codec_adjusted_tau_sec": _codec_adjusted_tau(compute_sec, sim_decode["decompress_sec"], sim_encode["compress_sec"]),
                "simulated_input_compressed_bytes": int(sim_decode["compressed_bytes"]),
                "simulated_output_compressed_bytes": int(sim_encode["compressed_bytes"]),
                "simulated_input_original_bytes": int(sim_decode["original_bytes"]),
                "simulated_output_original_bytes": int(sim_encode["original_bytes"]),
                "simulated_input_compression_ratio": float(sim_decode["compression_ratio"]),
                "simulated_output_compression_ratio": float(sim_encode["compression_ratio"]),
            }

        t_prepare = time.perf_counter()
        hidden_comp, stats = build_activation_payload(hidden, None, "identity", 1.0)
        outgoing = dict(payload)
        outgoing["activation_comp"] = hidden_comp
        outgoing["tp_original_bytes"][2] = stats["original_bytes"]
        prepare_sec = time.perf_counter() - t_prepare

        bytes_sent, send_sec = transport.send("D", MessageType.DECODER_STEP, outgoing)
        transport.send("A", MessageType.WORKER_STATS, {
            "task_id": payload["task_id"],
            "node_id": "C",
            "tp_index": 2,
            "tp_stats": {"bytes": int(bytes_sent), "elapsed": float(send_sec)},
            "profile": _worker_profile("C", compute_sec, restore_sec, prepare_sec, bytes_sent, send_sec),
            "sim_profiles": sim_profiles,
        })

def _run_worker_d_collect(transport, partition, device):
    while True:
        envelope = transport.receive(msg_type=MessageType.DECODER_STEP, source=None)
        payload = envelope["payload"]

        t_restore = time.perf_counter()
        hidden_in = restore_activation_payload(payload["activation_comp"], device)
        restore_sec = time.perf_counter() - t_restore
        logits, compute_sec = _profile_from_partition(partition, hidden_in)
        predictions = torch.argmax(logits, dim=1).detach().cpu().tolist()
        sim_profiles_d = {}
        for codec_name, sim_spec in payload["sim_specs"].items():
            sim_decode = _simulate_receiver_decompress_profile(hidden_in, sim_spec, device)
            sim_profiles_d[codec_name] = {
                "compress_sec": 0.0,
                "decompress_sec": float(sim_decode["decompress_sec"]),
                "codec_adjusted_tau_sec": _codec_adjusted_tau(compute_sec, sim_decode["decompress_sec"], 0.0),
                "simulated_input_compressed_bytes": int(sim_decode["compressed_bytes"]),
                "simulated_input_original_bytes": int(sim_decode["original_bytes"]),
                "simulated_input_compression_ratio": float(sim_decode["compression_ratio"]),
            }

        t_prepare = time.perf_counter()
        result_profile_d = _worker_profile("D", compute_sec, restore_sec, 0.0, 0, 0.0)
        result_prepare_sec = time.perf_counter() - t_prepare
        result_profile_d["prepare_send_sec"] = float(result_prepare_sec)
        result_profile_d["service_total_sec"] = float(compute_sec + restore_sec + result_prepare_sec)

        transport.send("A", MessageType.FINAL_RESULT, {
            "task_id": payload["task_id"],
            "batch_idx": payload["batch_idx"],
            "predictions": predictions,
            "tp_original_bytes": list(payload["tp_original_bytes"]),
            "profile_d": result_profile_d,
            "sim_profiles_d": sim_profiles_d,
        })

def _run_worker_collect(node_id, device, checkpoint_path, transport_backend, tx_limit_mbps, tx_bucket_capacity_bytes):
    if node_id == "A":
        raise ValueError("_run_worker_collect only supports B/C/D")

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
            _run_worker_b_collect(transport, partition, device)
        elif node_id == "C":
            _run_worker_c_collect(transport, partition, device)
        else:
            _run_worker_d_collect(transport, partition, device)
    finally:
        transport.close()

def _codec_field_name(codec_name, base_name, node_name=None):
    parts = [base_name, codec_name]
    if node_name is not None:
        parts.append("node_{}".format(node_name))
    return "_".join(parts) + "_sec"

def _build_trace_artifact(args, rows, sim_specs):
    batch_index = []
    activation_bytes = []
    tau_node_sec = []
    tau_residence_node_sec = []
    tau_compute_node_sec = []
    tau_codec_node_sec = {codec_name: [] for codec_name in sim_specs.keys()}
    tau_residence_codec_node_sec = {codec_name: [] for codec_name in sim_specs.keys()}
    channel_true_bps = []
    observed_total_delay_sec = []
    packet_bytes = []
    send_elapsed_sec = []

    for row in rows:
        batch_index.append(int(row["batch_idx"]))
        activation_bytes.append([int(row["a0_bytes"]), int(row["a1_bytes"]), int(row["a2_bytes"])])
        tau_node_sec.append([float(row["tau_node_A_sec"]), float(row["tau_node_B_sec"]), float(row["tau_node_C_sec"]), float(row["tau_node_D_sec"])])
        tau_residence_node_sec.append([float(row["tau_residence_node_A_sec"]), float(row["tau_residence_node_B_sec"]), float(row["tau_residence_node_C_sec"]), float(row["tau_residence_node_D_sec"])])
        tau_compute_node_sec.append([float(row["tau_node_A_compute_sec"]), float(row["tau_node_B_compute_sec"]), float(row["tau_node_C_compute_sec"]), float(row["tau_node_D_compute_sec"])])
        for codec_name in sim_specs.keys():
            tau_codec_node_sec[codec_name].append([
                float(row[_codec_field_name(codec_name, "tau_codec", "A")]),
                float(row[_codec_field_name(codec_name, "tau_codec", "B")]),
                float(row[_codec_field_name(codec_name, "tau_codec", "C")]),
                float(row[_codec_field_name(codec_name, "tau_codec", "D")]),
            ])
            tau_residence_codec_node_sec[codec_name].append([
                float(row[_codec_field_name(codec_name, "tau_residence_codec", "A")]),
                float(row[_codec_field_name(codec_name, "tau_residence_codec", "B")]),
                float(row[_codec_field_name(codec_name, "tau_residence_codec", "C")]),
                float(row[_codec_field_name(codec_name, "tau_residence_codec", "D")]),
            ])
        channel_true_bps.append([float(row["c0_true_bps"]), float(row["c1_true_bps"]), float(row["c2_true_bps"])])
        observed_total_delay_sec.append(float(row["total_delay_sec"]))
        packet_bytes.append([int(row["tp0_packet_bytes"]), int(row["tp1_packet_bytes"]), int(row["tp2_packet_bytes"])])
        send_elapsed_sec.append([float(row["tp0_send_elapsed_sec"]), float(row["tp1_send_elapsed_sec"]), float(row["tp2_send_elapsed_sec"])])

    delay_arr = np.asarray(observed_total_delay_sec, dtype=float)
    c_arr = np.asarray(channel_true_bps, dtype=float) if channel_true_bps else np.zeros((0, NUM_TRANSFER_POINTS))
    a_arr = np.asarray(activation_bytes, dtype=float) if activation_bytes else np.zeros((0, NUM_TRANSFER_POINTS))
    tau_arr = np.asarray(tau_node_sec, dtype=float) if tau_node_sec else np.zeros((0, NUM_PARTITIONS))
    tau_residence_arr = np.asarray(tau_residence_node_sec, dtype=float) if tau_residence_node_sec else np.zeros((0, NUM_PARTITIONS))
    tau_compute_arr = np.asarray(tau_compute_node_sec, dtype=float) if tau_compute_node_sec else np.zeros((0, NUM_PARTITIONS))

    trace_payload = {
        "batch_index": batch_index,
        "activation_bytes": activation_bytes,
        "tau_node_sec": tau_node_sec,
        "tau_residence_node_sec": tau_residence_node_sec,
        "tau_compute_node_sec": tau_compute_node_sec,
        "channel_true_bps": channel_true_bps,
        "observed_total_delay_sec": observed_total_delay_sec,
        "packet_bytes": packet_bytes,
        "send_elapsed_sec": send_elapsed_sec,
    }
    summary_payload = {
        "num_samples": len(rows),
        "median_total_delay_sec": float(np.median(delay_arr)) if len(delay_arr) > 0 else None,
        "avg_total_delay_sec": float(np.mean(delay_arr)) if len(delay_arr) > 0 else None,
        "reference_rate_hz_from_median_delay": float(1.0 / np.median(delay_arr)) if len(delay_arr) > 0 and np.median(delay_arr) > 0 else None,
        "median_activation_bytes": np.median(a_arr, axis=0).tolist() if len(a_arr) > 0 else [],
        "avg_activation_bytes": np.mean(a_arr, axis=0).tolist() if len(a_arr) > 0 else [],
        "median_tau_node_sec": np.median(tau_arr, axis=0).tolist() if len(tau_arr) > 0 else [],
        "avg_tau_node_sec": np.mean(tau_arr, axis=0).tolist() if len(tau_arr) > 0 else [],
        "median_tau_residence_node_sec": np.median(tau_residence_arr, axis=0).tolist() if len(tau_residence_arr) > 0 else [],
        "avg_tau_residence_node_sec": np.mean(tau_residence_arr, axis=0).tolist() if len(tau_residence_arr) > 0 else [],
        "median_tau_compute_node_sec": np.median(tau_compute_arr, axis=0).tolist() if len(tau_compute_arr) > 0 else [],
        "avg_tau_compute_node_sec": np.mean(tau_compute_arr, axis=0).tolist() if len(tau_compute_arr) > 0 else [],
        "median_channel_true_bps": np.median(c_arr, axis=0).tolist() if len(c_arr) > 0 else [],
        "avg_channel_true_bps": np.mean(c_arr, axis=0).tolist() if len(c_arr) > 0 else [],
    }

    for codec_name in sim_specs.keys():
        tau_codec_key = "tau_codec_{}_node_sec".format(codec_name)
        tau_residence_codec_key = "tau_residence_codec_{}_node_sec".format(codec_name)
        tau_codec_arr = np.asarray(tau_codec_node_sec[codec_name], dtype=float) if tau_codec_node_sec[codec_name] else np.zeros((0, NUM_PARTITIONS))
        tau_residence_codec_arr = np.asarray(tau_residence_codec_node_sec[codec_name], dtype=float) if tau_residence_codec_node_sec[codec_name] else np.zeros((0, NUM_PARTITIONS))
        trace_payload[tau_codec_key] = tau_codec_node_sec[codec_name]
        trace_payload[tau_residence_codec_key] = tau_residence_codec_node_sec[codec_name]
        summary_payload["median_{}".format(tau_codec_key)] = np.median(tau_codec_arr, axis=0).tolist() if len(tau_codec_arr) > 0 else []
        summary_payload["avg_{}".format(tau_codec_key)] = np.mean(tau_codec_arr, axis=0).tolist() if len(tau_codec_arr) > 0 else []
        summary_payload["median_{}".format(tau_residence_codec_key)] = np.median(tau_residence_codec_arr, axis=0).tolist() if len(tau_residence_codec_arr) > 0 else []
        summary_payload["avg_{}".format(tau_residence_codec_key)] = np.mean(tau_residence_codec_arr, axis=0).tolist() if len(tau_residence_codec_arr) > 0 else []

    return {
        "schema_version": "scenario_trace_v2",
        "scenario_name": str(args.scenario_name),
        "model_tag": MODEL_TAG,
        "task_type": "single_task",
        "trace_source": "resnet_no_compression_reference",
        "topology": {
            "node_order": list(NODE_ORDER),
            "num_nodes": len(NODE_ORDER),
            "num_links": NUM_TRANSFER_POINTS,
        },
        "simulated_codecs": {name: {"compressor_name": spec["compressor_name"], "compression_param": spec["compression_param"]} for name, spec in sim_specs.items()},
        "trace": trace_payload,
        "summary": summary_payload,
    }

def _build_codec_trace_artifact(args, rows, codec_name, scenario_name):
    sim_specs = _build_simulation_specs(args)
    base = _build_trace_artifact(args, rows, sim_specs)
    trace = base["trace"]
    summary = base["summary"]
    codec_spec = base["simulated_codecs"][codec_name]
    tau_codec_key = "tau_codec_{}_node_sec".format(codec_name)
    tau_residence_codec_key = "tau_residence_codec_{}_node_sec".format(codec_name)
    median_tau_codec_key = "median_tau_codec_{}_node_sec".format(codec_name)
    median_tau_residence_codec_key = "median_tau_residence_codec_{}_node_sec".format(codec_name)
    avg_tau_codec_key = "avg_tau_codec_{}_node_sec".format(codec_name)
    avg_tau_residence_codec_key = "avg_tau_residence_codec_{}_node_sec".format(codec_name)

    return {
        "schema_version": "scenario_trace_codec_v1",
        "experiment": {
            "scenario_name": str(scenario_name),
            "platform": "jetson",
            "model": "resnet56",
            "dataset": {
                "name": "cifar10",
                "batch_size": int(args.batch_size),
            },
            "device": str(args.device),
            "tx_limit_mbps": None if args.tx_limit_mbps is None else float(args.tx_limit_mbps),
            "tx_bucket_capacity_bytes": int(args.tx_bucket_capacity_bytes),
            "network": {
                "transport_backend": "wifi",
                "node_order": list(NODE_ORDER),
                "node_ips": dict(NODE_IPS),
                "node_ports": dict(NODE_PORTS),
                "num_nodes": len(NODE_ORDER),
                "num_links": NUM_TRANSFER_POINTS,
            },
        },
        "simulated_codec_overhead_profile": {
            "codec_name": codec_name,
            "compressor_name": codec_spec["compressor_name"],
            "compression_param": codec_spec["compression_param"],
            "used_for": "simulated_codec_overhead_only",
            "affects_actual_transport": False,
            "actual_transport_compressor": "identity",
        },
        "trace": {
            "batch_index": trace["batch_index"],
            "activation_bytes": trace["activation_bytes"],
            "tau_node_sec": trace["tau_node_sec"],
            "tau_residence_node_sec": trace["tau_residence_node_sec"],
            "tau_compute_node_sec": trace["tau_compute_node_sec"],
            "tau_codec_node_sec": trace[tau_codec_key],
            "tau_residence_codec_node_sec": trace[tau_residence_codec_key],
            "channel_true_bps": trace["channel_true_bps"],
            "observed_total_delay_sec": trace["observed_total_delay_sec"],
            "packet_bytes": trace["packet_bytes"],
            "send_elapsed_sec": trace["send_elapsed_sec"],
        },
        "summary": {
            "num_samples": summary["num_samples"],
            "median_total_delay_sec": summary["median_total_delay_sec"],
            "avg_total_delay_sec": summary["avg_total_delay_sec"],
            "reference_rate_hz_from_median_delay": summary["reference_rate_hz_from_median_delay"],
            "median_activation_bytes": summary["median_activation_bytes"],
            "avg_activation_bytes": summary["avg_activation_bytes"],
            "median_tau_node_sec": summary["median_tau_node_sec"],
            "avg_tau_node_sec": summary["avg_tau_node_sec"],
            "median_tau_residence_node_sec": summary["median_tau_residence_node_sec"],
            "avg_tau_residence_node_sec": summary["avg_tau_residence_node_sec"],
            "median_tau_compute_node_sec": summary["median_tau_compute_node_sec"],
            "avg_tau_compute_node_sec": summary["avg_tau_compute_node_sec"],
            "median_tau_codec_node_sec": summary[median_tau_codec_key],
            "avg_tau_codec_node_sec": summary[avg_tau_codec_key],
            "median_tau_residence_codec_node_sec": summary[median_tau_residence_codec_key],
            "avg_tau_residence_codec_node_sec": summary[avg_tau_residence_codec_key],
            "median_channel_true_bps": summary["median_channel_true_bps"],
            "avg_channel_true_bps": summary["avg_channel_true_bps"],
        },
    }

def _persist_experiment_artifacts(args, rows, output_dir, scenario_name, experiment_idx, wall_start, expected_samples):
    raw_rows_path = os.path.join(output_dir, "no_compression_trace_rows.csv")

    _save_rows(raw_rows_path, rows)
    codec_outputs = {}
    sim_specs = _build_simulation_specs(args)
    for codec_name in sim_specs.keys():
        trace_artifact = _build_codec_trace_artifact(args, rows, codec_name, scenario_name)
        scenario_trace_path = os.path.join(output_dir, _scenario_data_filename(args, codec_name, "scenario_trace"))
        trace_summary_path = os.path.join(output_dir, _scenario_data_filename(args, codec_name, "trace_summary"))
        summary = {
            "scenario_name": scenario_name,
            "experiment_idx": int(experiment_idx),
            "output_dir": output_dir,
            "codec_name": codec_name,
            "num_samples": len(rows),
            "expected_samples": int(expected_samples),
            "wall_sec": float(time.perf_counter() - wall_start),
            "reference_rate_hz_from_median_delay": trace_artifact["summary"]["reference_rate_hz_from_median_delay"],
            "median_total_delay_sec": trace_artifact["summary"]["median_total_delay_sec"],
            "avg_total_delay_sec": trace_artifact["summary"]["avg_total_delay_sec"],
            "median_tau_residence_node_sec": trace_artifact["summary"]["median_tau_residence_node_sec"],
            "avg_tau_residence_node_sec": trace_artifact["summary"]["avg_tau_residence_node_sec"],
            "median_tau_compute_node_sec": trace_artifact["summary"]["median_tau_compute_node_sec"],
            "avg_tau_compute_node_sec": trace_artifact["summary"]["avg_tau_compute_node_sec"],
            "median_tau_codec_node_sec": trace_artifact["summary"]["median_tau_codec_node_sec"],
            "avg_tau_codec_node_sec": trace_artifact["summary"]["avg_tau_codec_node_sec"],
            "median_tau_residence_codec_node_sec": trace_artifact["summary"]["median_tau_residence_codec_node_sec"],
            "avg_tau_residence_codec_node_sec": trace_artifact["summary"]["avg_tau_residence_codec_node_sec"],
            "median_channel_true_bps": trace_artifact["summary"]["median_channel_true_bps"],
            "avg_channel_true_bps": trace_artifact["summary"]["avg_channel_true_bps"],
            "simulated_codec_overhead_profile": trace_artifact["simulated_codec_overhead_profile"],
        }
        with open(scenario_trace_path, "w", encoding="utf-8") as handle:
            json.dump(trace_artifact, handle, ensure_ascii=False, indent=2)
        with open(trace_summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        codec_outputs[codec_name] = {
            "summary": summary,
            "scenario_trace_path": scenario_trace_path,
            "trace_summary_path": trace_summary_path,
        }
    return {"scenario_name": scenario_name, "experiment_idx": int(experiment_idx), "output_dir": output_dir, "raw_rows_path": raw_rows_path, "codec_outputs": codec_outputs}

def _run_single_collection_experiment(args, transport, partition, batches, sim_specs, experiment_idx):
    scenario_name = str(args.scenario_name)
    if int(args.experiment_count) > 1:
        scenario_name = "{}_exp{:03d}".format(scenario_name, int(experiment_idx))
    output_dir = _make_output_dir(args.output_dir, scenario_name, args.node, args.tx_limit_mbps, args.batch_size)
    logging.info("[A][exp=%d] Output directory: %s", int(experiment_idx), output_dir)
    rows = []
    pending = {}
    buffered_stats = {}
    dispatched = 0
    completed = 0
    total_correct = 0
    total_seen = 0
    wall_start = time.perf_counter()
    summary = None

    while completed < len(batches):
        _drain_worker_stats(transport, pending, buffered_stats)
        while dispatched < len(batches) and len(pending) < ASYNC_MAX_INFLIGHT_TASKS:
            batch = batches[dispatched]
            task_id = "collect_exp{:03d}_{}_{}".format(int(experiment_idx), int(dispatched), uuid.uuid4().hex[:8])
            task_state = _dispatch_async_task_collect(transport, partition, batch, task_id, args.device, sim_specs)
            if task_id in buffered_stats:
                task_state["worker_stats"].update(buffered_stats.pop(task_id))
            task_state["t_index"] = dispatched
            task_state["experiment_idx"] = int(experiment_idx)
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
        if task_id in buffered_stats:
            task_state["worker_stats"].update(buffered_stats.pop(task_id))

        stats_b = task_state["worker_stats"].get("B")
        stats_c = task_state["worker_stats"].get("C")
        profile_b = stats_b["profile"] if stats_b is not None else _worker_profile("B", 0.0, 0.0, 0.0, 0, 0.0)
        profile_c = stats_c["profile"] if stats_c is not None else _worker_profile("C", 0.0, 0.0, 0.0, 0, 0.0)
        profile_d = payload.get("profile_d") or _worker_profile("D", 0.0, 0.0, 0.0, 0, 0.0)
        sim_profiles_a = task_state["sim_profiles"].get("A", {})
        sim_profiles_b = stats_b.get("sim_profiles") if stats_b is not None else {}
        sim_profiles_c = stats_c.get("sim_profiles") if stats_c is not None else {}
        sim_profiles_d = payload.get("sim_profiles_d") or {}
        tp1_stats = stats_b["tp_stats"] if stats_b is not None else {"bytes": 0, "elapsed": 0.0}
        tp2_stats = stats_c["tp_stats"] if stats_c is not None else {"bytes": 0, "elapsed": 0.0}

        total_delay = time.perf_counter() - task_state["started_at"]
        correct, total, batch_acc = _accuracy_from_predictions(payload["predictions"], task_state["labels"])
        total_correct += correct
        total_seen += total
        a_bytes = list(payload.get("tp_original_bytes", task_state.get("tp_original_bytes", [0] * NUM_TRANSFER_POINTS)))
        tp0_elapsed = float(task_state["tp_stats"]["A"]["elapsed"])
        tp1_elapsed = float(tp1_stats["elapsed"])
        tp2_elapsed = float(tp2_stats["elapsed"])

        row = {
            "scenario_name": scenario_name,
            "experiment_idx": int(experiment_idx),
            "t_index": int(task_state["t_index"]),
            "batch_idx": int(payload["batch_idx"]),
            "task_id": task_id,
            "batch_accuracy": float(batch_acc),
            "running_accuracy": float(total_correct) / float(total_seen) if total_seen > 0 else 0.0,
            "total_delay_sec": float(total_delay),
            "a0_bytes": int(a_bytes[0]),
            "a1_bytes": int(a_bytes[1]),
            "a2_bytes": int(a_bytes[2]),
            "tp0_packet_bytes": int(task_state["tp_stats"]["A"]["bytes"]),
            "tp1_packet_bytes": int(tp1_stats["bytes"]),
            "tp2_packet_bytes": int(tp2_stats["bytes"]),
            "tp0_send_elapsed_sec": tp0_elapsed,
            "tp1_send_elapsed_sec": tp1_elapsed,
            "tp2_send_elapsed_sec": tp2_elapsed,
            "c0_true_bps": _safe_rate(a_bytes[0], tp0_elapsed),
            "c1_true_bps": _safe_rate(a_bytes[1], tp1_elapsed),
            "c2_true_bps": _safe_rate(a_bytes[2], tp2_elapsed),
            "tau_node_A_sec": float(task_state["profiles"]["A"]["service_total_sec"]),
            "tau_node_B_sec": float(profile_b["service_total_sec"]),
            "tau_node_C_sec": float(profile_c["service_total_sec"]),
            "tau_node_D_sec": float(profile_d["service_total_sec"]),
            "tau_residence_node_A_sec": float(task_state["profiles"]["A"]["service_total_sec"]),
            "tau_residence_node_B_sec": float(profile_b["service_total_sec"]),
            "tau_residence_node_C_sec": float(profile_c["service_total_sec"]),
            "tau_residence_node_D_sec": float(profile_d["service_total_sec"]),
            "tau_node_A_compute_sec": float(task_state["profiles"]["A"]["compute_sec"]),
            "tau_node_B_compute_sec": float(profile_b["compute_sec"]),
            "tau_node_C_compute_sec": float(profile_c["compute_sec"]),
            "tau_node_D_compute_sec": float(profile_d["compute_sec"]),
            **_flatten_dict({"profiles": {"A": task_state["profiles"]["A"], "B": profile_b, "C": profile_c, "D": profile_d}}),
        }
        codec_profile_map = {"A": sim_profiles_a, "B": sim_profiles_b, "C": sim_profiles_c, "D": sim_profiles_d}
        compute_fallbacks = {
            "A": task_state["profiles"]["A"]["compute_sec"],
            "B": profile_b["compute_sec"],
            "C": profile_c["compute_sec"],
            "D": profile_d["compute_sec"],
        }
        residence_fallbacks = {
            "A": task_state["profiles"]["A"]["service_total_sec"],
            "B": profile_b["service_total_sec"],
            "C": profile_c["service_total_sec"],
            "D": profile_d["service_total_sec"],
        }
        for codec_name, spec in sim_specs.items():
            row["sim_{}_compression_param".format(codec_name)] = json.dumps(spec["compression_param"])
            for node_name in ["A", "B", "C", "D"]:
                profile_map = codec_profile_map.get(node_name, {})
                codec_profile = profile_map.get(codec_name, {})
                row["sim_{}_{}_compress_sec".format(codec_name, node_name)] = float(codec_profile.get("compress_sec", 0.0))
                row["sim_{}_{}_decompress_sec".format(codec_name, node_name)] = float(codec_profile.get("decompress_sec", 0.0))
                row[_codec_field_name(codec_name, "tau_codec", node_name)] = float(codec_profile.get("codec_adjusted_tau_sec", compute_fallbacks[node_name]))
                row[_codec_field_name(codec_name, "tau_residence_codec", node_name)] = float(residence_fallbacks[node_name] + codec_profile.get("compress_sec", 0.0) + codec_profile.get("decompress_sec", 0.0))
            row["sim_{}_tp0_compressed_bytes".format(codec_name)] = int(sim_profiles_a.get(codec_name, {}).get("simulated_output_compressed_bytes", 0))
            row["sim_{}_tp1_compressed_bytes".format(codec_name)] = int(sim_profiles_b.get(codec_name, {}).get("simulated_output_compressed_bytes", 0))
            row["sim_{}_tp2_compressed_bytes".format(codec_name)] = int(sim_profiles_c.get(codec_name, {}).get("simulated_output_compressed_bytes", 0))
        rows.append(row)
        logging.info(
            "[A][exp=%d] t=%d batch=%d delay=%.4fs acc=%.4f c_true=[%.3e, %.3e, %.3e]",
            int(experiment_idx),
            int(task_state["t_index"]),
            int(payload["batch_idx"]),
            float(total_delay),
            float(batch_acc),
            float(row["c0_true_bps"]),
            float(row["c1_true_bps"]),
            float(row["c2_true_bps"]),
        )
        completed += 1

    logging.info("[A][exp=%d] All batches completed, saving artifacts per codec ...", int(experiment_idx))
    try:
        summary = _persist_experiment_artifacts(
            args=args,
            rows=rows,
            output_dir=output_dir,
            scenario_name=scenario_name,
            experiment_idx=experiment_idx,
            wall_start=wall_start,
            expected_samples=len(batches),
        )
        logging.info("[A][exp=%d] Saved: %s", int(experiment_idx), summary["raw_rows_path"])
        for codec_name, codec_output in summary["codec_outputs"].items():
            logging.info("[A][exp=%d][%s] Saved: %s", int(experiment_idx), codec_name, codec_output["scenario_trace_path"])
            logging.info("[A][exp=%d][%s] Saved: %s", int(experiment_idx), codec_name, codec_output["trace_summary_path"])
    except Exception as exc:
        error_path = os.path.join(output_dir, "trace_error.txt")
        logging.exception("[A][exp=%d] Finalization failed", int(experiment_idx))
        try:
            with open(error_path, "w", encoding="utf-8") as handle:
                handle.write(str(exc))
        except Exception:
            pass
        raise
    return summary

def _collect_no_compression_trace(args):
    if args.node != "A":
        raise ValueError("collector only supports --node A")

    factory = ResNet56PartitionFactory(checkpoint_path=args.checkpoint_path, device=args.device)
    partition = factory.build_partition_for_node("A")
    batches = load_cifar10_batches(
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        data_root=args.data_root,
        download=args.download_data,
    )
    sim_specs = _build_simulation_specs(args)

    transport = create_transport(
        "A",
        NODE_IPS,
        NODE_PORTS,
        backend=args.transport_backend,
        tx_limit_mbps=args.tx_limit_mbps,
        tx_bucket_capacity_bytes=args.tx_bucket_capacity_bytes,
    )
    logging.info("[A] Loaded %d batches", len(batches))
    logging.info(
        "[A] Simulated codecs: %s",
        ", ".join(
            "{}={}".format(codec_name, json.dumps(spec["compression_param"]))
            for codec_name, spec in sim_specs.items()
        ),
    )
    transport.start()
    logging.info("[A] Waiting for workers B/C/D ...")
    transport.wait_for_peers(["B", "C", "D"], timeout=TRANSPORT_READY_TIMEOUT_SEC)
    logging.info("[A] All workers are ready")

    summaries = []
    try:
        for experiment_idx in range(int(args.experiment_count)):
            logging.info("[A] Experiment start: exp=%d/%d", int(experiment_idx) + 1, int(args.experiment_count))
            summary = _run_single_collection_experiment(
                args=args,
                transport=transport,
                partition=partition,
                batches=batches,
                sim_specs=sim_specs,
                experiment_idx=experiment_idx,
            )
            if summary is not None:
                summaries.append(summary)
    finally:
        logging.info("[A] Closing transport ...")
        transport.close()
        logging.info("[A] Transport closed")
    if summaries:
        print(json.dumps({"experiments": summaries}, ensure_ascii=False, indent=2))

def _build_parser():
    parser = argparse.ArgumentParser(description="Collect no-compression ResNet trace data plus simulated codec overhead.")
    parser.add_argument("--node", choices=["A", "B", "C", "D"], required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint_path", default=MODEL_CHECKPOINT)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_batches", type=int, default=DEFAULT_MAX_BATCHES)
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--download_data", action="store_true", default=DEFAULT_DOWNLOAD_DATA)
    parser.add_argument("--transport_backend", default=TRANSPORT_BACKEND)
    parser.add_argument("--tx_limit_mbps", type=float, default=DEFAULT_TX_LIMIT_MBPS)
    parser.add_argument("--tx_bucket_capacity_bytes", type=int, default=DEFAULT_TX_BUCKET_CAPACITY_BYTES)
    parser.add_argument("--scenario_name", default="resnet_no_compression_trace")
    parser.add_argument("--output_dir", default=BASE_OUTPUT_DIR)
    parser.add_argument("--experiment_count", type=int, default=1, help="Number of sequential collect experiments to run over the same resident B/C/D workers.")
    parser.add_argument("--topk_param", default=None, help="Typical TopK parameter for simulated codec timing. Default: 0.5")
    parser.add_argument("--quantization_param", default=None, help="Typical quantization k parameter for simulated codec timing. Default: 0.25")
    parser.add_argument("--llmint8_param", default=None, help="Typical LLM.int8 parameter list for simulated codec timing. Default: [0.5,\"fp16\",\"int4\"]")
    parser.add_argument("--simulated_codecs", default=DEFAULT_SIMULATED_CODECS,
                        help="Comma-separated codec list. Default runs topk, quantization, and llmint8.")
    parser.add_argument("--log_level", default="INFO")
    return parser

def main():
    parser = _build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    if args.node != "A":
        _run_worker_collect(args.node, args.device, args.checkpoint_path, args.transport_backend, args.tx_limit_mbps, args.tx_bucket_capacity_bytes)
        return
    _collect_no_compression_trace(args)
