"""Collect identity-transport Flan-T5 timing traces and local codec overhead."""

import argparse

import datetime

import json

import logging

import os

import threading

import time

import uuid

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import pandas as pd

import torch

from jetson_inference.common.compressors import get_compressor

from fitting.calibrate_flan_t5 import (
    DEFAULT_DEVICE,
    DEFAULT_MAX_INPUT_LENGTH,
    DEFAULT_MODEL_NAME,
    DEFAULT_NEGATIVE_TOKEN,
    DEFAULT_POSITIVE_TOKEN,
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_SPLIT,
    FlanT5DecoderPartition,
    FlanT5EncoderPartition,
    LLMINT8_POLICY_SPECS,
    build_prompt,
    get_transformers_runtime,
    load_sst2_samples,
    normalize_attention_mask,
    resolve_single_token_verbalizers,
)

from jetson_inference.common.http_transport import DEFAULT_TX_BUCKET_CAPACITY_BYTES, MessageType, create_transport

NODE_IPS = {
    "A": "192.168.1.20",
    "B": "192.168.1.21",
    "C": "192.168.1.22",
    "D": "192.168.1.23",
}

NODE_PORTS = {
    "A": 52000,
    "B": 52001,
    "C": 52002,
    "D": 52003,
}

ENCODER_SPLITS = {
    "A": (0, 4),
    "B": (4, 8),
    "C": (8, 12),
}

DECODER_SPLITS = {
    "D": (0, 12),
}

BASE_OUTPUT_DIR = "scenario_trace_outputs"

TRACE_DATASET_TAG = "sst2"

TRANSPORT_BACKEND = "http"

TRANSPORT_READY_TIMEOUT_SEC = 180.0

ASYNC_QUEUE_POLL_SEC = 0.01

NUM_TRANSFER_POINTS = 3

NODE_ORDER = ["A", "B", "C", "D"]

DEFAULT_TX_LIMIT_MBPS = 25.0

DEFAULT_TOPK_PARAM = 0.5

DEFAULT_QUANTIZATION_PARAM = 0.25

DEFAULT_LLMINT8_POLICY = "fp16_int8"

DEFAULT_LLMINT8_OUTLIER = 0.1

DEFAULT_SIMULATED_CODECS = "topk,quantization,llmint8"

DEFAULT_MAX_INFLIGHT_TASKS = 4

def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def _sanitize_tag(text: Any) -> str:
    chars = []
    for ch in str(text).strip().lower():
        chars.append(ch if ch.isalnum() else "_")
    return "".join(chars).strip("_")

def _safe_rate(original_bytes: float, elapsed_sec: float) -> float:
    if original_bytes <= 0 or elapsed_sec <= 0:
        return 0.0
    return float(original_bytes) / float(elapsed_sec)

def _tensor_payload_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())

def _format_bandwidth_tag(tx_limit_mbps: Optional[float]) -> str:
    if tx_limit_mbps is None:
        return "unlimited"
    value = float(tx_limit_mbps)
    if value.is_integer():
        return "{}Mbps".format(int(value))
    return "{}Mbps".format(str(value).replace(".", "p"))

def _make_output_dir(base_output_dir: str, scenario_name: str, node_id: str, tx_limit_mbps: Optional[float]) -> str:
    dirname = "{}_{}_{}_{}".format(
        _sanitize_tag(scenario_name),
        _format_bandwidth_tag(tx_limit_mbps),
        str(node_id),
        _timestamp(),
    )
    path = os.path.join(base_output_dir, dirname)
    os.makedirs(path, exist_ok=True)
    return path

def _filename_model_tag(model_name: str) -> str:
    text = str(model_name).strip().lower()
    if "/" in text:
        text = text.split("/")[-1]
    text = text.replace("-", "_")
    return _sanitize_tag(text)

def _scenario_data_filename(args: argparse.Namespace, codec_name: str, kind: str) -> str:
    model_tag = _filename_model_tag(args.model_name)
    return "{}_{}_bs{}_{}_{}_{}.json".format(
        model_tag,
        TRACE_DATASET_TAG,
        1,
        _format_bandwidth_tag(args.tx_limit_mbps),
        kind,
        str(codec_name),
    )

def _resolve_llmint8_param(policy_name: str, outlier_value: float) -> List[Any]:
    if policy_name not in LLMINT8_POLICY_SPECS:
        raise ValueError(
            "Unsupported llmint8 policy '{}'. Choose from {}".format(
                policy_name,
                sorted(LLMINT8_POLICY_SPECS.keys()),
            )
        )
    policy_spec = LLMINT8_POLICY_SPECS[policy_name]
    return [
        float(outlier_value),
        str(policy_spec["outlier_precision"]),
        str(policy_spec["regular_precision"]),
    ]

def _parse_simulated_codecs(text: str) -> List[str]:
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

def _build_simulation_specs(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    all_specs = {
        "topk": {
            "compressor_name": "topk",
            "compression_param": float(args.topk_param),
        },
        "quantization": {
            "compressor_name": "quantization",
            "compression_param": float(args.quantization_param),
        },
        "llmint8": {
            "compressor_name": "llmint8",
            "compression_param": _resolve_llmint8_param(args.llmint8_policy, float(args.llmint8_outlier)),
        },
    }
    specs = {
        codec_name: dict(all_specs[codec_name])
        for codec_name in _parse_simulated_codecs(args.simulated_codecs)
    }
    for spec in specs.values():
        get_compressor(spec["compressor_name"])
    return specs

def _tensor_to_payload(tensor: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    if tensor is None:
        return None
    return tensor.detach().cpu().numpy()

def _payload_to_tensor(array_data: Optional[np.ndarray], device: str) -> Optional[torch.Tensor]:
    if array_data is None:
        return None
    return torch.from_numpy(array_data).to(device)

def _compress_tensor(tensor: torch.Tensor, compressor_name: str, compression_param: Any) -> Dict[str, Any]:
    compressor = get_compressor(compressor_name)
    tensor_cpu = tensor.detach().cpu()
    started = time.perf_counter()
    compressed = compressor.compress(tensor_cpu, compression_param)
    elapsed = time.perf_counter() - started
    original_bytes = _tensor_payload_bytes(tensor_cpu)
    if hasattr(compressor, "get_compressed_size"):
        compressed_bytes = int(compressor.get_compressed_size(compressed))
    else:
        compressed_bytes = int(compressed.get("compressed_bytes", original_bytes))
    return {
        "payload": compressed,
        "elapsed_sec": float(elapsed),
        "original_bytes": int(original_bytes),
        "compressed_bytes": int(compressed_bytes),
        "compression_ratio": float(original_bytes) / float(compressed_bytes) if compressed_bytes > 0 else 1.0,
    }

def _decompress_tensor(payload: Any, compressor_name: str, device: str) -> Dict[str, Any]:
    compressor = get_compressor(compressor_name)
    started = time.perf_counter()
    tensor = compressor.decompress(payload).to(device)
    elapsed = time.perf_counter() - started
    return {
        "tensor": tensor,
        "elapsed_sec": float(elapsed),
    }

def _simulate_sender_profile(tensor: torch.Tensor, sim_spec: Dict[str, Any]) -> Dict[str, Any]:
    encode_stats = _compress_tensor(tensor, sim_spec["compressor_name"], sim_spec["compression_param"])
    return {
        "compress_sec": float(encode_stats["elapsed_sec"]),
        "decompress_sec": 0.0,
        "codec_adjusted_tau_sec": float(encode_stats["elapsed_sec"]),
        "simulated_output_compressed_bytes": int(encode_stats["compressed_bytes"]),
        "simulated_output_original_bytes": int(encode_stats["original_bytes"]),
        "simulated_output_compression_ratio": float(encode_stats["compression_ratio"]),
    }

def _simulate_receiver_profile(tensor: torch.Tensor, sim_spec: Dict[str, Any], device: str) -> Dict[str, Any]:
    encode_stats = _compress_tensor(tensor, sim_spec["compressor_name"], sim_spec["compression_param"])
    decode_stats = _decompress_tensor(encode_stats["payload"], sim_spec["compressor_name"], device)
    return {
        "compress_sec": 0.0,
        "decompress_sec": float(decode_stats["elapsed_sec"]),
        "codec_adjusted_tau_sec": float(decode_stats["elapsed_sec"]),
        "simulated_input_compressed_bytes": int(encode_stats["compressed_bytes"]),
        "simulated_input_original_bytes": int(encode_stats["original_bytes"]),
        "simulated_input_compression_ratio": float(encode_stats["compression_ratio"]),
    }

def _worker_profile(node_id: str, compute_sec: float, restore_sec: float, prepare_sec: float, send_bytes: int, send_sec: float) -> Dict[str, Any]:
    return {
        "node": node_id,
        "compute_sec": float(compute_sec),
        "restore_sec": float(restore_sec),
        "prepare_send_sec": float(prepare_sec),
        "service_total_sec": float(compute_sec + restore_sec + prepare_sec),
        "send_packet_bytes": int(send_bytes),
        "send_packet_sec": float(send_sec),
    }

def _default_profile(node_id: str) -> Dict[str, Any]:
    return _worker_profile(node_id, 0.0, 0.0, 0.0, 0, 0.0)

def _start_shutdown_listener(transport, shutdown_event: threading.Event):
    def _run():
        while not shutdown_event.is_set():
            try:
                envelope = transport.receive(
                    msg_type=MessageType.CONTROL,
                    source="A",
                    timeout=ASYNC_QUEUE_POLL_SEC,
                )
            except TimeoutError:
                continue
            payload = envelope["payload"]
            if payload.get("cmd") == "shutdown":
                shutdown_event.set()
                return

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread

def _load_partition_bundle(node_id: str, device: str, model_name: str):
    runtime = get_transformers_runtime()
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda was requested, but torch.cuda.is_available() is False.")

    full_model = runtime.T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    full_model.eval()
    tokenizer = runtime.AutoTokenizer.from_pretrained(model_name)
    encoder_layers = len(full_model.encoder.block)
    decoder_layers = len(full_model.decoder.block)
    if encoder_layers != 12 or decoder_layers != 12:
        raise ValueError(
            "This script assumes a 12+12 T5 layout for the fixed 3+1 split, but got "
            "encoder_layers={} decoder_layers={} for model {}".format(
                encoder_layers,
                decoder_layers,
                model_name,
            )
        )

    decoder_start_token_id = getattr(full_model.config, "decoder_start_token_id", None)
    if decoder_start_token_id is None:
        decoder_start_token_id = tokenizer.pad_token_id

    if node_id == "A":
        start, end = ENCODER_SPLITS["A"]
        partition = FlanT5EncoderPartition(
            full_model=full_model,
            layer_indices=list(range(start, end)),
            is_first_partition=True,
            is_last_partition=False,
            runtime=runtime,
        )
    elif node_id == "B":
        start, end = ENCODER_SPLITS["B"]
        partition = FlanT5EncoderPartition(
            full_model=full_model,
            layer_indices=list(range(start, end)),
            is_first_partition=False,
            is_last_partition=False,
            runtime=runtime,
        )
    elif node_id == "C":
        start, end = ENCODER_SPLITS["C"]
        partition = FlanT5EncoderPartition(
            full_model=full_model,
            layer_indices=list(range(start, end)),
            is_first_partition=False,
            is_last_partition=True,
            runtime=runtime,
        )
    elif node_id == "D":
        start, end = DECODER_SPLITS["D"]
        partition = FlanT5DecoderPartition(
            full_model=full_model,
            layer_indices=list(range(start, end)),
            is_first_partition=True,
            is_last_partition=True,
            runtime=runtime,
        )
    else:
        raise ValueError("Unsupported node_id '{}'".format(node_id))

    return partition, tokenizer, int(decoder_start_token_id)

def _dispatch_task_from_a(transport, partition, sample_idx: int, sample, tokenizer, verbalizers, args) -> Dict[str, Any]:
    prompt_text = build_prompt(sample.sentence, args.prompt_template)
    encoded = tokenizer(
        prompt_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=int(args.max_input_length),
    )
    input_ids = encoded.input_ids.to(args.device)
    attention_mask = normalize_attention_mask(
        input_ids,
        encoded.get("attention_mask"),
        args.device,
    )

    task_id = "trace_{}_{}".format(sample.sample_id, uuid.uuid4().hex[:8])
    sim_specs = _build_simulation_specs(args)

    worker_start = time.perf_counter()
    started = time.perf_counter()
    with torch.no_grad():
        hidden, position_bias = partition(
            input_ids,
            attention_mask=attention_mask,
            position_bias=None,
        )
    compute_sec = time.perf_counter() - started

    sim_profiles_a = {}
    for codec_name, sim_spec in sim_specs.items():
        sender_profile = _simulate_sender_profile(hidden, sim_spec)
        sender_profile["codec_adjusted_tau_sec"] = float(compute_sec + sender_profile["compress_sec"])
        sim_profiles_a[codec_name] = sender_profile

    t_prepare = time.perf_counter()
    outgoing = {
        "task_id": task_id,
        "sample_id": sample.sample_id,
        "batch_idx": int(sample_idx),
        "label": int(sample.label),
        "prompt_text": prompt_text,
        "decoder_start_token_id": int(args.decoder_start_token_id),
        "positive_id": int(verbalizers["positive_id"]),
        "negative_id": int(verbalizers["negative_id"]),
        "hidden_comp": _tensor_to_payload(hidden),
        "position_bias": _tensor_to_payload(position_bias),
        "encoder_attention_mask": _tensor_to_payload(attention_mask),
        "tp_original_bytes": [int(_tensor_payload_bytes(hidden)), 0, 0],
    }
    prepare_sec = time.perf_counter() - t_prepare

    send_bytes, send_sec = transport.send("B", MessageType.TASK_INPUT, outgoing)
    profile_a = _worker_profile("A", compute_sec, 0.0, prepare_sec, send_bytes, send_sec)

    return {
        "task_id": task_id,
        "sample": sample,
        "prompt_text": prompt_text,
        "started_at": worker_start,
        "profile_a": profile_a,
        "sim_profiles_a": sim_profiles_a,
        "a0_bytes": int(outgoing["tp_original_bytes"][0]),
        "tp0_bytes": int(send_bytes),
        "tp0_elapsed": float(send_sec),
    }

def _run_worker_b(transport, partition, device: str, sim_specs: Dict[str, Dict[str, Any]], shutdown_event: threading.Event):
    transport.wait_for_peers(["A", "C"], timeout=TRANSPORT_READY_TIMEOUT_SEC)
    while not shutdown_event.is_set():
        try:
            envelope = transport.receive(
                msg_type=MessageType.TASK_INPUT,
                source="A",
                timeout=ASYNC_QUEUE_POLL_SEC,
            )
        except TimeoutError:
            continue

        payload = envelope["payload"]
        worker_start = time.perf_counter()

        t_restore = time.perf_counter()
        hidden = _payload_to_tensor(payload["hidden_comp"], device)
        position_bias = _payload_to_tensor(payload.get("position_bias"), device)
        attention_mask = _payload_to_tensor(payload.get("encoder_attention_mask"), device)
        restore_sec = time.perf_counter() - t_restore

        t_comp = time.perf_counter()
        with torch.no_grad():
            hidden_out, _ = partition(
                hidden,
                attention_mask=attention_mask,
                position_bias=position_bias,
            )
        compute_sec = time.perf_counter() - t_comp

        sim_profiles_b = {}
        for codec_name, sim_spec in sim_specs.items():
            recv_profile = _simulate_receiver_profile(hidden, sim_spec, device)
            send_profile = _simulate_sender_profile(hidden_out, sim_spec)
            sim_profiles_b[codec_name] = {
                "compress_sec": float(send_profile["compress_sec"]),
                "decompress_sec": float(recv_profile["decompress_sec"]),
                "codec_adjusted_tau_sec": float(compute_sec + recv_profile["decompress_sec"] + send_profile["compress_sec"]),
                "simulated_input_compressed_bytes": int(recv_profile["simulated_input_compressed_bytes"]),
                "simulated_input_original_bytes": int(recv_profile["simulated_input_original_bytes"]),
                "simulated_output_compressed_bytes": int(send_profile["simulated_output_compressed_bytes"]),
                "simulated_output_original_bytes": int(send_profile["simulated_output_original_bytes"]),
            }

        t_prepare = time.perf_counter()
        outgoing = dict(payload)
        outgoing["hidden_comp"] = _tensor_to_payload(hidden_out)
        outgoing["position_bias"] = None
        outgoing["tp_original_bytes"] = list(payload["tp_original_bytes"])
        outgoing["tp_original_bytes"][1] = int(_tensor_payload_bytes(hidden_out))
        prepare_sec = time.perf_counter() - t_prepare

        send_bytes, send_sec = transport.send("C", MessageType.ENCODER_OUTPUT, outgoing)
        profile_b = _worker_profile("B", compute_sec, restore_sec, prepare_sec, send_bytes, send_sec)

        transport.send(
            "A",
            MessageType.WORKER_STATS,
            {
                "task_id": payload["task_id"],
                "node_id": "B",
                "tp_index": 1,
                "tp_stats": {"bytes": int(send_bytes), "elapsed": float(send_sec)},
                "profile": profile_b,
                "sim_profiles": sim_profiles_b,
            },
        )

def _run_worker_c(transport, partition, device: str, sim_specs: Dict[str, Dict[str, Any]], shutdown_event: threading.Event):
    transport.wait_for_peers(["A", "B", "D"], timeout=TRANSPORT_READY_TIMEOUT_SEC)
    while not shutdown_event.is_set():
        try:
            envelope = transport.receive(
                msg_type=MessageType.ENCODER_OUTPUT,
                source="B",
                timeout=ASYNC_QUEUE_POLL_SEC,
            )
        except TimeoutError:
            continue

        payload = envelope["payload"]

        t_restore = time.perf_counter()
        hidden = _payload_to_tensor(payload["hidden_comp"], device)
        position_bias = _payload_to_tensor(payload.get("position_bias"), device)
        attention_mask = _payload_to_tensor(payload.get("encoder_attention_mask"), device)
        restore_sec = time.perf_counter() - t_restore

        t_comp = time.perf_counter()
        with torch.no_grad():
            hidden_out, _ = partition(
                hidden,
                attention_mask=attention_mask,
                position_bias=position_bias,
            )
        compute_sec = time.perf_counter() - t_comp

        sim_profiles_c = {}
        for codec_name, sim_spec in sim_specs.items():
            recv_profile = _simulate_receiver_profile(hidden, sim_spec, device)
            send_profile = _simulate_sender_profile(hidden_out, sim_spec)
            sim_profiles_c[codec_name] = {
                "compress_sec": float(send_profile["compress_sec"]),
                "decompress_sec": float(recv_profile["decompress_sec"]),
                "codec_adjusted_tau_sec": float(compute_sec + recv_profile["decompress_sec"] + send_profile["compress_sec"]),
                "simulated_input_compressed_bytes": int(recv_profile["simulated_input_compressed_bytes"]),
                "simulated_input_original_bytes": int(recv_profile["simulated_input_original_bytes"]),
                "simulated_output_compressed_bytes": int(send_profile["simulated_output_compressed_bytes"]),
                "simulated_output_original_bytes": int(send_profile["simulated_output_original_bytes"]),
            }

        t_prepare = time.perf_counter()
        outgoing = dict(payload)
        outgoing["hidden_comp"] = _tensor_to_payload(hidden_out)
        outgoing["position_bias"] = None
        outgoing["tp_original_bytes"] = list(payload["tp_original_bytes"])
        outgoing["tp_original_bytes"][2] = int(_tensor_payload_bytes(hidden_out))
        prepare_sec = time.perf_counter() - t_prepare

        send_bytes, send_sec = transport.send("D", MessageType.ENCODER_OUTPUT, outgoing)
        profile_c = _worker_profile("C", compute_sec, restore_sec, prepare_sec, send_bytes, send_sec)

        transport.send(
            "A",
            MessageType.WORKER_STATS,
            {
                "task_id": payload["task_id"],
                "node_id": "C",
                "tp_index": 2,
                "tp_stats": {"bytes": int(send_bytes), "elapsed": float(send_sec)},
                "profile": profile_c,
                "sim_profiles": sim_profiles_c,
            },
        )

def _run_worker_d(transport, partition, device: str, sim_specs: Dict[str, Dict[str, Any]], shutdown_event: threading.Event):
    transport.wait_for_peers(["A", "C"], timeout=TRANSPORT_READY_TIMEOUT_SEC)
    while not shutdown_event.is_set():
        try:
            envelope = transport.receive(
                msg_type=MessageType.ENCODER_OUTPUT,
                source="C",
                timeout=ASYNC_QUEUE_POLL_SEC,
            )
        except TimeoutError:
            continue

        payload = envelope["payload"]

        t_restore = time.perf_counter()
        encoder_hidden = _payload_to_tensor(payload["hidden_comp"], device)
        attention_mask = _payload_to_tensor(payload.get("encoder_attention_mask"), device)
        decoder_input_ids = torch.tensor(
            [[int(payload["decoder_start_token_id"])]],
            dtype=torch.long,
            device=device,
        )
        restore_sec = time.perf_counter() - t_restore

        t_comp = time.perf_counter()
        with torch.no_grad():
            logits, _, _ = partition(
                decoder_input_ids,
                encoder_hidden,
                payload["task_id"],
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

        sim_profiles_d = {}
        for codec_name, sim_spec in sim_specs.items():
            recv_profile = _simulate_receiver_profile(encoder_hidden, sim_spec, device)
            sim_profiles_d[codec_name] = {
                "compress_sec": 0.0,
                "decompress_sec": float(recv_profile["decompress_sec"]),
                "codec_adjusted_tau_sec": float(compute_sec + recv_profile["decompress_sec"]),
                "simulated_input_compressed_bytes": int(recv_profile["simulated_input_compressed_bytes"]),
                "simulated_input_original_bytes": int(recv_profile["simulated_input_original_bytes"]),
            }

        t_prepare = time.perf_counter()
        final_payload = {
            "task_id": payload["task_id"],
            "sample_id": payload["sample_id"],
            "batch_idx": payload["batch_idx"],
            "label": payload["label"],
            "prompt_text": payload["prompt_text"],
            "predicted_label": int(predicted_label),
            "positive_logit": positive_logit,
            "negative_logit": negative_logit,
            "tp_original_bytes": list(payload["tp_original_bytes"]),
            "sim_profiles_d": sim_profiles_d,
        }
        result_prepare_sec = time.perf_counter() - t_prepare
        profile_d = _worker_profile("D", compute_sec, restore_sec, 0.0, 0, 0.0)
        profile_d["prepare_send_sec"] = float(result_prepare_sec)
        profile_d["service_total_sec"] = float(compute_sec + restore_sec + result_prepare_sec)
        partition.clear_cache(payload["task_id"])
        final_payload["profile_d"] = profile_d

        transport.send("A", MessageType.FINAL_RESULT, final_payload)

def _codec_field_name(codec_name: str, base_name: str, node_name: str) -> str:
    return "{}_{}_node_{}_sec".format(base_name, codec_name, node_name)

def _build_result_row(
    args: argparse.Namespace,
    local_state: Dict[str, Any],
    final_payload: Dict[str, Any],
    worker_stats: Dict[str, Dict[str, Any]],
    sim_specs: Dict[str, Dict[str, Any]],
    running_accuracy: float,
) -> Dict[str, Any]:
    stats_b = worker_stats.get("B", {})
    stats_c = worker_stats.get("C", {})
    profile_b = stats_b.get("profile", _default_profile("B"))
    profile_c = stats_c.get("profile", _default_profile("C"))
    profile_d = final_payload.get("profile_d", _default_profile("D"))
    sim_profiles_a = local_state["sim_profiles_a"]
    sim_profiles_b = stats_b.get("sim_profiles", {})
    sim_profiles_c = stats_c.get("sim_profiles", {})
    sim_profiles_d = final_payload.get("sim_profiles_d", {})
    tp1_stats = stats_b.get("tp_stats", {"bytes": 0, "elapsed": 0.0})
    tp2_stats = stats_c.get("tp_stats", {"bytes": 0, "elapsed": 0.0})

    total_delay = time.perf_counter() - local_state["started_at"]
    a_bytes = [
        int(local_state["tp0_bytes"]),
        int(tp1_stats["bytes"]),
        int(tp2_stats["bytes"]),
    ]
    row = {
        "scenario_name": str(args.scenario_name),
        "batch_idx": int(final_payload["batch_idx"]),
        "task_id": local_state["task_id"],
        "sample_id": str(final_payload["sample_id"]),
        "batch_accuracy": float(int(final_payload["predicted_label"] == int(final_payload["label"]))),
        "running_accuracy": float(running_accuracy),
        "total_delay_sec": float(total_delay),
        "a0_bytes": int(a_bytes[0]),
        "a1_bytes": int(a_bytes[1]),
        "a2_bytes": int(a_bytes[2]),
        "tp0_packet_bytes": int(local_state["tp0_bytes"]),
        "tp1_packet_bytes": int(tp1_stats["bytes"]),
        "tp2_packet_bytes": int(tp2_stats["bytes"]),
        "tp0_send_elapsed_sec": float(local_state["tp0_elapsed"]),
        "tp1_send_elapsed_sec": float(tp1_stats["elapsed"]),
        "tp2_send_elapsed_sec": float(tp2_stats["elapsed"]),
        "c0_true_bps": _safe_rate(a_bytes[0], local_state["tp0_elapsed"]),
        "c1_true_bps": _safe_rate(a_bytes[1], tp1_stats["elapsed"]),
        "c2_true_bps": _safe_rate(a_bytes[2], tp2_stats["elapsed"]),
        "tau_node_A_sec": float(local_state["profile_a"]["service_total_sec"]),
        "tau_node_B_sec": float(profile_b["service_total_sec"]),
        "tau_node_C_sec": float(profile_c["service_total_sec"]),
        "tau_node_D_sec": float(profile_d["service_total_sec"]),
        "tau_residence_node_A_sec": float(local_state["profile_a"]["service_total_sec"]),
        "tau_residence_node_B_sec": float(profile_b["service_total_sec"]),
        "tau_residence_node_C_sec": float(profile_c["service_total_sec"]),
        "tau_residence_node_D_sec": float(profile_d["service_total_sec"]),
        "tau_node_A_compute_sec": float(local_state["profile_a"]["compute_sec"]),
        "tau_node_B_compute_sec": float(profile_b["compute_sec"]),
        "tau_node_C_compute_sec": float(profile_c["compute_sec"]),
        "tau_node_D_compute_sec": float(profile_d["compute_sec"]),
    }

    codec_profile_map = {"A": sim_profiles_a, "B": sim_profiles_b, "C": sim_profiles_c, "D": sim_profiles_d}
    compute_fallbacks = {
        "A": local_state["profile_a"]["compute_sec"],
        "B": profile_b["compute_sec"],
        "C": profile_c["compute_sec"],
        "D": profile_d["compute_sec"],
    }
    residence_fallbacks = {
        "A": local_state["profile_a"]["service_total_sec"],
        "B": profile_b["service_total_sec"],
        "C": profile_c["service_total_sec"],
        "D": profile_d["service_total_sec"],
    }
    for codec_name, spec in sim_specs.items():
        row["sim_{}_compression_param".format(codec_name)] = json.dumps(spec["compression_param"])
        for node_name in ["A", "B", "C", "D"]:
            codec_profile = codec_profile_map.get(node_name, {}).get(codec_name, {})
            row["sim_{}_{}_compress_sec".format(codec_name, node_name)] = float(codec_profile.get("compress_sec", 0.0))
            row["sim_{}_{}_decompress_sec".format(codec_name, node_name)] = float(codec_profile.get("decompress_sec", 0.0))
            row[_codec_field_name(codec_name, "tau_codec", node_name)] = float(codec_profile.get("codec_adjusted_tau_sec", compute_fallbacks[node_name]))
            row[_codec_field_name(codec_name, "tau_residence_codec", node_name)] = float(residence_fallbacks[node_name] + codec_profile.get("compress_sec", 0.0) + codec_profile.get("decompress_sec", 0.0))
        row["sim_{}_tp0_compressed_bytes".format(codec_name)] = int(sim_profiles_a.get(codec_name, {}).get("simulated_output_compressed_bytes", 0))
        row["sim_{}_tp1_compressed_bytes".format(codec_name)] = int(sim_profiles_b.get(codec_name, {}).get("simulated_output_compressed_bytes", 0))
        row["sim_{}_tp2_compressed_bytes".format(codec_name)] = int(sim_profiles_c.get(codec_name, {}).get("simulated_output_compressed_bytes", 0))
    return row

def _build_trace_artifact(args: argparse.Namespace, rows: Sequence[Dict[str, Any]], sim_specs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    batch_index = [int(row["batch_idx"]) for row in rows]
    sample_ids = [str(row["sample_id"]) for row in rows]
    activation_bytes = [[int(row["a0_bytes"]), int(row["a1_bytes"]), int(row["a2_bytes"])] for row in rows]
    tau_node_sec = [[float(row["tau_node_A_sec"]), float(row["tau_node_B_sec"]), float(row["tau_node_C_sec"]), float(row["tau_node_D_sec"])] for row in rows]
    tau_residence_node_sec = [[float(row["tau_residence_node_A_sec"]), float(row["tau_residence_node_B_sec"]), float(row["tau_residence_node_C_sec"]), float(row["tau_residence_node_D_sec"])] for row in rows]
    tau_compute_node_sec = [[float(row["tau_node_A_compute_sec"]), float(row["tau_node_B_compute_sec"]), float(row["tau_node_C_compute_sec"]), float(row["tau_node_D_compute_sec"])] for row in rows]
    channel_true_bps = [[float(row["c0_true_bps"]), float(row["c1_true_bps"]), float(row["c2_true_bps"])] for row in rows]
    observed_total_delay_sec = [float(row["total_delay_sec"]) for row in rows]
    packet_bytes = [[int(row["tp0_packet_bytes"]), int(row["tp1_packet_bytes"]), int(row["tp2_packet_bytes"])] for row in rows]
    send_elapsed_sec = [[float(row["tp0_send_elapsed_sec"]), float(row["tp1_send_elapsed_sec"]), float(row["tp2_send_elapsed_sec"])] for row in rows]

    delay_arr = np.asarray(observed_total_delay_sec, dtype=float) if observed_total_delay_sec else np.zeros((0,), dtype=float)
    c_arr = np.asarray(channel_true_bps, dtype=float) if channel_true_bps else np.zeros((0, NUM_TRANSFER_POINTS), dtype=float)
    a_arr = np.asarray(activation_bytes, dtype=float) if activation_bytes else np.zeros((0, NUM_TRANSFER_POINTS), dtype=float)
    tau_arr = np.asarray(tau_node_sec, dtype=float) if tau_node_sec else np.zeros((0, len(NODE_ORDER)), dtype=float)
    tau_residence_arr = np.asarray(tau_residence_node_sec, dtype=float) if tau_residence_node_sec else np.zeros((0, len(NODE_ORDER)), dtype=float)
    tau_compute_arr = np.asarray(tau_compute_node_sec, dtype=float) if tau_compute_node_sec else np.zeros((0, len(NODE_ORDER)), dtype=float)
    trace_payload = {
        "batch_index": batch_index,
        "sample_ids": sample_ids,
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
        tau_codec_node_sec = [
            [float(row["tau_codec_{}_node_A_sec".format(codec_name)]), float(row["tau_codec_{}_node_B_sec".format(codec_name)]), float(row["tau_codec_{}_node_C_sec".format(codec_name)]), float(row["tau_codec_{}_node_D_sec".format(codec_name)])]
            for row in rows
        ]
        tau_residence_codec_node_sec = [
            [float(row["tau_residence_codec_{}_node_A_sec".format(codec_name)]), float(row["tau_residence_codec_{}_node_B_sec".format(codec_name)]), float(row["tau_residence_codec_{}_node_C_sec".format(codec_name)]), float(row["tau_residence_codec_{}_node_D_sec".format(codec_name)])]
            for row in rows
        ]
        tau_codec_arr = np.asarray(tau_codec_node_sec, dtype=float) if tau_codec_node_sec else np.zeros((0, len(NODE_ORDER)), dtype=float)
        tau_residence_codec_arr = np.asarray(tau_residence_codec_node_sec, dtype=float) if tau_residence_codec_node_sec else np.zeros((0, len(NODE_ORDER)), dtype=float)
        trace_payload[tau_codec_key] = tau_codec_node_sec
        trace_payload[tau_residence_codec_key] = tau_residence_codec_node_sec
        summary_payload["median_{}".format(tau_codec_key)] = np.median(tau_codec_arr, axis=0).tolist() if len(tau_codec_arr) > 0 else []
        summary_payload["avg_{}".format(tau_codec_key)] = np.mean(tau_codec_arr, axis=0).tolist() if len(tau_codec_arr) > 0 else []
        summary_payload["median_{}".format(tau_residence_codec_key)] = np.median(tau_residence_codec_arr, axis=0).tolist() if len(tau_residence_codec_arr) > 0 else []
        summary_payload["avg_{}".format(tau_residence_codec_key)] = np.mean(tau_residence_codec_arr, axis=0).tolist() if len(tau_residence_codec_arr) > 0 else []

    return {
        "schema_version": "scenario_trace_v2",
        "scenario_name": str(args.scenario_name),
        "model_tag": str(args.model_name),
        "task_type": "single_task",
        "trace_source": "flan_t5_no_compression_reference_distributed",
        "topology": {
            "node_order": list(NODE_ORDER),
            "num_nodes": len(NODE_ORDER),
            "num_links": NUM_TRANSFER_POINTS,
        },
        "simulated_codecs": {
            name: {
                "compressor_name": spec["compressor_name"],
                "compression_param": spec["compression_param"],
            }
            for name, spec in sim_specs.items()
        },
        "trace": trace_payload,
        "summary": summary_payload,
    }

def _build_codec_trace_artifact(args: argparse.Namespace, base_artifact: Dict[str, Any], codec_name: str) -> Dict[str, Any]:
    trace = base_artifact["trace"]
    summary = base_artifact["summary"]
    codec_spec = base_artifact["simulated_codecs"][codec_name]
    tau_codec_key = "tau_codec_{}_node_sec".format(codec_name)
    tau_residence_codec_key = "tau_residence_codec_{}_node_sec".format(codec_name)
    median_tau_codec_key = "median_tau_codec_{}_node_sec".format(codec_name)
    median_tau_residence_codec_key = "median_tau_residence_codec_{}_node_sec".format(codec_name)
    avg_tau_codec_key = "avg_tau_codec_{}_node_sec".format(codec_name)
    avg_tau_residence_codec_key = "avg_tau_residence_codec_{}_node_sec".format(codec_name)

    return {
        "schema_version": "scenario_trace_codec_v1",
        "experiment": {
            "scenario_name": str(args.scenario_name),
            "platform": "jetson",
            "model": str(args.model_name),
            "max_input_length": int(args.max_input_length),
            "dataset": {
                "name": TRACE_DATASET_TAG,
                "batch_size": 1,
                "split": str(args.split),
            },
            "device": str(args.device),
            "tx_limit_mbps": None if args.tx_limit_mbps is None else float(args.tx_limit_mbps),
            "tx_bucket_capacity_bytes": int(args.tx_bucket_capacity_bytes),
            "network": {
                "transport_backend": str(args.transport_backend),
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
            "sample_ids": trace["sample_ids"],
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

def _persist_artifacts(args: argparse.Namespace, rows: Sequence[Dict[str, Any]], output_dir: str, sim_specs: Dict[str, Dict[str, Any]], wall_start: float) -> Dict[str, Any]:
    raw_rows_path = os.path.join(output_dir, "no_compression_trace_rows.csv")
    pd.DataFrame.from_records(rows).to_csv(raw_rows_path, index=False)
    base_trace = _build_trace_artifact(args, rows, sim_specs)
    codec_outputs = {}
    for codec_name in sim_specs.keys():
        trace_artifact = _build_codec_trace_artifact(args, base_trace, codec_name)
        scenario_trace_path = os.path.join(output_dir, _scenario_data_filename(args, codec_name, "scenario_trace"))
        trace_summary_path = os.path.join(output_dir, _scenario_data_filename(args, codec_name, "trace_summary"))
        summary = {
            "scenario_name": str(args.scenario_name),
            "output_dir": output_dir,
            "codec_name": codec_name,
            "num_samples": len(rows),
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
            "scenario_trace_path": scenario_trace_path,
            "trace_summary_path": trace_summary_path,
        }

    return {
        "scenario_name": str(args.scenario_name),
        "output_dir": output_dir,
        "raw_rows_path": raw_rows_path,
        "codec_outputs": codec_outputs,
    }

def _drain_worker_stats_for_task(transport, task_id: str, worker_stats: Dict[str, Dict[str, Any]]):
    while True:
        try:
            envelope = transport.receive(
                msg_type=MessageType.WORKER_STATS,
                source=None,
                predicate=lambda payload: payload.get("task_id") == task_id,
                timeout=0.0,
            )
        except TimeoutError:
            return
        payload = envelope["payload"]
        worker_stats[payload["node_id"]] = payload

def _drain_worker_stats(transport, pending: Dict[str, Dict[str, Any]], buffered_stats: Dict[str, Dict[str, Any]]):
    while True:
        try:
            envelope = transport.receive(
                msg_type=MessageType.WORKER_STATS,
                source=None,
                timeout=0.0,
            )
        except TimeoutError:
            return
        payload = envelope["payload"]
        task_id = str(payload["task_id"])
        if task_id in pending:
            pending[task_id]["worker_stats"][payload["node_id"]] = payload
        else:
            if task_id not in buffered_stats:
                buffered_stats[task_id] = {}
            buffered_stats[task_id][payload["node_id"]] = payload

def _collect_result_payload(transport):
    try:
        return transport.receive(
            msg_type=MessageType.FINAL_RESULT,
            source="D",
            timeout=0.0,
        )
    except TimeoutError:
        return None

def _run_node_a(args: argparse.Namespace):
    partition, tokenizer, decoder_start_token_id = _load_partition_bundle("A", args.device, args.model_name)
    args.decoder_start_token_id = decoder_start_token_id
    verbalizers = resolve_single_token_verbalizers(
        tokenizer,
        positive_text=args.positive_token,
        negative_text=args.negative_token,
    )
    samples = load_sst2_samples(dataset_path=args.dataset_path, split=args.split)
    if args.max_samples is not None and int(args.max_samples) > 0:
        samples = samples[: int(args.max_samples)]

    transport = create_transport(
        "A",
        NODE_IPS,
        NODE_PORTS,
        backend=args.transport_backend,
        tx_limit_mbps=args.tx_limit_mbps,
        tx_bucket_capacity_bytes=args.tx_bucket_capacity_bytes,
    )
    output_dir = _make_output_dir(args.output_dir, args.scenario_name, args.node, args.tx_limit_mbps)
    sim_specs = _build_simulation_specs(args)
    wall_start = time.perf_counter()

    transport.start()
    transport.wait_for_peers(["B", "C", "D"], timeout=TRANSPORT_READY_TIMEOUT_SEC)
    logging.info("[A] All workers are ready")

    rows = []
    pending: Dict[str, Dict[str, Any]] = {}
    buffered_stats: Dict[str, Dict[str, Any]] = {}
    total_correct = 0
    dispatched = 0
    completed = 0
    try:
        while completed < len(samples):
            _drain_worker_stats(transport, pending, buffered_stats)
            while dispatched < len(samples) and len(pending) < int(args.max_inflight_tasks):
                sample = samples[dispatched]
                local_state = _dispatch_task_from_a(transport, partition, dispatched, sample, tokenizer, verbalizers, args)
                local_state["sample_idx"] = int(dispatched)
                local_state["worker_stats"] = {}
                if local_state["task_id"] in buffered_stats:
                    local_state["worker_stats"].update(buffered_stats.pop(local_state["task_id"]))
                pending[local_state["task_id"]] = local_state
                dispatched += 1

            final_envelope = _collect_result_payload(transport)
            if final_envelope is None:
                time.sleep(ASYNC_QUEUE_POLL_SEC)
                continue

            final_payload = final_envelope["payload"]
            task_id = str(final_payload["task_id"])
            if task_id not in pending:
                continue

            local_state = pending.pop(task_id)
            if task_id in buffered_stats:
                local_state["worker_stats"].update(buffered_stats.pop(task_id))

            total_correct += int(final_payload["predicted_label"] == int(final_payload["label"]))
            row = _build_result_row(
                args=args,
                local_state=local_state,
                final_payload=final_payload,
                worker_stats=local_state["worker_stats"],
                sim_specs=sim_specs,
                running_accuracy=float(total_correct) / float(completed + 1),
            )
            rows.append(row)
            logging.info(
                "[A] sample=%d/%d id=%s delay=%.4fs acc=%.4f c_true=[%.3e, %.3e, %.3e]",
                completed + 1,
                len(samples),
                final_payload["sample_id"],
                float(row["total_delay_sec"]),
                float(row["batch_accuracy"]),
                float(row["c0_true_bps"]),
                float(row["c1_true_bps"]),
                float(row["c2_true_bps"]),
            )
            completed += 1
    finally:
        transport.broadcast(["B", "C", "D"], MessageType.CONTROL, {"cmd": "shutdown"})
        time.sleep(1.0)
        transport.close()

    summary = _persist_artifacts(args, rows, output_dir, sim_specs, wall_start)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

def _run_worker(args: argparse.Namespace):
    partition, _, _ = _load_partition_bundle(args.node, args.device, args.model_name)
    sim_specs = _build_simulation_specs(args)
    transport = create_transport(
        args.node,
        NODE_IPS,
        NODE_PORTS,
        backend=args.transport_backend,
        tx_limit_mbps=args.tx_limit_mbps,
        tx_bucket_capacity_bytes=args.tx_bucket_capacity_bytes,
    )
    transport.start()
    shutdown_event = threading.Event()
    shutdown_thread = _start_shutdown_listener(transport, shutdown_event)
    try:
        if args.node == "B":
            _run_worker_b(transport, partition, args.device, sim_specs, shutdown_event)
        elif args.node == "C":
            _run_worker_c(transport, partition, args.device, sim_specs, shutdown_event)
        elif args.node == "D":
            _run_worker_d(transport, partition, args.device, sim_specs, shutdown_event)
        else:
            raise ValueError("Worker node must be one of B/C/D")
    finally:
        shutdown_event.set()
        shutdown_thread.join(timeout=1.0)
        transport.close()

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect Flan-T5 no-compression trace data plus simulated codec overhead.")
    parser.add_argument("--node", choices=["A", "B", "C", "D"], required=True)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--max_samples", type=int, default=100, help="0 or negative means use the full dataset.")
    parser.add_argument("--prompt_template", default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--max_input_length", type=int, default=DEFAULT_MAX_INPUT_LENGTH)
    parser.add_argument("--positive_token", default=DEFAULT_POSITIVE_TOKEN)
    parser.add_argument("--negative_token", default=DEFAULT_NEGATIVE_TOKEN)
    parser.add_argument("--transport_backend", default=TRANSPORT_BACKEND)
    parser.add_argument("--tx_limit_mbps", type=float, default=DEFAULT_TX_LIMIT_MBPS)
    parser.add_argument("--tx_bucket_capacity_bytes", type=int, default=DEFAULT_TX_BUCKET_CAPACITY_BYTES)
    parser.add_argument("--scenario_name", default="flan_t5_no_compression_trace")
    parser.add_argument("--output_dir", default=BASE_OUTPUT_DIR)
    parser.add_argument("--topk_param", type=float, default=DEFAULT_TOPK_PARAM)
    parser.add_argument("--quantization_param", type=float, default=DEFAULT_QUANTIZATION_PARAM)
    parser.add_argument("--llmint8_policy", default=DEFAULT_LLMINT8_POLICY)
    parser.add_argument("--llmint8_outlier", type=float, default=DEFAULT_LLMINT8_OUTLIER)
    parser.add_argument("--simulated_codecs", default=DEFAULT_SIMULATED_CODECS,
                        help="Comma-separated codec list. Default runs topk, quantization, and llmint8.")
    parser.add_argument("--max_inflight_tasks", type=int, default=DEFAULT_MAX_INFLIGHT_TASKS)
    parser.add_argument("--log_level", default="INFO")
    return parser

def main():
    parser = _build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    if args.node == "A":
        _run_node_a(args)
    else:
        _run_worker(args)
