"""worker execution, message processing and node lifecycle."""

import logging
import numpy as np
import pickle
import queue
import threading
import time
import torch
import uuid

from jetson_inference.common.http_transport_dynamic import MessageType
from jetson_inference.flan_t5.config import (
    DEFAULT_MAX_INPUT_LENGTH as FLAN_DEFAULT_MAX_INPUT_LENGTH,
    DEFAULT_MODEL_NAME as FLAN_DEFAULT_MODEL_NAME,
    DEFAULT_NEGATIVE_TOKEN as FLAN_DEFAULT_NEGATIVE_TOKEN,
    DEFAULT_POSITIVE_TOKEN as FLAN_DEFAULT_POSITIVE_TOKEN,
)
from jetson_inference.flan_t5.model import (
    FlanT5PartitionFactory,
    resolve_single_token_verbalizers,
)
from jetson_inference.multi_task.codec import (
    build_activation_payload,
    restore_activation_payload,
)
from jetson_inference.multi_task.config import (
    DEFAULT_DEVICE,
    DEFAULT_TX_BUCKET_CAPACITY_BYTES,
    DEFAULT_WORKER_THREADS,
    TRANSPORT_BACKEND,
    TRANSPORT_READY_TIMEOUT_SEC,
)
from jetson_inference.multi_task.state import RuntimeTaskDef
from jetson_inference.multi_task.tasks import (
    _build_execution_plan,
    _build_worker_task_defs,
)
from jetson_inference.multi_task.transport import (
    ThreadSafeTransportSender,
    _create_multitask_transport,
)
from jetson_inference.resnet.config import MODEL_CHECKPOINT as RESNET_CHECKPOINT_PATH
from jetson_inference.resnet.model import ResNet56PartitionFactory
from typing import (
    Any,
    Dict,
    Sequence,
    Tuple,
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
