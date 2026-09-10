"""worker execution, message processing and node lifecycle."""

import logging
import numpy as np
import time
import torch

from jetson_inference.common.http_transport import (
    MessageType,
    create_transport,
)
from jetson_inference.resnet.compression import (
    build_activation_payload,
    restore_activation_payload,
)
from jetson_inference.resnet.config import (
    DEFAULT_TX_BUCKET_CAPACITY_BYTES,
    DEFAULT_TX_LIMIT_MBPS,
    MODEL_CHECKPOINT,
    NODE_IPS,
    NODE_PORTS,
    TRANSPORT_BACKEND,
    TRANSPORT_READY_TIMEOUT_SEC,
)
from jetson_inference.resnet.model import ResNet56PartitionFactory


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
