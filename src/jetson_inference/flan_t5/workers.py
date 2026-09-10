"""worker execution, message processing and node lifecycle."""

import logging
import time
import torch

from jetson_inference.common.http_transport import (
    MessageType,
    create_transport,
)
from jetson_inference.flan_t5.compression import (
    build_activation_payload,
    restore_activation_payload,
)
from jetson_inference.flan_t5.config import (
    DEFAULT_MAX_INPUT_LENGTH,
    DEFAULT_MODEL_NAME,
    DEFAULT_TX_BUCKET_CAPACITY_BYTES,
    DEFAULT_TX_LIMIT_MBPS,
    NODE_IPS,
    NODE_PORTS,
    TRANSPORT_BACKEND,
    TRANSPORT_READY_TIMEOUT_SEC,
)
from jetson_inference.flan_t5.model import FlanT5PartitionFactory


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

def _dispatch_async_task(transport, partition, batch, task_id, execution_plan, device, decoder_start_token_id, verbalizers):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
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
        "task_id": task_id,
        "batch_idx": batch["batch_idx"],
        "sample_id": batch["sample_id"],
        "label": int(batch["label"]),
        "prompt_text": batch["prompt_text"],
        "decoder_start_token_id": int(decoder_start_token_id),
        "positive_id": int(verbalizers["positive_id"]),
        "negative_id": int(verbalizers["negative_id"]),
        "compressor_name": compressor_name,
        "compression_params_list": compression_params_list,
        "feature_k_values": exec_eta,
        "requested_eta": list(execution_plan["requested_eta"]),
        "mapping_info": execution_plan.get("mapping_info"),
        "activation_comp": hidden_comp,
        "position_bias": None if position_bias is None else position_bias.detach().cpu(),
        "encoder_attention_mask": attention_mask.detach().cpu(),
        "tp_original_bytes": [comp_stats["original_bytes"], 0, 0],
        "worker_stats_chain": {},
    }
    prepare_sec = time.perf_counter() - t_prepare
    bytes_sent, send_sec = transport.send("B", MessageType.TASK_INPUT, outgoing)
    a_profile = _worker_profile("A", compute_sec, 0.0, prepare_sec, bytes_sent, send_sec)
    return {
        "task_id": task_id,
        "batch_idx": batch["batch_idx"],
        "sample_id": batch["sample_id"],
        "label": int(batch["label"]),
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
        position_bias = payload.get("position_bias")
        if position_bias is not None:
            position_bias = position_bias.to(device)
        attention_mask = payload.get("encoder_attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        restore_sec = time.perf_counter() - t_restore
        t_comp = time.perf_counter()
        with torch.no_grad():
            hidden_out, next_position_bias = partition(
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
        prepare_sec = time.perf_counter() - t_prepare
        outgoing["worker_stats_chain"] = dict(payload.get("worker_stats_chain", {}))
        outgoing["worker_stats_chain"]["A"] = _receiver_side_tp_stats(envelope, "A", task_id=payload["task_id"])
        outgoing["worker_stats_chain"]["B"] = {
            "task_id": payload["task_id"],
            "node_id": "B",
            "profile": _worker_profile("B", compute_sec, restore_sec, prepare_sec, 0, 0.0),
        }
        bytes_sent, send_sec = transport.send("C", MessageType.ENCODER_OUTPUT, outgoing)

def _run_worker_c(transport, partition, device):
    while True:
        envelope = transport.receive(msg_type=MessageType.ENCODER_OUTPUT, source=None)
        payload = envelope["payload"]
        if payload.get("cmd") == "shutdown":
            break
        upstream_stats = _receiver_side_tp_stats(
            envelope,
            "B",
            profile=payload.get("worker_stats_chain", {}).get("B", {}).get("profile"),
            task_id=payload["task_id"],
        )
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
            hidden_out, _ = partition(
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
        prepare_sec = time.perf_counter() - t_prepare
        outgoing["worker_stats_chain"] = dict(payload.get("worker_stats_chain", {}))
        outgoing["worker_stats_chain"]["B"] = upstream_stats
        outgoing["worker_stats_chain"]["C"] = {
            "task_id": payload["task_id"],
            "node_id": "C",
            "profile": _worker_profile("C", compute_sec, restore_sec, prepare_sec, 0, 0.0),
        }
        bytes_sent, send_sec = transport.send("D", MessageType.DECODER_STEP, outgoing)

def _run_worker_d(transport, partition, device):
    while True:
        envelope = transport.receive(msg_type=MessageType.DECODER_STEP, source=None)
        payload = envelope["payload"]
        if payload.get("cmd") == "shutdown":
            break
        upstream_stats = _receiver_side_tp_stats(
            envelope,
            "C",
            profile=payload.get("worker_stats_chain", {}).get("C", {}).get("profile"),
            task_id=payload["task_id"],
        )
        t_restore = time.perf_counter()
        encoder_hidden = restore_activation_payload(payload["activation_comp"], device)
        attention_mask = payload.get("encoder_attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        decoder_input_ids = torch.tensor([[int(payload["decoder_start_token_id"])]], dtype=torch.long, device=device)
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
        t_prepare = time.perf_counter()
        result = {
            "task_id": payload["task_id"],
            "batch_idx": payload["batch_idx"],
            "sample_id": payload["sample_id"],
            "label": payload["label"],
            "prompt_text": payload["prompt_text"],
            "requested_eta": payload["requested_eta"],
            "executed_eta": payload["feature_k_values"],
            "compressor_name": payload["compressor_name"],
            "compression_params_list": payload["compression_params_list"],
            "mapping_info": payload.get("mapping_info"),
            "predicted_label": int(predicted_label),
            "positive_logit": positive_logit,
            "negative_logit": negative_logit,
            "tp_original_bytes": list(payload["tp_original_bytes"]),
            "worker_stats_chain": dict(payload.get("worker_stats_chain", {})),
            "profile_d": None,
        }
        prepare_sec = time.perf_counter() - t_prepare
        result["worker_stats_chain"]["C"] = upstream_stats
        result["profile_d"] = _worker_profile("D", compute_sec, restore_sec, prepare_sec, 0, 0.0)
        partition.clear_cache(payload["task_id"])
        transport.send("A", MessageType.FINAL_RESULT, result)

def run_worker(node_id, device="cpu", model_name=DEFAULT_MODEL_NAME, max_input_length=DEFAULT_MAX_INPUT_LENGTH, transport_backend=TRANSPORT_BACKEND, tx_limit_mbps=DEFAULT_TX_LIMIT_MBPS, tx_bucket_capacity_bytes=DEFAULT_TX_BUCKET_CAPACITY_BYTES):
    if node_id == "A":
        raise ValueError("run_worker only supports B/C/D")
    factory = FlanT5PartitionFactory(model_name=model_name, device=device, max_input_length=max_input_length)
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
    if node_id == "B":
        transport.wait_for_peers(["A", "C"], timeout=TRANSPORT_READY_TIMEOUT_SEC)
    elif node_id == "C":
        transport.wait_for_peers(["A", "B", "D"], timeout=TRANSPORT_READY_TIMEOUT_SEC)
    else:
        transport.wait_for_peers(["A", "C"], timeout=TRANSPORT_READY_TIMEOUT_SEC)
    logging.info("[%s] Worker ready with partition %s", node_id, factory.partition_plan[node_id])
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
