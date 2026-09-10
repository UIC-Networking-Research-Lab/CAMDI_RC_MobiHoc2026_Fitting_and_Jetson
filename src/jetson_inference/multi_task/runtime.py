"""Warmup, measurements and asynchronous control loops."""

import copy
import logging
import numpy as np
import pandas as pd
import time
import torch
import uuid

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from jetson_inference.common.http_transport_dynamic import MessageType
from jetson_inference.multi_task.channel import (
    _current_total_bandwidth_bps,
    _warn_if_per_link_bandwidth_low,
)
from jetson_inference.multi_task.config import (
    NODE_ORDER,
    NUM_TRANSFER_POINTS,
    _format_float4,
)
from jetson_inference.multi_task.policies import (
    _policy_uses_estimator,
    _policy_uses_fixed_channel,
)
from jetson_inference.multi_task.state import (
    LatestSnapshot,
    _clone_allocations,
    _compute_task_predictions,
)
from jetson_inference.multi_task.workers import (
    _dispatch_async_request,
    _finalize_request,
)
from typing import Sequence


def _build_snapshot_from_completion(meta, aggregate_stats):
    tp_original_bytes = np.asarray(meta["tp_original_bytes"], dtype=float)
    tp_compressed_bytes = np.asarray(meta["tp_compressed_bytes"], dtype=float)
    worker_stats_chain = dict(meta.get("worker_stats_chain", {}))
    stats_a = worker_stats_chain.get("A") or {}
    stats_b = worker_stats_chain.get("B") or {}
    stats_c = worker_stats_chain.get("C") or {}
    stats_d = worker_stats_chain.get("D") or {}
    tp_stats = {
        "A": dict(stats_a.get("tp_stats", {})),
        "B": dict(stats_b.get("tp_stats", {})),
        "C": dict(stats_c.get("tp_stats", {})),
    }
    profiles = {
        "A": dict(stats_a.get("profile", {})),
        "B": dict(stats_b.get("profile", {})),
        "C": dict(stats_c.get("profile", {})),
        "D": dict(stats_d.get("profile", {})),
    }
    end_to_end_delay_sec = float(time.perf_counter() - float(meta["started_at"]))
    tp_bandwidth_bps = []
    fixed_overhead_bytes = []
    link_elapsed_values = []
    for node_id in ["A", "B", "C"]:
        stats = tp_stats.get(node_id, {})
        bytes_sent = max(float(stats.get("bytes", 0.0)), 0.0)
        elapsed = max(float(stats.get("elapsed", 0.0)), 1e-9)
        tp_bandwidth_bps.append((bytes_sent * 8.0) / elapsed)
        link_elapsed_values.append(float(stats.get("elapsed", 0.0)))
    for link_idx, node_id in enumerate(["A", "B", "C"]):
        stats = tp_stats.get(node_id, {})
        bytes_sent = max(float(stats.get("bytes", 0.0)), 0.0)
        compressed_bytes = max(float(tp_compressed_bytes[link_idx]), 0.0)
        fixed_overhead_bytes.append(max(0.0, bytes_sent - compressed_bytes))
    tau_list = []
    for node_id in NODE_ORDER:
        profile = profiles.get(node_id) or {}
        tau_list.append(float(profile.get("service_total_sec", 0.0)))
    service_delay_sec = float(np.max(np.asarray(tau_list, dtype=float))) if tau_list else 0.0
    communication_delay_sec = (
        float(np.max(np.asarray(link_elapsed_values, dtype=float))) if link_elapsed_values else 0.0
    )
    total_delay_sec = max(float(service_delay_sec), float(communication_delay_sec))
    miscellaneous_delay_sec = 0.0
    task_stats = aggregate_stats[int(meta["logical_task_id"])]
    return LatestSnapshot(
        a_ref=np.asarray(tp_original_bytes, dtype=float),
        fixed_overhead_bytes=np.asarray(fixed_overhead_bytes, dtype=float),
        tau_list=np.asarray(tau_list, dtype=float),
        total_delay_sec=total_delay_sec,
        end_to_end_delay_sec=end_to_end_delay_sec,
        tp_bandwidth_bps=np.asarray(tp_bandwidth_bps, dtype=float),
        sample_correct=float(task_stats["last_correct"]),
        running_accuracy=float(task_stats["correct"]) / max(float(task_stats["seen"]), 1.0),
        service_delay_sec=service_delay_sec,
        communication_delay_sec=communication_delay_sec,
        miscellaneous_delay_sec=miscellaneous_delay_sec,
    )

def _mean_snapshot_from_list(snapshot_list: Sequence[LatestSnapshot]) -> LatestSnapshot:
    if not snapshot_list:
        raise ValueError("Expected non-empty snapshot_list")

    def _mean_array(attr_name):
        return np.mean(
            np.stack([np.asarray(getattr(item, attr_name), dtype=float) for item in snapshot_list], axis=0),
            axis=0,
        )

    def _mean_scalar(attr_name):
        return float(np.mean(np.asarray([getattr(item, attr_name) for item in snapshot_list], dtype=float)))

    return LatestSnapshot(
        a_ref=np.asarray(_mean_array("a_ref"), dtype=float),
        fixed_overhead_bytes=np.asarray(_mean_array("fixed_overhead_bytes"), dtype=float),
        tau_list=np.asarray(_mean_array("tau_list"), dtype=float),
        total_delay_sec=_mean_scalar("total_delay_sec"),
        end_to_end_delay_sec=_mean_scalar("end_to_end_delay_sec"),
        tp_bandwidth_bps=np.asarray(_mean_array("tp_bandwidth_bps"), dtype=float),
        sample_correct=_mean_scalar("sample_correct"),
        running_accuracy=_mean_scalar("running_accuracy"),
        service_delay_sec=_mean_scalar("service_delay_sec"),
        communication_delay_sec=_mean_scalar("communication_delay_sec"),
        miscellaneous_delay_sec=_mean_scalar("miscellaneous_delay_sec"),
    )

def _aggregate_control_window_snapshots(control_records, tasks, latest_snapshots):
    grouped = {int(task.logical_task_id): [] for task in tasks}
    for item in control_records:
        grouped[int(item["logical_task_id"])].append(item["snapshot"])
    averaged = {}
    for task in tasks:
        tid = int(task.logical_task_id)
        snapshots_for_task = grouped.get(tid, [])
        if snapshots_for_task:
            averaged[tid] = _mean_snapshot_from_list(snapshots_for_task)
        elif tid in latest_snapshots:
            averaged[tid] = latest_snapshots[tid]
    return averaged

def _aggregate_control_window_total_bandwidth_bps(control_records, tasks, latest_snapshots):
    grouped = {int(task.logical_task_id): [] for task in tasks}
    for item in control_records:
        grouped[int(item["logical_task_id"])].append(np.asarray(item["snapshot"].tp_bandwidth_bps, dtype=float))
    total_bw_bps = np.zeros(NUM_TRANSFER_POINTS, dtype=float)
    for task in tasks:
        tid = int(task.logical_task_id)
        bandwidth_rows = grouped.get(tid, [])
        if bandwidth_rows:
            task_mean_bw_bps = np.mean(np.stack(bandwidth_rows, axis=0), axis=0)
        elif tid in latest_snapshots:
            task_mean_bw_bps = np.asarray(latest_snapshots[tid].tp_bandwidth_bps, dtype=float)
        else:
            task_mean_bw_bps = np.zeros(NUM_TRANSFER_POINTS, dtype=float)
        total_bw_bps += np.asarray(task_mean_bw_bps, dtype=float)
    return np.asarray(total_bw_bps, dtype=float)

def _tp_stats_from_total_bandwidth_bps(total_bw_bps):
    return [(float(bps) / 8.0, 1.0) for bps in np.asarray(total_bw_bps, dtype=float)]

def _predict_total_bandwidth_bps_from_estimator(channel_estimator, fallback_total_bw_bps):
    if channel_estimator is None:
        return np.asarray(fallback_total_bw_bps, dtype=float)
    try:
        has_history = any(len(link.history) > 0 for link in channel_estimator.links)
    except Exception:
        has_history = False
    if not has_history:
        return np.asarray(fallback_total_bw_bps, dtype=float)
    predicted_bytes_per_sec = np.asarray(channel_estimator.predict_all(), dtype=float)
    return predicted_bytes_per_sec * 8.0

def _solve_multi_policy_snapshot(policy, averaged_snapshots, total_bw_bps, tasks_by_id, window_id):
    started = time.perf_counter()
    policy_copy = copy.deepcopy(policy)
    actual_delay_by_task = {
        int(tid): float(snapshot.total_delay_sec)
        for tid, snapshot in averaged_snapshots.items()
    }
    policy_copy.update_dual(actual_delay_by_task, tasks_by_id)
    allocations = policy_copy.current_allocations(averaged_snapshots, np.asarray(total_bw_bps, dtype=float))
    predicted_accuracy_by_task, predicted_delay_by_task = _compute_task_predictions(
        tasks_by_id,
        allocations,
        averaged_snapshots,
        total_bw_bps,
        enable_accuracy_prediction=_policy_uses_estimator(policy=policy_copy, algorithm_type=getattr(policy_copy, "policy_key", None)),
    )
    return {
        "window_id": int(window_id),
        "allocations": {
            int(tid): {
                "eta": np.asarray(alloc["eta"], dtype=float),
                "s_comm": np.asarray(alloc["s_comm"], dtype=float),
                "s_comp": np.asarray(alloc["s_comp"], dtype=float),
            }
            for tid, alloc in allocations.items()
        },
        "lambda_k": copy.deepcopy(getattr(policy_copy, "lambda_k", {})),
        "actual_delay_by_task": actual_delay_by_task,
        "control_solve_sec": float(time.perf_counter() - started),
        "total_bw_bps": np.asarray(total_bw_bps, dtype=float),
        "averaged_snapshots": averaged_snapshots,
        "predicted_accuracy_by_task": dict(predicted_accuracy_by_task),
        "predicted_delay_by_task": dict(predicted_delay_by_task),
    }

def _compute_control_record(
    t_index,
    algorithm_name,
    algorithm_type,
    policy,
    tasks,
    allocations,
    latest_snapshots,
    current_total_bw_bps,
    aggregate_stats=None,
    control_solve_sec=None,
    window_id=None,
    dynamic_timeslot_size=None,
    actual_window_size=None,
    anchor_task_id=None,
    predicted_accuracy_by_task=None,
    predicted_delay_by_task=None,
):
    weighted_accuracy_true = 0.0
    weight_sum = sum(float(task.weight) for task in tasks)
    avg_delay_actual_sec = float(np.mean([latest_snapshots[int(task.logical_task_id)].total_delay_sec for task in tasks]))
    if predicted_accuracy_by_task is None:
        predicted_accuracy_by_task = {}
    if predicted_delay_by_task is None:
        predicted_delay_by_task = {}
    weighted_accuracy_fit_values = []
    pred_delay_values = []
    pred_excess_values = []
    pred_ratio_values = []
    avg_excess_delay_actual_sec = float(
        np.mean(
            [
                latest_snapshots[int(task.logical_task_id)].total_delay_sec - (1.0 / float(task.target_rate_hz))
                if float(task.target_rate_hz) > 0.0 else np.nan
                for task in tasks
            ]
        )
    )
    avg_delay_ratio_actual = float(
        np.mean(
            [
                latest_snapshots[int(task.logical_task_id)].total_delay_sec * float(task.target_rate_hz)
                for task in tasks
            ]
        )
    )
    anchor_task = None
    if tasks:
        if anchor_task_id is None:
            anchor_task = tasks[0]
        else:
            for task in tasks:
                if int(task.logical_task_id) == int(anchor_task_id):
                    anchor_task = task
                    break
        if anchor_task is None:
            anchor_task = tasks[0]
    record = {
        "t": int(t_index),
        "algorithm": algorithm_name,
        "policy": algorithm_name,
        "policy_key": algorithm_type,
        "algorithm_type": algorithm_type,
        "mu": float(policy.mu) if policy.mu is not None else np.nan,
        "control_solve_sec": (float(control_solve_sec) if control_solve_sec is not None else np.nan),
        "weighted_accuracy_fit": np.nan,
        "weighted_accuracy_true": 0.0,
        "weighted_utility_fit": np.nan,
        "weighted_utility_true": 0.0,
        "avg_delay_actual_sec": avg_delay_actual_sec,
        "avg_delay_pred_sec": np.nan,
        "avg_excess_delay_actual_sec": avg_excess_delay_actual_sec,
        "avg_excess_delay_pred_sec": np.nan,
        "avg_delay_ratio_actual": avg_delay_ratio_actual,
        "avg_delay_ratio_pred": np.nan,
        "window_id": (int(window_id) if window_id is not None else np.nan),
        "dynamic_timeslot_size": (int(dynamic_timeslot_size) if dynamic_timeslot_size is not None else np.nan),
        "actual_window_size": (int(actual_window_size) if actual_window_size is not None else np.nan),
        "anchor_task_id": (int(anchor_task.logical_task_id) if anchor_task is not None else np.nan),
    }
    if aggregate_stats is not None:
        record["completion_total"] = int(
            sum(int(aggregate_stats[int(task.logical_task_id)]["seen"]) for task in tasks)
        )
    for task in tasks:
        tid = int(task.logical_task_id)
        snap = latest_snapshots[tid]
        alloc = allocations[tid]
        weighted_accuracy_true += float(task.weight) * float(snap.running_accuracy)
        eta = np.asarray(alloc["eta"], dtype=float)
        s_comm = np.asarray(alloc["s_comm"], dtype=float)
        s_comp = np.asarray(alloc["s_comp"], dtype=float)
        predicted_accuracy = float(predicted_accuracy_by_task.get(int(tid), np.nan))
        predicted_delay = float(predicted_delay_by_task.get(int(tid), np.nan))
        if not pd.isna(predicted_accuracy):
            weighted_accuracy_fit_values.append(float(task.weight) * predicted_accuracy)
        if not pd.isna(predicted_delay):
            predicted_excess = (
                predicted_delay - (1.0 / float(task.target_rate_hz))
                if float(task.target_rate_hz) > 0.0 else np.nan
            )
            predicted_ratio = predicted_delay * float(task.target_rate_hz) if float(task.target_rate_hz) > 0.0 else np.nan
            pred_delay_values.append(predicted_delay)
            pred_excess_values.append(predicted_excess)
            pred_ratio_values.append(predicted_ratio)
        else:
            predicted_excess = np.nan
            predicted_ratio = np.nan
        if aggregate_stats is not None:
            record["completion_task{}".format(tid)] = int(aggregate_stats[tid]["seen"])
        record["target_rate_hz_task{}".format(tid)] = float(task.target_rate_hz)
        record["accuracy_fit_task{}".format(tid)] = predicted_accuracy
        record["accuracy_true_task{}".format(tid)] = float(snap.running_accuracy)
        record["delay_actual_sec_task{}".format(tid)] = float(snap.total_delay_sec)
        record["end_to_end_delay_sec_task{}".format(tid)] = float(snap.end_to_end_delay_sec)
        record["delay_pred_sec_task{}".format(tid)] = predicted_delay
        record["excess_delay_sec_task{}".format(tid)] = (
            float(snap.total_delay_sec) - (1.0 / float(task.target_rate_hz))
            if float(task.target_rate_hz) > 0.0 else np.nan
        )
        record["excess_delay_pred_sec_task{}".format(tid)] = predicted_excess
        record["delay_ratio_actual_task{}".format(tid)] = float(snap.total_delay_sec) * float(task.target_rate_hz)
        record["delay_ratio_pred_task{}".format(tid)] = predicted_ratio
        record["service_delay_sec_task{}".format(tid)] = float(snap.service_delay_sec)
        record["communication_delay_sec_task{}".format(tid)] = float(snap.communication_delay_sec)
        record["miscellaneous_delay_sec_task{}".format(tid)] = float(snap.miscellaneous_delay_sec)
        record["lambda_task{}".format(tid)] = float(getattr(policy, "lambda_k", {}).get(tid, 0.0))
        record["w_k_task{}".format(tid)] = float(task.weight)
        record["sample_correct_task{}".format(tid)] = float(snap.sample_correct)
        for link_idx, value in enumerate(eta):
            record["eta_task{}_{}".format(tid, link_idx)] = float(value)
        for link_idx, value in enumerate(s_comm):
            record["s_comm_task{}_{}".format(tid, link_idx)] = float(value)
        for node_idx, value in enumerate(s_comp):
            record["s_comp_task{}_{}".format(tid, node_idx)] = float(value)
    if weighted_accuracy_fit_values:
        record["weighted_accuracy_fit"] = float(np.sum(np.asarray(weighted_accuracy_fit_values, dtype=float)) / max(weight_sum, 1e-12))
    record["weighted_accuracy_true"] = float(weighted_accuracy_true / max(weight_sum, 1e-12))
    record["weighted_utility_fit"] = float(record["weighted_accuracy_fit"]) if not pd.isna(record["weighted_accuracy_fit"]) else np.nan
    record["weighted_utility_true"] = float(record["weighted_accuracy_true"])
    if pred_delay_values:
        record["avg_delay_pred_sec"] = float(np.mean(np.asarray(pred_delay_values, dtype=float)))
    if pred_excess_values:
        record["avg_excess_delay_pred_sec"] = float(np.mean(np.asarray(pred_excess_values, dtype=float)))
    if pred_ratio_values:
        record["avg_delay_ratio_pred"] = float(np.mean(np.asarray(pred_ratio_values, dtype=float)))
    if anchor_task is not None:
        anchor_tid = int(anchor_task.logical_task_id)
        anchor_snap = latest_snapshots[anchor_tid]
        anchor_alloc = allocations[anchor_tid]
        anchor_eta = np.asarray(anchor_alloc["eta"], dtype=float)
        anchor_target_rate_hz = float(anchor_task.target_rate_hz)
        anchor_target_time_sec = (1.0 / anchor_target_rate_hz) if anchor_target_rate_hz > 0.0 else np.nan
        record["target_rate_hz"] = anchor_target_rate_hz
        record["target_time_sec"] = anchor_target_time_sec
        record["accuracy_fit"] = float(record.get("accuracy_fit_task{}".format(anchor_tid), np.nan))
        record["accuracy_true"] = float(anchor_snap.running_accuracy)
        record["delay_actual_sec"] = float(anchor_snap.total_delay_sec)
        record["delay_pred_sec"] = float(record.get("delay_pred_sec_task{}".format(anchor_tid), np.nan))
        record["delay_ratio_actual"] = float(anchor_snap.total_delay_sec) * anchor_target_rate_hz
        anchor_pred_delay = float(record.get("delay_pred_sec", np.nan))
        record["delay_ratio_pred"] = float(anchor_pred_delay) * anchor_target_rate_hz
        record["lambda_value"] = float(getattr(policy, "lambda_k", {}).get(anchor_tid, 0.0))
        record["requested_eta"] = ",".join(_format_float4(value) for value in anchor_eta)
        record["executed_eta"] = ",".join(_format_float4(value) for value in anchor_eta)
        for idx, value in enumerate(anchor_eta):
            record["requested_eta_{}".format(idx)] = float(value)
            record["executed_eta_{}".format(idx)] = float(value)
            record["eta_{}".format(idx)] = float(value)
            record["executed_eta_{}".format(idx)] = float(value)
    return record

def _retarget_control_records(rows, tasks, algorithm_name=None, algorithm_type=None, anchor_task_id=None):
    patched = []
    task_by_id = {int(task.logical_task_id): task for task in tasks}
    for row in rows:
        item = dict(row)
        if algorithm_name is not None:
            item["algorithm"] = str(algorithm_name)
            item["policy"] = str(algorithm_name)
        if algorithm_type is not None:
            item["algorithm_type"] = str(algorithm_type)
            item["policy_key"] = str(algorithm_type)
        delay_ratio_values = []
        excess_delay_values = []
        delay_ratio_pred_values = []
        excess_delay_pred_values = []
        for task in tasks:
            tid = int(task.logical_task_id)
            target_rate = float(task.target_rate_hz)
            delay_sec = float(item.get("delay_actual_sec_task{}".format(tid), 0.0))
            delay_pred_sec = float(item.get("delay_pred_sec_task{}".format(tid), np.nan))
            item["target_rate_hz_task{}".format(tid)] = target_rate
            item["excess_delay_sec_task{}".format(tid)] = (
                delay_sec - (1.0 / target_rate) if target_rate > 0.0 else np.nan
            )
            item["excess_delay_pred_sec_task{}".format(tid)] = (
                delay_pred_sec - (1.0 / target_rate) if target_rate > 0.0 else np.nan
            )
            item["delay_ratio_actual_task{}".format(tid)] = delay_sec * target_rate
            item["delay_ratio_pred_task{}".format(tid)] = delay_pred_sec * target_rate if target_rate > 0.0 else np.nan
            excess_delay_values.append(item["excess_delay_sec_task{}".format(tid)])
            delay_ratio_values.append(delay_sec * target_rate)
            excess_delay_pred_values.append(item["excess_delay_pred_sec_task{}".format(tid)])
            delay_ratio_pred_values.append(item["delay_ratio_pred_task{}".format(tid)])
        if excess_delay_values:
            item["avg_excess_delay_actual_sec"] = float(np.mean(np.asarray(excess_delay_values, dtype=float)))
        if delay_ratio_values:
            item["avg_delay_ratio_actual"] = float(np.mean(np.asarray(delay_ratio_values, dtype=float)))
        if excess_delay_pred_values:
            item["avg_excess_delay_pred_sec"] = float(np.mean(np.asarray(excess_delay_pred_values, dtype=float)))
        if delay_ratio_pred_values:
            item["avg_delay_ratio_pred"] = float(np.mean(np.asarray(delay_ratio_pred_values, dtype=float)))
        if "weighted_accuracy_fit" in item:
            item["weighted_utility_fit"] = float(item["weighted_accuracy_fit"])
        if "weighted_accuracy_true" in item:
            item["weighted_utility_true"] = float(item["weighted_accuracy_true"])
        anchor_tid = None
        if anchor_task_id is not None:
            anchor_tid = int(anchor_task_id)
        elif "anchor_task_id" in item:
            try:
                anchor_tid = int(item["anchor_task_id"])
            except Exception:
                anchor_tid = None
        if anchor_tid is None and tasks:
            anchor_tid = int(tasks[0].logical_task_id)
        anchor_task = task_by_id.get(int(anchor_tid)) if anchor_tid is not None else None
        if anchor_task is not None:
            target_rate = float(anchor_task.target_rate_hz)
            delay_actual = float(item.get("delay_actual_sec_task{}".format(anchor_tid), np.nan))
            delay_pred = float(item.get("delay_pred_sec_task{}".format(anchor_tid), np.nan))
            item["anchor_task_id"] = int(anchor_tid)
            item["target_rate_hz"] = target_rate
            item["target_time_sec"] = (1.0 / target_rate) if target_rate > 0.0 else np.nan
            item["accuracy_true"] = float(item.get("accuracy_true_task{}".format(anchor_tid), np.nan))
            item["accuracy_fit"] = float(item.get("accuracy_fit_task{}".format(anchor_tid), np.nan))
            item["delay_actual_sec"] = delay_actual
            item["delay_pred_sec"] = delay_pred
            item["delay_ratio_actual"] = delay_actual * target_rate if target_rate > 0.0 else np.nan
            item["delay_ratio_pred"] = delay_pred * target_rate if target_rate > 0.0 else np.nan
            item["lambda_value"] = float(item.get("lambda_task{}".format(anchor_tid), np.nan))
            eta_values = []
            for link_idx in range(NUM_TRANSFER_POINTS):
                eta_value = float(item.get("eta_task{}_{}".format(anchor_tid, link_idx), np.nan))
                item["requested_eta_{}".format(link_idx)] = eta_value
                item["executed_eta_{}".format(link_idx)] = eta_value
                item["eta_{}".format(link_idx)] = eta_value
                eta_values.append(eta_value)
            item["requested_eta"] = ",".join(_format_float4(value) for value in eta_values)
            item["executed_eta"] = ",".join(_format_float4(value) for value in eta_values)
        patched.append(item)
    return patched

def _process_completion(
    result_payload,
    pending,
    inflight_by_task,
    aggregate_stats,
    latest_snapshots,
    shared_bandwidth,
):
    request_id = result_payload["request_id"]
    meta = _finalize_request(request_id, result_payload, pending)
    if meta is None:
        return None
    logical_task_id = int(meta["logical_task_id"])
    inflight_by_task[logical_task_id] = max(0, inflight_by_task[logical_task_id] - 1)
    if meta["task_family"] == "resnet":
        predictions = result_payload["predictions"]
        labels = meta["labels"]
        pred_tensor = torch.as_tensor(predictions, dtype=torch.long)
        labels_cpu = labels.detach().cpu().to(torch.long)
        correct_count = int((pred_tensor == labels_cpu).to(torch.long).sum().item())
        sample_total = int(labels_cpu.numel())
        sample_correct = float(correct_count) / float(sample_total) if sample_total > 0 else 0.0
    else:
        correct_count = int(int(int(result_payload["predicted_label"]) == int(meta["label"])))
        sample_total = 1
        sample_correct = float(correct_count)
    aggregate_stats[logical_task_id]["seen"] += int(sample_total)
    aggregate_stats[logical_task_id]["correct"] += float(correct_count)
    aggregate_stats[logical_task_id]["last_correct"] = sample_correct
    aggregate_stats[logical_task_id]["completions"] += 1
    snapshot = _build_snapshot_from_completion(meta, aggregate_stats)
    latest_snapshots[logical_task_id] = snapshot
    shared_bandwidth.observe(logical_task_id, snapshot.tp_bandwidth_bps)
    return {
        "logical_task_id": logical_task_id,
        "snapshot": snapshot,
        "meta": meta,
        "correct_count": int(correct_count),
        "sample_total": int(sample_total),
        "sample_correct": float(sample_correct),
    }

def _run_stage_round_robin(
    transport,
    sender,
    tasks,
    runtime_map,
    items_by_task,
    initial_allocations,
    shared_bandwidth,
    max_inflight_per_task,
    device,
    policy=None,
    anchor_task_id=None,
    output_dir=None,
    plot_context=None,
    algorithm_name=None,
    algorithm_type=None,
    channel_estimator=None,
    terminate_on_task_id=None,
    loop_non_terminating_tasks=False,
    record_every_completion=False,
    freeze_bandwidth_during_stage=False,
    fixed_stage_total_bw_bps=None,
    tx_limit_mbps_for_warning=None,
    dynamic_timeslot_size=None,
    max_dynamic_timeslot_count=None,
    max_total_batches=None,
    target_completion_count_override=None,
):
    tasks_by_id = {int(task.logical_task_id): task for task in tasks}
    task_order = [int(task.logical_task_id) for task in tasks]
    rr_index = 0
    pending = {}
    dispatch_futures = {}
    inflight_by_task = {tid: 0 for tid in task_order}
    aggregate_stats = {
        tid: {
            "seen": 0,
            "correct": 0.0,
            "last_correct": 0.0,
            "completions": 0,
            "delay_ratio_sum": 0.0,
            "delay_ratio_count": 0,
        }
        for tid in task_order
    }
    latest_snapshots = {}
    current_allocations = _clone_allocations(initial_allocations)
    if fixed_stage_total_bw_bps is not None:
        total_bw_bps = np.asarray(fixed_stage_total_bw_bps, dtype=float).copy()
    else:
        total_bw_bps = _current_total_bandwidth_bps(channel_estimator, shared_bandwidth)
    _warn_if_per_link_bandwidth_low(
        total_bw_bps,
        tx_limit_mbps=tx_limit_mbps_for_warning,
        context_label="{}:stage_start".format(algorithm_name or "warmup"),
    )
    fixed_channel_policy = _policy_uses_fixed_channel(policy=policy, algorithm_type=algorithm_type) or bool(
        freeze_bandwidth_during_stage
    ) or bool(
        fixed_stage_total_bw_bps is not None
    )
    window_bandwidth_control = bool(
        policy is not None
        and anchor_task_id is not None
        and dynamic_timeslot_size is not None
        and not fixed_channel_policy
    )
    fixed_total_bw_bps = (
        np.asarray(fixed_stage_total_bw_bps, dtype=float).copy()
        if fixed_stage_total_bw_bps is not None
        else np.asarray(total_bw_bps, dtype=float).copy()
    )
    control_records = []
    completion_rows = []
    control_step = 0
    total_tp_stats_history = []
    terminated_task_id = None
    termination_reason = None
    completion_window_queues = {
        int(tid): deque()
        for tid in task_order
    }
    solver_job = None
    window_summaries = []
    window_summary_by_id = {}
    next_window_id = 1
    next_policy_version = 1
    anchor_dispatch_count = 0
    dispatch_horizon_closed = False
    base_items_by_task = {int(tid): list(items_by_task[tid]) for tid in task_order}
    effective_max_dynamic_timeslot_count = (
        None if target_completion_count_override is not None else max_dynamic_timeslot_count
    )
    effective_max_total_batches = (
        None if target_completion_count_override is not None else max_total_batches
    )
    target_completion_count = (
        None if target_completion_count_override is None else max(1, int(target_completion_count_override))
    )
    if terminate_on_task_id is not None:
        if target_completion_count is not None:
            pass
        elif effective_max_total_batches is not None:
            target_completion_count = max(1, int(effective_max_total_batches))
        elif effective_max_dynamic_timeslot_count is None:
            target_completion_count = int(len(base_items_by_task[int(terminate_on_task_id)]))
    current_control_state = {
        "policy_version": 0,
        "window_id": 0,
        "allocations": _clone_allocations(current_allocations),
        "lambda_k": (
            {int(k): float(v) for k, v in getattr(policy, "lambda_k", {}).items()}
            if hasattr(policy, "lambda_k") else {}
        ),
        "control_solve_sec": np.nan,
        "total_bw_bps": np.asarray(total_bw_bps, dtype=float).copy(),
        "predicted_accuracy_by_task": {},
        "predicted_delay_by_task": {},
    }

    logging.info(
        "[A][%s] Stage start: tasks=%s pending_items=%s max_inflight_per_task=%s total_bw_bps=%s terminate_on_task_id=%s target_completion_count=%s loop_non_terminating_tasks=%s",
        algorithm_name or "warmup",
        task_order,
        {int(tid): len(items_by_task[tid]) for tid in task_order},
        int(max_inflight_per_task),
        [float(x) for x in np.asarray(total_bw_bps, dtype=float)],
        (None if terminate_on_task_id is None else int(terminate_on_task_id)),
        target_completion_count,
        bool(loop_non_terminating_tasks),
    )
    if fixed_channel_policy:
        logging.info(
            "[A][%s] Fixed-channel baseline: freezing total_bw_bps=%s for entire stage",
            algorithm_name or "warmup",
            [float(x) for x in np.asarray(fixed_total_bw_bps, dtype=float)],
        )
    if policy is not None:
        logging.info(
            "[A][%s] Initial allocations: %s",
            algorithm_name,
            {
                int(tid): {
                    "eta": [float(x) for x in np.asarray(initial_allocations[tid]["eta"], dtype=float)],
                    "s_comm": [float(x) for x in np.asarray(initial_allocations[tid]["s_comm"], dtype=float)],
                    "s_comp": [float(x) for x in np.asarray(initial_allocations[tid]["s_comp"], dtype=float)],
                }
                for tid in task_order
            },
        )

    def _anchor_buffer_count():
        if anchor_task_id is None:
            return int(sum(len(queue_items) for queue_items in completion_window_queues.values()))
        return int(len(completion_window_queues.get(int(anchor_task_id), ())))

    def _flatten_completion_window_queues():
        snapshot_records = []
        for tid in task_order:
            snapshot_records.extend(list(completion_window_queues[int(tid)]))
        return snapshot_records

    def _should_stop_new_control_updates():
        if effective_max_dynamic_timeslot_count is None:
            return False
        return int(len(window_summaries)) >= int(effective_max_dynamic_timeslot_count)

    def _dispatch_allowed():
        if terminate_on_task_id is None:
            return True
        if target_completion_count_override is not None:
            return int(anchor_dispatch_count) < int(target_completion_count)
        if effective_max_total_batches is not None and int(anchor_dispatch_count) >= int(effective_max_total_batches):
            return False
        if effective_max_dynamic_timeslot_count is None:
            return int(anchor_dispatch_count) < int(len(base_items_by_task[int(terminate_on_task_id)]))
        return int(len(window_summaries)) < int(effective_max_dynamic_timeslot_count)

    def _close_dispatch_horizon(reason):
        nonlocal dispatch_horizon_closed, terminated_task_id, termination_reason
        if dispatch_horizon_closed:
            return
        dispatch_horizon_closed = True
        if termination_reason is None:
            termination_reason = str(reason)
        for tid in task_order:
            items_by_task[tid].clear()
        if terminated_task_id is None and terminate_on_task_id is not None:
            terminated_task_id = int(terminate_on_task_id)
        logging.info(
            "[A][%s] Dispatch horizon closed: reason=%s anchor_task=%s anchor_dispatched=%s windows=%s max_total_batches=%s max_dynamic_timeslot_count=%s",
            algorithm_name or "warmup",
            str(reason),
            (None if terminate_on_task_id is None else int(terminate_on_task_id)),
            int(anchor_dispatch_count),
            int(len(window_summaries)),
            (None if effective_max_total_batches is None else int(effective_max_total_batches)),
            (None if effective_max_dynamic_timeslot_count is None else int(effective_max_dynamic_timeslot_count)),
        )

    def _try_start_control_solver(solver_executor):
        nonlocal solver_job, completion_window_queues, next_window_id
        if policy is None or anchor_task_id is None or dynamic_timeslot_size is None:
            return
        if solver_job is not None:
            return
        if _should_stop_new_control_updates():
            return
        if len(latest_snapshots) < len(tasks):
            return
        if _anchor_buffer_count() < max(1, int(dynamic_timeslot_size)):
            return
        snapshot_records = _flatten_completion_window_queues()
        snapshot_records_by_task = {
            int(tid): deque(list(completion_window_queues[int(tid)]))
            for tid in task_order
        }
        completion_window_queues = {
            int(tid): deque()
            for tid in task_order
        }
        averaged_snapshots = _aggregate_control_window_snapshots(snapshot_records, tasks, latest_snapshots)
        if len(averaged_snapshots) < len(tasks):
            completion_window_queues = {
                int(tid): deque(list(snapshot_records_by_task[int(tid)]) + list(completion_window_queues[int(tid)]))
                for tid in task_order
            }
            return
        window_total_bw_bps = _aggregate_control_window_total_bandwidth_bps(
            snapshot_records,
            tasks,
            latest_snapshots,
        )
        window_id = int(next_window_id)
        next_window_id += 1
        anchor_completion_count = int(
            sum(1 for item in snapshot_records if int(item["logical_task_id"]) == int(anchor_task_id))
        )
        summary_row = {
            "window_id": int(window_id),
            "dynamic_timeslot_size": int(dynamic_timeslot_size),
            "actual_window_size": int(len(snapshot_records)),
            "anchor_task_id": int(anchor_task_id),
            "anchor_completion_count": int(anchor_completion_count),
            "solver_elapsed_sec": np.nan,
        }
        anchor_snap = averaged_snapshots.get(int(anchor_task_id))
        anchor_task = tasks_by_id.get(int(anchor_task_id))
        if anchor_snap is not None and anchor_task is not None:
            anchor_records = [
                item for item in snapshot_records if int(item["logical_task_id"]) == int(anchor_task_id)
            ]
            anchor_total_samples = int(sum(int(item.get("sample_total", 0)) for item in anchor_records))
            anchor_total_correct = int(sum(int(item.get("correct_count", 0)) for item in anchor_records))
            summary_row["sample_count"] = int(anchor_completion_count)
            summary_row["batch_count"] = int(anchor_completion_count)
            summary_row["window_total_samples"] = int(anchor_total_samples)
            summary_row["window_correct_samples"] = int(anchor_total_correct)
            summary_row["avg_accuracy"] = (
                float(anchor_total_correct) / float(anchor_total_samples)
                if anchor_total_samples > 0 else np.nan
            )
            summary_row["avg_delay_sec"] = float(anchor_snap.total_delay_sec)
            summary_row["target_time_sec"] = (
                (1.0 / float(anchor_task.target_rate_hz)) if float(anchor_task.target_rate_hz) > 0.0 else np.nan
            )
        for task in tasks:
            tid = int(task.logical_task_id)
            records_for_task = [item for item in snapshot_records if int(item["logical_task_id"]) == tid]
            snap = averaged_snapshots[tid]
            task_total_samples = int(sum(int(item.get("sample_total", 0)) for item in records_for_task))
            task_total_correct = int(sum(int(item.get("correct_count", 0)) for item in records_for_task))
            summary_row["completion_count_task{}".format(tid)] = int(len(records_for_task))
            summary_row["sample_count_task{}".format(tid)] = int(len(records_for_task))
            summary_row["batch_count_task{}".format(tid)] = int(len(records_for_task))
            summary_row["window_total_samples_task{}".format(tid)] = int(task_total_samples)
            summary_row["window_correct_samples_task{}".format(tid)] = int(task_total_correct)
            summary_row["avg_accuracy_task{}".format(tid)] = (
                float(task_total_correct) / float(task_total_samples)
                if task_total_samples > 0 else np.nan
            )
            summary_row["avg_delay_sec_task{}".format(tid)] = float(snap.total_delay_sec)
            summary_row["avg_end_to_end_delay_sec_task{}".format(tid)] = float(snap.end_to_end_delay_sec)
            for link_idx, value in enumerate(np.asarray(snap.a_ref, dtype=float)):
                summary_row["avg_a_task{}_{}".format(tid, link_idx)] = float(value)
            for link_idx, value in enumerate(np.asarray(snap.tp_bandwidth_bps, dtype=float)):
                summary_row["avg_c_task{}_{}".format(tid, link_idx)] = float(value)
            for node_idx, value in enumerate(np.asarray(snap.tau_list, dtype=float)):
                summary_row["avg_tau_task{}_{}".format(tid, node_idx)] = float(value)
        window_summaries.append(summary_row)
        window_summary_by_id[int(window_id)] = summary_row
        if fixed_channel_policy:
            bw_snapshot = np.asarray(fixed_total_bw_bps, dtype=float).copy()
        elif channel_estimator is not None:
            channel_estimator.observe_task(_tp_stats_from_total_bandwidth_bps(window_total_bw_bps))
            bw_snapshot = _predict_total_bandwidth_bps_from_estimator(channel_estimator, window_total_bw_bps)
        else:
            bw_snapshot = np.asarray(window_total_bw_bps, dtype=float).copy()
        summary_row["window_total_bw_link0_mbps"] = float(bw_snapshot[0]) / 1e6 if bw_snapshot.size > 0 else np.nan
        summary_row["window_total_bw_link1_mbps"] = float(bw_snapshot[1]) / 1e6 if bw_snapshot.size > 1 else np.nan
        summary_row["window_total_bw_link2_mbps"] = float(bw_snapshot[2]) / 1e6 if bw_snapshot.size > 2 else np.nan
        solver_job = solver_executor.submit(
            _solve_multi_policy_snapshot,
            policy,
            averaged_snapshots,
            bw_snapshot,
            tasks_by_id,
            int(window_id),
        )

    def _maybe_collect_control_solver_result():
        nonlocal solver_job, current_allocations, current_control_state, control_step, next_policy_version
        if solver_job is None or not solver_job.done():
            return
        solved = solver_job.result()
        current_allocations = _clone_allocations(solved["allocations"])
        predicted_accuracy_by_task = dict(solved.get("predicted_accuracy_by_task", {}))
        predicted_delay_by_task = dict(solved.get("predicted_delay_by_task", {}))
        if hasattr(policy, "lambda_k") and isinstance(solved.get("lambda_k"), dict):
            policy.lambda_k = {int(k): float(v) for k, v in solved["lambda_k"].items()}
        current_control_state = {
            "policy_version": int(next_policy_version),
            "window_id": int(solved["window_id"]),
            "allocations": _clone_allocations(current_allocations),
            "lambda_k": {int(k): float(v) for k, v in solved.get("lambda_k", {}).items()},
            "control_solve_sec": float(solved["control_solve_sec"]),
            "total_bw_bps": np.asarray(solved["total_bw_bps"], dtype=float).copy(),
            "predicted_accuracy_by_task": dict(predicted_accuracy_by_task),
            "predicted_delay_by_task": dict(predicted_delay_by_task),
        }
        summary_row = window_summary_by_id.get(int(solved["window_id"]))
        if summary_row is not None:
            summary_row["solver_elapsed_sec"] = float(solved["control_solve_sec"])
            summary_row["policy_version"] = int(next_policy_version)
            summary_row["lambda_input"] = float(solved.get("lambda_k", {}).get(int(anchor_task_id), 0.0))
            summary_row["predicted_accuracy"] = float(predicted_accuracy_by_task.get(int(anchor_task_id), np.nan))
            summary_row["predicted_delay_sec"] = float(predicted_delay_by_task.get(int(anchor_task_id), np.nan))
            for task in tasks:
                tid = int(task.logical_task_id)
                alloc = current_allocations[tid]
                for link_idx, value in enumerate(np.asarray(alloc["eta"], dtype=float)):
                    summary_row["eta_task{}_{}".format(tid, link_idx)] = float(value)
                    if int(tid) == int(anchor_task_id):
                        summary_row["requested_eta_{}".format(link_idx)] = float(value)
                for link_idx, value in enumerate(np.asarray(alloc["s_comm"], dtype=float)):
                    summary_row["s_comm_task{}_{}".format(tid, link_idx)] = float(value)
                for node_idx, value in enumerate(np.asarray(alloc["s_comp"], dtype=float)):
                    summary_row["s_comp_task{}_{}".format(tid, node_idx)] = float(value)
        control_records.append(
            _compute_control_record(
                control_step,
                algorithm_name or str(getattr(policy, "policy_name", "policy")),
                algorithm_type or str(getattr(policy, "policy_key", "policy")),
                policy,
                tasks,
                current_allocations,
                solved["averaged_snapshots"],
                solved["total_bw_bps"],
                aggregate_stats=aggregate_stats,
                control_solve_sec=float(solved["control_solve_sec"]),
                window_id=int(solved["window_id"]),
                dynamic_timeslot_size=(None if dynamic_timeslot_size is None else int(dynamic_timeslot_size)),
                actual_window_size=(
                    None if summary_row is None else int(summary_row.get("actual_window_size", 0))
                ),
                anchor_task_id=anchor_task_id,
                predicted_accuracy_by_task=predicted_accuracy_by_task,
                predicted_delay_by_task=predicted_delay_by_task,
            )
        )
        logging.info(
            "[A][%s] Control update #%s window=%s solve_sec=%.4f policy_v=%s lambda=%s",
            algorithm_name or "policy",
            int(control_step),
            int(solved["window_id"]),
            float(solved["control_solve_sec"]),
            int(next_policy_version),
            {int(k): float(v) for k, v in getattr(policy, "lambda_k", {}).items()},
        )
        control_step += 1
        next_policy_version += 1
        solver_job = None

    dispatch_workers = max(len(task_order) * int(max_inflight_per_task), 1)
    with ThreadPoolExecutor(max_workers=dispatch_workers) as dispatch_executor, ThreadPoolExecutor(max_workers=1) as solver_executor:
        while any(items_by_task[tid] for tid in task_order) or pending or dispatch_futures:
            _maybe_collect_control_solver_result()
            _try_start_control_solver(solver_executor)
            if _should_stop_new_control_updates():
                _close_dispatch_horizon("max_dynamic_timeslot_count_reached")
            if (not dispatch_horizon_closed) and (not _dispatch_allowed()):
                _close_dispatch_horizon("anchor_dispatch_limit_reached")
            for future, future_meta in list(dispatch_futures.items()):
                if not future.done():
                    continue
                dispatch_futures.pop(future, None)
                logical_task_id = int(future_meta["logical_task_id"])
                try:
                    meta = future.result()
                except Exception:
                    inflight_by_task[logical_task_id] = max(0, inflight_by_task[logical_task_id] - 1)
                    logging.exception(
                        "[A][%s] Dispatch worker failed request=%s task=%s batch=%s",
                        algorithm_name or "warmup",
                        future_meta["request_id"],
                        logical_task_id,
                        int(future_meta["batch_idx"]),
                    )
                    continue
                request_id = str(meta["request_id"])
                meta["policy_version"] = int(future_meta["policy_version"])
                meta["window_id"] = int(future_meta["window_id"])
                meta["control_solve_sec"] = float(future_meta["control_solve_sec"])
                meta["total_bw_bps"] = [float(x) for x in np.asarray(future_meta["total_bw_bps"], dtype=float)]
                meta["s_comm"] = [float(x) for x in np.asarray(future_meta["s_comm"], dtype=float)]
                meta["s_comp"] = [float(x) for x in np.asarray(future_meta["s_comp"], dtype=float)]
                meta["selected_accuracy_est"] = float(future_meta["selected_accuracy_est"])
                meta["selected_delay_est"] = float(future_meta["selected_delay_est"])
                meta["lambda_value"] = float(future_meta["lambda_value"])
                pending[request_id] = meta
                logging.info(
                    "[A][%s] Dispatch task=%s batch=%s eta=%s s_comm=%s total_bw_bps=%s policy_v=%s window=%s",
                    algorithm_name or "warmup",
                    logical_task_id,
                    int(future_meta["batch_idx"]),
                    [float(x) for x in np.asarray(future_meta["eta"], dtype=float)],
                    [float(x) for x in np.asarray(future_meta["s_comm"], dtype=float)],
                    [float(x) for x in np.asarray(future_meta["total_bw_bps"], dtype=float)],
                    int(future_meta["policy_version"]),
                    int(future_meta["window_id"]),
                )

            while terminated_task_id is None:
                if dispatch_horizon_closed or _should_stop_new_control_updates() or (not _dispatch_allowed()):
                    break
                candidate_tid = None
                for _ in range(len(task_order)):
                    tid = task_order[rr_index % len(task_order)]
                    rr_index += 1
                    if (
                        terminate_on_task_id is not None
                        and int(tid) != int(terminate_on_task_id)
                        and not items_by_task[int(terminate_on_task_id)]
                        and inflight_by_task[int(terminate_on_task_id)] > 0
                    ):
                        continue
                    if (
                        terminate_on_task_id is not None
                        and not items_by_task[tid]
                        and base_items_by_task.get(int(tid))
                    ):
                        should_refill = False
                        if int(tid) == int(terminate_on_task_id):
                            should_refill = bool(
                                (
                                    target_completion_count_override is not None
                                    or effective_max_total_batches is not None
                                    or effective_max_dynamic_timeslot_count is not None
                                )
                                and _dispatch_allowed()
                            )
                        else:
                            should_refill = bool(loop_non_terminating_tasks and _dispatch_allowed())
                        if should_refill:
                            items_by_task[tid] = deque(list(base_items_by_task[int(tid)]))
                    if items_by_task[tid] and inflight_by_task[tid] < int(max_inflight_per_task):
                        candidate_tid = tid
                        break
                if candidate_tid is None:
                    break
                batch = items_by_task[candidate_tid].popleft()
                effective_total_bw_bps = (
                    np.asarray(fixed_total_bw_bps, dtype=float).copy()
                    if fixed_channel_policy
                    else np.asarray(current_control_state["total_bw_bps"], dtype=float).copy()
                )
                alloc = copy.deepcopy(current_control_state["allocations"][candidate_tid])
                alloc["total_bw_bps"] = effective_total_bw_bps.copy()
                request_id = str(uuid.uuid4())
                inflight_by_task[candidate_tid] += 1
                future_meta = {
                    "request_id": request_id,
                    "logical_task_id": int(candidate_tid),
                    "batch_idx": int(batch.get("batch_idx", -1)),
                    "eta": np.asarray(alloc["eta"], dtype=float).copy(),
                    "s_comm": np.asarray(alloc["s_comm"], dtype=float).copy(),
                    "s_comp": np.asarray(alloc["s_comp"], dtype=float).copy(),
                    "total_bw_bps": effective_total_bw_bps.copy(),
                    "policy_version": int(current_control_state["policy_version"]),
                    "window_id": int(current_control_state["window_id"]),
                    "control_solve_sec": float(current_control_state["control_solve_sec"]),
                    "selected_accuracy_est": float(current_control_state.get("predicted_accuracy_by_task", {}).get(int(candidate_tid), np.nan)),
                    "selected_delay_est": float(current_control_state.get("predicted_delay_by_task", {}).get(int(candidate_tid), np.nan)),
                    "lambda_value": float(current_control_state.get("lambda_k", {}).get(int(candidate_tid), 0.0)),
                }
                if terminate_on_task_id is not None and int(candidate_tid) == int(terminate_on_task_id):
                    anchor_dispatch_count += 1
                dispatch_futures[
                    dispatch_executor.submit(
                        _dispatch_async_request,
                        sender,
                        runtime_map,
                        tasks_by_id,
                        batch,
                        candidate_tid,
                        alloc,
                        device,
                        request_id,
                    )
                ] = future_meta

            if terminated_task_id is not None and not pending and not dispatch_futures:
                break
            if not pending and not dispatch_futures and not any(items_by_task[tid] for tid in task_order):
                break
            if not pending:
                time.sleep(0.01)
                continue
            try:
                envelope = transport.receive(msg_type=MessageType.FINAL_RESULT, source="D", timeout=0.25)
            except TimeoutError:
                continue
            completion = _process_completion(
                envelope["payload"],
                pending,
                inflight_by_task,
                aggregate_stats,
                latest_snapshots,
                shared_bandwidth,
            )
            if completion is None:
                logging.warning("[A][%s] Received FINAL_RESULT for unknown request_id=%s", algorithm_name or "warmup", envelope["payload"].get("request_id"))
                continue
            measured_channel_mbps = [
                float(value) / 1e6 for value in np.asarray(completion["snapshot"].tp_bandwidth_bps, dtype=float)
            ]
            alloc_total_bw_bps = np.asarray(completion["meta"].get("total_bw_bps", []), dtype=float)
            alloc_s_comm = np.asarray(completion["meta"].get("s_comm", []), dtype=float)
            if alloc_total_bw_bps.size > 0 and alloc_s_comm.size > 0:
                alloc_link_mbps = [
                    float(value) / 1e6
                    for value in np.asarray(alloc_total_bw_bps * alloc_s_comm, dtype=float)
                ]
            else:
                alloc_link_mbps = []
            logging.info(
                "[A][%s] Completion task=%s delay=%.4fs service=%.4fs comm=%.4fs channel_mbps=%s alloc_link_mbps=%s policy_v=%s window=%s",
                algorithm_name or "warmup",
                int(completion["logical_task_id"]),
                float(completion["snapshot"].total_delay_sec),
                float(completion["snapshot"].service_delay_sec),
                float(completion["snapshot"].communication_delay_sec),
                [round(float(x), 4) for x in measured_channel_mbps],
                [round(float(x), 4) for x in alloc_link_mbps],
                int(completion["meta"].get("policy_version", 0)),
                int(completion["meta"].get("window_id", 0)),
            )
            task_id = int(completion["logical_task_id"])
            task = tasks_by_id[task_id]
            delay_ratio_actual = (
                float(completion["snapshot"].total_delay_sec) * float(task.target_rate_hz)
                if float(task.target_rate_hz) > 0.0
                else np.nan
            )
            if not pd.isna(delay_ratio_actual):
                aggregate_stats[task_id]["delay_ratio_sum"] += float(delay_ratio_actual)
                aggregate_stats[task_id]["delay_ratio_count"] += 1
            running_acc = (
                float(aggregate_stats[task_id]["correct"]) / float(aggregate_stats[task_id]["seen"])
                if int(aggregate_stats[task_id]["seen"]) > 0
                else np.nan
            )
            running_delay_ratio = (
                float(aggregate_stats[task_id]["delay_ratio_sum"]) / float(aggregate_stats[task_id]["delay_ratio_count"])
                if int(aggregate_stats[task_id]["delay_ratio_count"]) > 0
                else np.nan
            )
            logging.info(
                "[A][%s] Running summary task=%s family=%s acc=%.4f delay_ratio=%.4f",
                algorithm_name or "warmup",
                task_id,
                str(completion["meta"].get("task_family", "")),
                float(running_acc) if not pd.isna(running_acc) else np.nan,
                float(running_delay_ratio) if not pd.isna(running_delay_ratio) else np.nan,
            )
            if (
                policy is not None
                and tx_limit_mbps_for_warning is not None
                and len(latest_snapshots) == len(tasks)
            ):
                observed_total_bw_bps = np.asarray(shared_bandwidth.total_bandwidth_bps(), dtype=float)
                breach_threshold_bps = float(tx_limit_mbps_for_warning) * 1.5 * 1e6
                if observed_total_bw_bps.size > 0 and np.any(observed_total_bw_bps > breach_threshold_bps):
                    breached_links = [
                        int(idx)
                        for idx, value in enumerate(observed_total_bw_bps)
                        if float(value) > breach_threshold_bps
                    ]
                    logging.warning(
                        "[A][%s] Channel collapse detected after completion task=%s: total_link_mbps=%s threshold_mbps=%.4f breached_links=%s; stopping current policy",
                        algorithm_name or "policy",
                        int(completion["logical_task_id"]),
                        [round(float(x) / 1e6, 4) for x in observed_total_bw_bps],
                        float(breach_threshold_bps) / 1e6,
                        breached_links,
                    )
                    if terminated_task_id is None and terminate_on_task_id is None:
                        terminated_task_id = int(completion["logical_task_id"])
                    _close_dispatch_horizon("channel_collapse_over_1p5x_limit")
            completion_rows.append(
                {
                    "logical_task_id": int(completion["logical_task_id"]),
                    "algorithm": str(algorithm_name or ""),
                    "policy": str(algorithm_name or ""),
                    "policy_key": (algorithm_type or str(getattr(policy, "policy_key", ""))),
                    "mu": (
                        float(getattr(policy, "mu", np.nan))
                        if policy is not None and getattr(policy, "mu", None) is not None
                        else np.nan
                    ),
                    "epsilon": (
                        float(getattr(policy, "epsilon", np.nan))
                        if policy is not None and getattr(policy, "epsilon", None) is not None
                        else np.nan
                    ),
                    "batch_idx": int(completion["meta"].get("batch_idx", -1)),
                    "delay_sec": float(completion["snapshot"].total_delay_sec),
                    "end_to_end_delay_sec": float(completion["snapshot"].end_to_end_delay_sec),
                    "service_delay_sec": float(completion["snapshot"].service_delay_sec),
                    "communication_delay_sec": float(completion["snapshot"].communication_delay_sec),
                    "running_accuracy": float(completion["snapshot"].running_accuracy),
                    "sample_correct": float(completion["sample_correct"]),
                    "correct_count": int(completion["correct_count"]),
                    "sample_total": int(completion["sample_total"]),
                    "policy_version": int(completion["meta"].get("policy_version", 0)),
                    "window_id": int(completion["meta"].get("window_id", 0)),
                    "selected_accuracy_est": float(completion["meta"].get("selected_accuracy_est", np.nan)),
                    "selected_delay_est": float(completion["meta"].get("selected_delay_est", np.nan)),
                    "lambda_value": float(completion["meta"].get("lambda_value", np.nan)),
                    "compressor_name": completion["meta"].get("compressor_name"),
                    "requested_eta": list(completion["meta"].get("requested_eta", [])),
                    "executed_eta": list(completion["meta"].get("executed_eta", [])),
                }
            )
            if policy is not None and anchor_task_id is not None and dynamic_timeslot_size is not None:
                completion_window_queues[int(completion["logical_task_id"])].append(
                    {
                        "logical_task_id": int(completion["logical_task_id"]),
                        "snapshot": completion["snapshot"],
                        "correct_count": int(completion["correct_count"]),
                        "sample_total": int(completion["sample_total"]),
                        "policy_version": int(completion["meta"].get("policy_version", 0)),
                    }
                )
            if terminated_task_id is None:
                if (
                    terminate_on_task_id is not None
                    and target_completion_count is not None
                    and int(aggregate_stats[int(terminate_on_task_id)]["completions"]) >= int(target_completion_count)
                    and inflight_by_task[int(terminate_on_task_id)] == 0
                ):
                    terminated_task_id = int(terminate_on_task_id)
                    logging.info(
                        "[A][%s] Termination triggered: anchor/slower task=%s reached target completion count=%s; draining remaining in-flight requests before ending stage",
                        algorithm_name or "warmup",
                        int(terminated_task_id),
                        int(target_completion_count),
                    )
                else:
                    for tid in task_order:
                        if (not items_by_task[tid]) and inflight_by_task[tid] == 0 and not loop_non_terminating_tasks:
                            terminated_task_id = int(tid)
                            logging.info(
                                "[A][%s] Termination triggered: task=%s has completed all items; draining remaining in-flight requests before ending stage",
                                algorithm_name or "warmup",
                                int(terminated_task_id),
                            )
                            break
            if len(latest_snapshots) == len(tasks):
                total_tp_stats = shared_bandwidth.total_tp_stats()
                total_tp_stats_history.append(total_tp_stats)
                if window_bandwidth_control:
                    total_bw_bps = np.asarray(current_control_state["total_bw_bps"], dtype=float).copy()
                elif fixed_channel_policy:
                    total_bw_bps = fixed_total_bw_bps.copy()
                elif channel_estimator is not None:
                    channel_estimator.observe_task(total_tp_stats)
                    total_bw_bps = _current_total_bandwidth_bps(channel_estimator, shared_bandwidth)
                else:
                    total_bw_bps = shared_bandwidth.total_bandwidth_bps()
                if not fixed_channel_policy:
                    _warn_if_per_link_bandwidth_low(
                        total_bw_bps,
                        tx_limit_mbps=tx_limit_mbps_for_warning,
                        context_label="{}:control_step_{}".format(algorithm_name or "warmup", int(control_step)),
                    )
                if policy is not None and record_every_completion:
                    control_records.append(
                        _compute_control_record(
                            control_step,
                            algorithm_name or str(getattr(policy, "policy_name", "policy")),
                            algorithm_type or str(getattr(policy, "policy_key", "policy")),
                            policy,
                            tasks,
                            current_control_state["allocations"],
                            latest_snapshots,
                            current_control_state["total_bw_bps"],
                            aggregate_stats=aggregate_stats,
                            control_solve_sec=np.nan,
                            anchor_task_id=anchor_task_id,
                            predicted_accuracy_by_task=current_control_state.get("predicted_accuracy_by_task", {}),
                            predicted_delay_by_task=current_control_state.get("predicted_delay_by_task", {}),
                        )
                    )
                    control_step += 1
                _maybe_collect_control_solver_result()
                _try_start_control_solver(solver_executor)
        while solver_job is not None or (
            policy is not None
            and anchor_task_id is not None
            and dynamic_timeslot_size is not None
            and _anchor_buffer_count() >= max(1, int(dynamic_timeslot_size))
        ):
            _maybe_collect_control_solver_result()
            _try_start_control_solver(solver_executor)
            if solver_job is None and _anchor_buffer_count() < max(1, int(dynamic_timeslot_size)):
                break
            time.sleep(0.01)
    return {
        "latest_snapshots": latest_snapshots,
        "control_records": control_records,
        "completion_rows": completion_rows,
        "aggregate_stats": aggregate_stats,
        "final_allocations": current_control_state["allocations"],
        "total_tp_stats_history": total_tp_stats_history,
        "terminated_task_id": terminated_task_id,
        "termination_reason": termination_reason,
        "window_summaries": window_summaries,
    }
