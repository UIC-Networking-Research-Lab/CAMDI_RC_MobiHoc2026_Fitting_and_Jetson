"""Warmup, measurements and asynchronous control loops."""

import copy
import logging
import numpy as np
import os
import time
import uuid

from concurrent.futures import ThreadPoolExecutor
from jetson_inference.flan_t5.compression import _build_execution_plan
from jetson_inference.flan_t5.config import (
    ASYNC_MAX_INFLIGHT_TASKS,
    ASYNC_QUEUE_POLL_SEC,
    NUM_PARTITIONS,
    NUM_TRANSFER_POINTS,
    _flatten_dict,
    _format_float4,
    _sanitize_tag,
    _save_rows,
)
from jetson_inference.flan_t5.estimation import _build_seeded_channel_estimator
from jetson_inference.flan_t5.policies import (
    _finite_mean,
    _sanitize_lambda_value,
    compute_pipeline_delay,
)
from jetson_inference.flan_t5.workers import (
    _collect_result_payload,
    _dispatch_async_task,
    _worker_profile,
)


def _prepare_profile_context(transport, partition, batches, device, output_dir, profile_name, warmup_total, channel_model_type, channel_predictor_mode, channel_window_size, decoder_start_token_id, verbalizers):
    warmup_tp_stats = []
    warmup_profiles = []
    warmup_times = []
    a_samples = [[] for _ in range(NUM_TRANSFER_POINTS)]
    tau_samples = [[] for _ in range(NUM_PARTITIONS)]
    tau_list = [0.001] * NUM_PARTITIONS
    a_ref = [1.0] * NUM_TRANSFER_POINTS
    warmup_median_delay = 0.0
    if warmup_total > 0:
        logging.info("[A][%s] Warmup: %d samples with identity baseline", profile_name, warmup_total)
        pending = {}
        dispatched = 0
        completed = 0
        identity_plan = {
            "codec_name": "identity",
            "requested_eta": [1.0] * NUM_TRANSFER_POINTS,
            "execution_feature_k_values": [1.0] * NUM_TRANSFER_POINTS,
            "compression_params_list": [None] * NUM_TRANSFER_POINTS,
            "mapping_info": None,
        }
        while completed < warmup_total:
            while dispatched < warmup_total and len(pending) < ASYNC_MAX_INFLIGHT_TASKS:
                batch = batches[dispatched]
                task_id = "warmup_{}_{}_{}".format(profile_name, dispatched, uuid.uuid4().hex[:8])
                task_state = _dispatch_async_task(
                    transport,
                    partition,
                    batch,
                    task_id,
                    identity_plan,
                    device,
                    decoder_start_token_id,
                    verbalizers,
                )
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
            task_state["worker_stats"].update(payload.get("worker_stats_chain", {}))
            end_to_end_delay = time.perf_counter() - task_state["started_at"]
            stats_a = task_state["worker_stats"].get("A")
            stats_b = task_state["worker_stats"].get("B")
            stats_c = task_state["worker_stats"].get("C")
            profile_b = stats_b["profile"] if stats_b is not None else _worker_profile("B", 0.0, 0.0, 0.0, 0, 0.0)
            profile_c = stats_c["profile"] if stats_c is not None else _worker_profile("C", 0.0, 0.0, 0.0, 0, 0.0)
            tp0_stats = stats_a["tp_stats"] if stats_a is not None else task_state["tp_stats"].get("A", {"bytes": 0, "elapsed": 0.0})
            tp1_stats = stats_b["tp_stats"] if stats_b is not None else {"bytes": 0, "elapsed": 0.0}
            tp2_stats = stats_c["tp_stats"] if stats_c is not None else {"bytes": 0, "elapsed": 0.0}
            tp_stats = [
                (tp0_stats["bytes"], tp0_stats["elapsed"]),
                (tp1_stats["bytes"], tp1_stats["elapsed"]),
                (tp2_stats["bytes"], tp2_stats["elapsed"]),
            ]
            warmup_tp_stats.append(tp_stats)
            for idx, item in enumerate(payload.get("tp_original_bytes", [])):
                if float(item) > 0:
                    a_samples[idx].append(float(item))
            tau_values = [
                task_state["profiles"]["A"]["service_total_sec"],
                profile_b["service_total_sec"],
                profile_c["service_total_sec"],
                payload["profile_d"]["service_total_sec"],
            ]
            c_values = np.asarray(
                [
                    (float(item[0]) / float(item[1])) if float(item[1]) > 0.0 else 0.0
                    for item in tp_stats
                ],
                dtype=float,
            )
            a_values = np.asarray(payload.get("tp_original_bytes", [1.0] * NUM_TRANSFER_POINTS), dtype=float)
            total_delay = float(
                compute_pipeline_delay(
                    np.ones(NUM_TRANSFER_POINTS, dtype=float),
                    a_values,
                    np.asarray(tau_values, dtype=float),
                    c_values,
                )
            )
            for idx, tau_val in enumerate(tau_values):
                if tau_val > 0:
                    tau_samples[idx].append(float(tau_val))
            warmup_times.append(total_delay)
            warmup_profiles.append(
                {
                    "compressor_profile": profile_name,
                    "phase": "warmup",
                    "task_id": task_id,
                    "batch_idx": payload["batch_idx"],
                    "sample_id": payload["sample_id"],
                    "total_delay_sec": total_delay,
                    "end_to_end_delay_sec": float(end_to_end_delay),
                    "tp0_bytes": tp0_stats["bytes"],
                    "tp1_bytes": tp1_stats["bytes"],
                    "tp2_bytes": tp2_stats["bytes"],
                    "tp0_elapsed": tp0_stats["elapsed"],
                    "tp1_elapsed": tp1_stats["elapsed"],
                    "tp2_elapsed": tp2_stats["elapsed"],
                    **_flatten_dict({"profiles": {"A": task_state["profiles"]["A"], "B": profile_b, "C": profile_c, "D": payload["profile_d"]}}),
                }
            )
            completed += 1
        tau_list = [float(np.median(tau_samples[idx])) if tau_samples[idx] else 0.001 for idx in range(NUM_PARTITIONS)]
        a_ref = [float(np.median(a_samples[idx])) if a_samples[idx] else 1.0 for idx in range(NUM_TRANSFER_POINTS)]
        warmup_median_delay = float(np.median(np.asarray(warmup_times))) if warmup_times else 0.0
        _save_rows(os.path.join(output_dir, "task_profiles_warmup_{}.csv".format(_sanitize_tag(profile_name))), warmup_profiles)
        logging.info("[A][%s] Warmup done: median_delay=%.4fs a_ref=%s tau=%s", profile_name, warmup_median_delay, a_ref, tau_list)
    seeded_channel_estimator = _build_seeded_channel_estimator(
        warmup_tp_stats=warmup_tp_stats,
        device=device,
        model_type=channel_model_type,
        update_mode=channel_predictor_mode,
        window_size=channel_window_size,
    )
    return {
        "warmup_tp_stats": warmup_tp_stats,
        "warmup_median_delay": warmup_median_delay,
        "seeded_channel_estimator": seeded_channel_estimator,
        "tau_list": tau_list,
        "a_ref": a_ref,
    }

def _mean_array_from_records(records, key, length, fallback):
    if records:
        stacked = np.asarray([np.asarray(item[key], dtype=float) for item in records], dtype=float)
        if stacked.ndim == 2 and stacked.shape[1] == int(length):
            return np.mean(stacked, axis=0)
    return np.asarray(fallback, dtype=float).reshape(int(length))

def _solve_policy_snapshot(policy, a_t, tau_t, c_hat, deadline, lambda_value, window_id):
    start = time.perf_counter()
    if hasattr(policy, "epsilon"):
        policy.lambda_t = _sanitize_lambda_value(lambda_value, policy.epsilon)
    else:
        policy.lambda_t = float(lambda_value)
    selection = policy.select_eta(
        a_t=np.asarray(a_t, dtype=float),
        tau_t=np.asarray(tau_t, dtype=float),
        c_hat=np.asarray(c_hat, dtype=float),
        deadline=float(deadline),
    )
    return {
        "window_id": int(window_id),
        "lambda_value": float(policy.lambda_t),
        "selection": selection,
        "solver_elapsed_sec": time.perf_counter() - start,
        "c_hat": [float(item) for item in np.asarray(c_hat, dtype=float)],
        "a_t": [float(item) for item in np.asarray(a_t, dtype=float)],
        "tau_t": [float(item) for item in np.asarray(tau_t, dtype=float)],
    }

def _run_policy(transport, partition, batches, device, profile_name, profile_spec, t0_ratio, t0_deadline, seeded_channel_estimator, tau_list, a_ref, policy, decoder_start_token_id, verbalizers, dynamic_timeslot_size, max_dynamic_timeslot_count=None, max_total_batches=None):
    logging.info(
        "[A][%s][%s][t0=%.2f][dyn=%d] Policy start deadline=%.4fs",
        profile_name,
        policy.policy_name,
        t0_ratio,
        int(dynamic_timeslot_size),
        t0_deadline,
    )
    channel_estimator = copy.deepcopy(seeded_channel_estimator)
    policy.initialize_from_channel_estimator(channel_estimator)
    pending = {}
    dispatched = 0
    completed = 0
    total_correct = 0
    total_seen = 0
    base_sample_count = len(batches)
    if base_sample_count <= 0:
        raise RuntimeError("No samples available for policy execution")
    max_window_count = (
        None if max_dynamic_timeslot_count is None else max(1, int(max_dynamic_timeslot_count))
    )
    max_total_batches = None if max_total_batches is None else max(1, int(max_total_batches))
    task_profiles = []
    accuracy_by_t = []
    delay_by_t = []
    lambda_by_t = []
    requested_eta_by_t = []
    executed_eta_by_t = []
    predicted_accuracy_by_t = []
    predicted_delay_by_t = []
    current_c_hat = np.asarray(channel_estimator.predict_all(), dtype=float)
    initial_solution = _solve_policy_snapshot(
        policy=policy,
        a_t=np.asarray(a_ref, dtype=float),
        tau_t=np.asarray(tau_list, dtype=float),
        c_hat=current_c_hat,
        deadline=t0_deadline,
        lambda_value=float(policy.lambda_t),
        window_id=0,
    )
    current_control_state = {
        "policy_version": 0,
        "window_id": 0,
        "requested_eta": [float(item) for item in np.asarray(initial_solution["selection"]["requested_eta"], dtype=float)],
        "predicted_accuracy": initial_solution["selection"]["predicted_accuracy"],
        "predicted_delay": initial_solution["selection"]["predicted_delay"],
        "solver_z": initial_solution["selection"]["solver_z"],
        "lambda_value": float(initial_solution["lambda_value"]),
        "solver_elapsed_sec": float(initial_solution["solver_elapsed_sec"]),
        "c_hat": list(initial_solution["c_hat"]),
        "window_avg_a": list(initial_solution["a_t"]),
        "window_avg_tau": list(initial_solution["tau_t"]),
        "window_size": 0,
    }
    active_window_records = []
    solver_job = None
    latest_window_id = 0
    next_policy_version = 1
    window_summaries = []
    window_summary_by_id = {}

    def _maybe_collect_solver_result():
        nonlocal solver_job, current_control_state, next_policy_version
        if solver_job is None or not solver_job.done():
            return
        solved = solver_job.result()
        selection = solved["selection"]
        current_control_state = {
            "policy_version": int(next_policy_version),
            "window_id": int(solved["window_id"]),
            "requested_eta": [float(item) for item in np.asarray(selection["requested_eta"], dtype=float)],
            "predicted_accuracy": selection["predicted_accuracy"],
            "predicted_delay": selection["predicted_delay"],
            "solver_z": selection["solver_z"],
            "lambda_value": float(solved["lambda_value"]),
            "solver_elapsed_sec": float(solved["solver_elapsed_sec"]),
            "c_hat": list(solved["c_hat"]),
            "window_avg_a": list(solved["a_t"]),
            "window_avg_tau": list(solved["tau_t"]),
            "window_size": 0,
        }
        summary_row = window_summary_by_id.get(int(solved["window_id"]))
        if summary_row is not None:
            current_control_state["window_size"] = int(summary_row.get("sample_count", 0))
            summary_row["solver_elapsed_sec"] = float(solved["solver_elapsed_sec"])
            summary_row["lambda_input"] = float(solved["lambda_value"])
            summary_row["predicted_accuracy"] = (
                np.nan if selection["predicted_accuracy"] is None else float(selection["predicted_accuracy"])
            )
            summary_row["predicted_delay_sec"] = (
                np.nan if selection["predicted_delay"] is None else float(selection["predicted_delay"])
            )
            summary_row["solver_z"] = float(selection["solver_z"])
            requested_eta = np.asarray(selection["requested_eta"], dtype=float)
            for idx, value in enumerate(requested_eta):
                summary_row["requested_eta_{}".format(idx)] = float(value)
        next_policy_version += 1
        solver_job = None

    def _try_start_solver(executor):
        nonlocal latest_window_id, active_window_records, solver_job
        if solver_job is not None:
            return
        if len(active_window_records) < max(1, int(dynamic_timeslot_size)):
            return
        if max_window_count is not None and latest_window_id >= max_window_count:
            return
        latest_window_id += 1
        snapshot_records = list(active_window_records)
        active_window_records = []
        avg_a = _mean_array_from_records(snapshot_records, "a_values", NUM_TRANSFER_POINTS, a_ref)
        avg_tau = _mean_array_from_records(snapshot_records, "tau_values", NUM_PARTITIONS, tau_list)
        avg_c = _mean_array_from_records(snapshot_records, "c_values", NUM_TRANSFER_POINTS, current_control_state["c_hat"])
        avg_delay = _finite_mean((float(item["delay"]) for item in snapshot_records), float(t0_deadline))
        total_window_correct = int(sum(int(item.get("correct_count", 0)) for item in snapshot_records))
        total_window_samples = int(sum(int(item.get("sample_total", 0)) for item in snapshot_records))
        avg_acc = (
            float(total_window_correct) / float(total_window_samples)
            if total_window_samples > 0 else np.nan
        )
        if hasattr(policy, "epsilon"):
            lambda_value = _sanitize_lambda_value(
                float(current_control_state["lambda_value"]) + avg_delay - float(t0_deadline),
                policy.epsilon,
            )
        else:
            lambda_value = float(current_control_state["lambda_value"])
        summary_row = {
            "window_id": int(latest_window_id),
            "algorithm": policy.policy_name,
            "policy": policy.policy_name,
            "policy_key": policy.policy_key,
            "dynamic_timeslot_size": int(dynamic_timeslot_size),
            "sample_count": int(len(snapshot_records)),
            "batch_count": int(len(snapshot_records)),
            "window_total_samples": int(total_window_samples),
            "window_correct_samples": int(total_window_correct),
            "start_t_index": int(snapshot_records[0]["t_index"]) if snapshot_records else np.nan,
            "end_t_index": int(snapshot_records[-1]["t_index"]) if snapshot_records else np.nan,
            "start_dataset_epoch": int(snapshot_records[0]["dataset_epoch"]) if snapshot_records else np.nan,
            "end_dataset_epoch": int(snapshot_records[-1]["dataset_epoch"]) if snapshot_records else np.nan,
            "start_dataset_index": int(snapshot_records[0]["dataset_index"]) if snapshot_records else np.nan,
            "end_dataset_index": int(snapshot_records[-1]["dataset_index"]) if snapshot_records else np.nan,
            "start_batch_idx": int(snapshot_records[0]["batch_idx"]) if snapshot_records else np.nan,
            "end_batch_idx": int(snapshot_records[-1]["batch_idx"]) if snapshot_records else np.nan,
            "start_sample_id": str(snapshot_records[0]["sample_id"]) if snapshot_records else None,
            "end_sample_id": str(snapshot_records[-1]["sample_id"]) if snapshot_records else None,
            "avg_accuracy": float(avg_acc) if not np.isnan(avg_acc) else np.nan,
            "avg_delay_sec": float(avg_delay),
            "target_time_sec": float(t0_deadline),
            "avg_excess_delay_sec": float(avg_delay - float(t0_deadline)),
            "lambda_input": float(lambda_value),
            "solver_elapsed_sec": np.nan,
            "predicted_accuracy": np.nan,
            "predicted_delay_sec": np.nan,
            "solver_z": np.nan,
        }
        for idx, value in enumerate(np.asarray(avg_a, dtype=float)):
            summary_row["avg_a_{}".format(idx)] = float(value)
        for idx, value in enumerate(np.asarray(avg_tau, dtype=float)):
            summary_row["avg_tau_{}".format(idx)] = float(value)
        for idx, value in enumerate(np.asarray(avg_c, dtype=float)):
            summary_row["avg_c_{}".format(idx)] = float(value)
        window_summaries.append(summary_row)
        window_summary_by_id[int(latest_window_id)] = summary_row
        solver_job = executor.submit(
            _solve_policy_snapshot,
            policy,
            np.asarray(avg_a, dtype=float),
            np.asarray(avg_tau, dtype=float),
            np.asarray(avg_c, dtype=float),
            t0_deadline,
            float(lambda_value),
            int(latest_window_id),
        )

    def _maybe_advance_control(executor):
        previous_job = solver_job
        _maybe_collect_solver_result()
        if previous_job is None or solver_job is None:
            _try_start_solver(executor)

    def _dispatch_allowed():
        if max_total_batches is not None and dispatched >= int(max_total_batches):
            return False
        if max_window_count is None:
            return dispatched < int(base_sample_count)
        return int(latest_window_id) < int(max_window_count)

    with ThreadPoolExecutor(max_workers=1) as solver_executor:
        while True:
            _maybe_advance_control(solver_executor)
            while _dispatch_allowed() and len(pending) < ASYNC_MAX_INFLIGHT_TASKS:
                dataset_index = int(dispatched % base_sample_count)
                dataset_epoch = int(dispatched // base_sample_count)
                batch = batches[dataset_index]
                execution_plan = _build_execution_plan(profile_spec, current_control_state["requested_eta"])
                task_id = "{}_{}_{}_{}".format(profile_name, policy.policy_key, dispatched, uuid.uuid4().hex[:8])
                task_state = _dispatch_async_task(
                    transport,
                    partition,
                    batch,
                    task_id,
                    execution_plan,
                    device,
                    decoder_start_token_id,
                    verbalizers,
                )
                task_state["t_index"] = dispatched
                task_state["policy_name"] = policy.policy_name
                task_state["policy_key"] = policy.policy_key
                task_state["selected_accuracy_est"] = current_control_state["predicted_accuracy"]
                task_state["selected_delay_est"] = current_control_state["predicted_delay"]
                task_state["solver_z"] = current_control_state["solver_z"]
                task_state["lambda_before"] = current_control_state["lambda_value"]
                task_state["c_hat"] = list(current_control_state["c_hat"])
                task_state["policy_version"] = int(current_control_state["policy_version"])
                task_state["window_id"] = int(current_control_state["window_id"])
                task_state["dataset_epoch"] = int(dataset_epoch)
                task_state["dataset_index"] = int(dataset_index)
                task_state["window_avg_a"] = list(current_control_state["window_avg_a"])
                task_state["window_avg_tau"] = list(current_control_state["window_avg_tau"])
                task_state["solver_elapsed_sec"] = float(current_control_state["solver_elapsed_sec"])
                pending[task_id] = task_state
                dispatched += 1

            envelope = _collect_result_payload(transport)
            if envelope is None:
                if (not _dispatch_allowed()) and not pending:
                    break
                time.sleep(ASYNC_QUEUE_POLL_SEC)
                continue
            payload = envelope["payload"]
            task_id = payload["task_id"]
            if task_id not in pending:
                continue
            task_state = pending.pop(task_id)
            task_state["worker_stats"].update(payload.get("worker_stats_chain", {}))

            t_index = int(task_state["t_index"])
            predicted_label = int(payload["predicted_label"])
            label = int(task_state["label"])
            correct = int(predicted_label == label)
            total_correct += correct
            total_seen += 1
            running_acc = float(total_correct) / float(total_seen) if total_seen > 0 else 0.0
            end_to_end_delay = time.perf_counter() - task_state["started_at"]

            stats_a = task_state["worker_stats"].get("A")
            stats_b = task_state["worker_stats"].get("B")
            stats_c = task_state["worker_stats"].get("C")
            profile_b = stats_b["profile"] if stats_b is not None else _worker_profile("B", 0.0, 0.0, 0.0, 0, 0.0)
            profile_c = stats_c["profile"] if stats_c is not None else _worker_profile("C", 0.0, 0.0, 0.0, 0, 0.0)
            tp0_stats = stats_a["tp_stats"] if stats_a is not None else task_state["tp_stats"].get("A", {"bytes": 0, "elapsed": 0.0})
            tp1_stats = stats_b["tp_stats"] if stats_b is not None else {"bytes": 0, "elapsed": 0.0}
            tp2_stats = stats_c["tp_stats"] if stats_c is not None else {"bytes": 0, "elapsed": 0.0}
            tp_stats = [
                (tp0_stats["bytes"], tp0_stats["elapsed"]),
                (tp1_stats["bytes"], tp1_stats["elapsed"]),
                (tp2_stats["bytes"], tp2_stats["elapsed"]),
            ]
            channel_estimator.observe_task(tp_stats)
            policy.observe_channel(tp_stats)
            c_values = np.asarray(
                [
                    (float(item[0]) / float(item[1])) if float(item[1]) > 0.0 else 0.0
                    for item in tp_stats
                ],
                dtype=float,
            )
            tau_values = np.asarray(
                [
                    task_state["profiles"]["A"]["service_total_sec"],
                    profile_b["service_total_sec"],
                    profile_c["service_total_sec"],
                    payload["profile_d"]["service_total_sec"],
                ],
                dtype=float,
            )
            actual_delay = float(
                compute_pipeline_delay(
                    np.asarray(task_state["executed_eta"], dtype=float),
                    np.asarray(payload.get("tp_original_bytes", [1.0] * NUM_TRANSFER_POINTS), dtype=float),
                    tau_values,
                    c_values,
                )
            )
            active_window_records.append(
                {
                    "t_index": int(t_index),
                    "dataset_epoch": int(task_state["dataset_epoch"]),
                    "dataset_index": int(task_state["dataset_index"]),
                    "batch_idx": int(payload["batch_idx"]),
                    "sample_id": payload["sample_id"],
                    "a_values": np.asarray(payload.get("tp_original_bytes", [1.0] * NUM_TRANSFER_POINTS), dtype=float),
                    "tau_values": tau_values,
                    "c_values": c_values,
                    "delay": float(actual_delay),
                    "accuracy": float(correct),
                    "correct_count": int(correct),
                    "sample_total": 1,
                }
            )
            _maybe_advance_control(solver_executor)

            accuracy_by_t.append(running_acc)
            delay_by_t.append(actual_delay)
            lambda_by_t.append(float(task_state["lambda_before"]))
            requested_eta_by_t.append(list(task_state["requested_eta"]))
            executed_eta_by_t.append(list(task_state["executed_eta"]))
            predicted_accuracy_by_t.append(np.nan if task_state["selected_accuracy_est"] is None else float(task_state["selected_accuracy_est"]))
            predicted_delay_by_t.append(np.nan if task_state["selected_delay_est"] is None else float(task_state["selected_delay_est"]))

            profile_row = {
                "compressor_profile": profile_name,
                "compressor_display_name": profile_spec["display_name"],
                "t0_ratio": float(t0_ratio),
                "t0_deadline_sec": float(t0_deadline),
                "dynamic_timeslot_size": int(dynamic_timeslot_size),
                "policy": policy.policy_name,
                "task_id": task_id,
                "batch_idx": payload["batch_idx"],
                "sample_id": payload["sample_id"],
                "dataset_epoch": int(task_state["dataset_epoch"]),
                "dataset_index": int(task_state["dataset_index"]),
                "t_index": t_index,
                "sample_correct": float(correct),
                "correct_count": int(correct),
                "sample_total": 1,
                "batch_accuracy": running_acc,
                "running_accuracy": running_acc,
                "lambda_before": task_state["lambda_before"],
                "lambda_after": float(task_state["lambda_before"]),
                "requested_eta": ",".join(_format_float4(item) for item in task_state["requested_eta"]),
                "executed_eta": ",".join(_format_float4(item) for item in task_state["executed_eta"]),
                "predicted_accuracy": task_state["selected_accuracy_est"],
                "predicted_delay_sec": task_state["selected_delay_est"],
                "solver_z": task_state["solver_z"],
                "solver_elapsed_sec": float(task_state["solver_elapsed_sec"]),
                "policy_version": int(task_state["policy_version"]),
                "window_id": int(task_state["window_id"]),
                "compressor_name": task_state["compressor_name"],
                "mapping_distance_l2": float(task_state["mapping_info"]["match_distance_l2"]) if task_state.get("mapping_info") else 0.0,
                "total_delay_sec": actual_delay,
                "end_to_end_delay_sec": float(end_to_end_delay),
                "tp0_bytes": tp0_stats["bytes"],
                "tp1_bytes": tp1_stats["bytes"],
                "tp2_bytes": tp2_stats["bytes"],
                "tp0_elapsed": tp0_stats["elapsed"],
                "tp1_elapsed": tp1_stats["elapsed"],
                "tp2_elapsed": tp2_stats["elapsed"],
                "c_hat_0": float(task_state["c_hat"][0]),
                "c_hat_1": float(task_state["c_hat"][1]),
                "c_hat_2": float(task_state["c_hat"][2]),
                "predicted_label": predicted_label,
                "label": label,
                "positive_logit": float(payload["positive_logit"]),
                "negative_logit": float(payload["negative_logit"]),
                **_flatten_dict({"profiles": {"A": task_state["profiles"]["A"], "B": profile_b, "C": profile_c, "D": payload["profile_d"]}}),
            }
            task_profiles.append(profile_row)
            completed += 1

            logging.info(
                "[A][%s][%s][t0=%.2f][dyn=%d] t=%d sample=%s req_eta=%s exec_eta=%s c=%s c_hat=%s acc=%.4f delay=%.4fs service=%.4fs comm=%.4fs channel_mbps=%s lambda=%.4f window=%d policy_v=%d",
                profile_name,
                policy.policy_name,
                t0_ratio,
                int(dynamic_timeslot_size),
                t_index,
                payload["sample_id"],
                task_state["requested_eta"],
                task_state["executed_eta"],
                [float(x) for x in c_values],
                [float(x) for x in task_state["c_hat"]],
                running_acc,
                actual_delay,
                float(np.max(np.asarray(tau_values, dtype=float))) if len(tau_values) > 0 else 0.0,
                float(np.max(np.asarray([item[1] for item in tp_stats], dtype=float))) if tp_stats else 0.0,
                [round(float(x) / 1e6, 4) for x in np.asarray(c_values, dtype=float)],
                task_state["lambda_before"],
                int(task_state["window_id"]),
                int(task_state["policy_version"]),
            )

        while solver_job is not None or (
            max_window_count is None and len(active_window_records) >= max(1, int(dynamic_timeslot_size))
        ):
            _maybe_advance_control(solver_executor)
            if solver_job is None and (
                max_window_count is not None or len(active_window_records) < max(1, int(dynamic_timeslot_size))
            ):
                break
            time.sleep(ASYNC_QUEUE_POLL_SEC)

    return {
        "compressor_profile": profile_name,
        "compressor_display_name": profile_spec["display_name"],
        "accuracy_estimator_mode": getattr(policy.acc_model, "mode_name", None) if hasattr(policy, "acc_model") else None,
        "dynamic_timeslot_size": int(dynamic_timeslot_size),
        "dynamic_timeslot_mode": "tumbling_window_count",
        "max_dynamic_timeslot_count": (int(max_window_count) if max_window_count is not None else None),
        "policy_name": policy.policy_name,
        "policy_key": policy.policy_key,
        "mu_value": (float(policy.mu) if policy.mu is not None else None),
        "epsilon_value": (float(policy.epsilon) if hasattr(policy, "epsilon") else None),
        "accuracy_list": accuracy_by_t,
        "delay_list": delay_by_t,
        "lambda_list": lambda_by_t,
        "requested_eta_history": requested_eta_by_t,
        "executed_eta_history": executed_eta_by_t,
        "predicted_accuracy_history": predicted_accuracy_by_t,
        "predicted_delay_history": predicted_delay_by_t,
        "window_summaries": window_summaries,
        "task_profiles": list(task_profiles),
        "t0_deadline_sec": float(t0_deadline),
    }
