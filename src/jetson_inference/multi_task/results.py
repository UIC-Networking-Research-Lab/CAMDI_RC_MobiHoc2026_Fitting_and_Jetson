"""Result aggregation and saved figure context."""

import json
import logging
import numpy as np
import os
import pandas as pd

from jetson_inference.analysis.multi_task import render_trial_plots as render_multi_trial_plots
from jetson_inference.multi_task.config import (
    NUM_TRANSFER_POINTS,
    _format_float4,
    _format_target_dir_name,
    _format_tx_limit_tag,
)


def _accuracy_summary_from_rows(rows):
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {}
    metric_column = "accuracy_true" if "accuracy_true" in frame.columns else "weighted_accuracy_true"
    grouped = frame.groupby("algorithm")[metric_column].mean().sort_values(ascending=False)
    best_algorithm = str(grouped.index[0])
    return {
        "policy_count": int(grouped.shape[0]),
        "mean_accuracy_across_policies": float(grouped.mean()),
        "best_accuracy_policy": best_algorithm,
        "best_avg_weighted_accuracy_true": float(grouped.iloc[0]),
    }

def _finite_mean(values):
    finite_values = [float(value) for value in values if not pd.isna(value)]
    if not finite_values:
        return np.nan
    return float(np.mean(np.asarray(finite_values, dtype=float)))

def _completion_rows_to_batch_timeseries_for_task(
    completion_rows,
    task_def,
    codec_name,
    dynamic_timeslot_size=None,
    algorithm_name=None,
    algorithm_type=None,
):
    task_id = int(task_def.logical_task_id)
    target_rate_hz = float(task_def.target_rate_hz)
    target_time_sec = (1.0 / target_rate_hz) if target_rate_hz > 0.0 else np.nan
    task_rows = [
        dict(row)
        for row in completion_rows
        if int(row.get("logical_task_id", -1)) == int(task_id)
    ]
    timeseries_rows = []
    for idx, row in enumerate(task_rows):
        requested_eta = [float(x) for x in row.get("requested_eta", [])]
        executed_eta = [float(x) for x in row.get("executed_eta", requested_eta)]
        pred_delay = row.get("selected_delay_est", np.nan)
        batch_row = {
            "t": int(idx),
            "logical_task_id": int(task_id),
            "series_scope": "task{}".format(int(task_id)),
            "algorithm": str(algorithm_name if algorithm_name is not None else row.get("algorithm", "")),
            "policy": str(algorithm_name if algorithm_name is not None else row.get("policy", row.get("algorithm", ""))),
            "policy_key": (algorithm_type if algorithm_type is not None else row.get("policy_key")),
            "algorithm_type": (algorithm_type if algorithm_type is not None else row.get("policy_key")),
            "codec_name": str(codec_name),
            "mu": float(row["mu"]) if row.get("mu") is not None and not pd.isna(row.get("mu")) else np.nan,
            "epsilon": float(row["epsilon"]) if row.get("epsilon") is not None and not pd.isna(row.get("epsilon")) else np.nan,
            "target_rate_hz": float(target_rate_hz) if target_rate_hz > 0.0 else np.nan,
            "target_time_sec": float(target_time_sec) if not pd.isna(target_time_sec) else np.nan,
            "accuracy_true": float(row.get("running_accuracy", np.nan)),
            "accuracy_fit": float(row["selected_accuracy_est"]) if row.get("selected_accuracy_est") is not None and not pd.isna(row.get("selected_accuracy_est")) else np.nan,
            "delay_actual_sec": float(row.get("delay_sec", np.nan)),
            "delay_pred_sec": float(pred_delay) if pred_delay is not None and not pd.isna(pred_delay) else np.nan,
            "delay_ratio_actual": (
                float(row.get("delay_sec", np.nan)) * float(target_rate_hz)
                if target_rate_hz > 0.0 and not pd.isna(row.get("delay_sec", np.nan))
                else np.nan
            ),
            "delay_ratio_pred": (
                float(pred_delay) * float(target_rate_hz)
                if target_rate_hz > 0.0 and pred_delay is not None and not pd.isna(pred_delay)
                else np.nan
            ),
            "lambda_value": float(row.get("lambda_value", np.nan)) if row.get("lambda_value") is not None else np.nan,
            "accuracy": float(row.get("running_accuracy", np.nan)),
            "delay_sec": float(row.get("delay_sec", np.nan)),
            "lambda": float(row.get("lambda_value", np.nan)) if row.get("lambda_value") is not None else np.nan,
            "sample_correct": float(row.get("sample_correct", np.nan)),
            "correct_count": int(row.get("correct_count", 0)),
            "sample_total": int(row.get("sample_total", 0)),
            "policy_version": int(row.get("policy_version", 0)),
            "window_id": int(row.get("window_id", 0)),
            "batch_idx": int(row.get("batch_idx", -1)),
            "end_to_end_delay_sec": float(row.get("end_to_end_delay_sec", np.nan)),
            "communication_delay_sec": float(row.get("communication_delay_sec", np.nan)),
            "service_delay_sec": float(row.get("service_delay_sec", np.nan)),
            "compressor_name": row.get("compressor_name"),
            "requested_eta": ",".join(_format_float4(x) for x in requested_eta),
            "executed_eta": ",".join(_format_float4(x) for x in executed_eta),
        }
        if dynamic_timeslot_size is not None:
            batch_row["dynamic_timeslot_size"] = int(dynamic_timeslot_size)
        for eta_idx, value in enumerate(requested_eta):
            batch_row["requested_eta_{}".format(eta_idx)] = float(value)
            batch_row["eta_{}".format(eta_idx)] = float(value)
        for eta_idx, value in enumerate(executed_eta):
            batch_row["executed_eta_{}".format(eta_idx)] = float(value)
        timeseries_rows.append(batch_row)
    return timeseries_rows

def _average_numeric(values):
    finite = [float(value) for value in values if value is not None and not pd.isna(value)]
    if not finite:
        return np.nan
    return float(np.mean(np.asarray(finite, dtype=float)))

def _task_weight_map(tasks):
    weights = {int(task.logical_task_id): max(float(task.weight), 0.0) for task in tasks}
    total = float(sum(weights.values()))
    if total <= 1e-12:
        if not weights:
            return {}
        uniform = 1.0 / float(len(weights))
        return {int(task_id): float(uniform) for task_id in weights.keys()}
    return {int(task_id): float(value) / total for task_id, value in weights.items()}

def _weighted_average_by_task(value_by_task, task_weights):
    valid = []
    for task_id, value in value_by_task.items():
        if value is None or pd.isna(value):
            continue
        weight = float(task_weights.get(int(task_id), 0.0))
        if weight <= 0.0:
            continue
        valid.append((weight, float(value)))
    if not valid:
        return np.nan
    denom = float(sum(weight for weight, _ in valid))
    if denom <= 1e-12:
        return np.nan
    return float(sum(weight * value for weight, value in valid) / denom)

def _weighted_average_from_rows_by_task(rows_by_task, field_name, task_weights):
    return _weighted_average_by_task(
        {
            int(task_id): row.get(field_name, np.nan)
            for task_id, row in rows_by_task.items()
        },
        task_weights,
    )

def _task_scope_rows(summary_by_scope):
    rows_by_task = {}
    for scope_key, row in summary_by_scope.items():
        if row is None:
            continue
        scope_text = str(scope_key)
        if not scope_text.startswith("task"):
            continue
        try:
            task_id = int(scope_text.replace("task", ""))
        except Exception:
            continue
        rows_by_task[int(task_id)] = dict(row)
    return rows_by_task

def _apply_weighted_non_accuracy_timeseries_fields(avg_row, rows_by_task, task_weights):
    weighted_fields = [
        "target_rate_hz",
        "target_time_sec",
        "delay_actual_sec",
        "delay_pred_sec",
        "delay_ratio_actual",
        "delay_ratio_pred",
        "lambda_value",
        "delay_sec",
        "lambda",
        "end_to_end_delay_sec",
        "communication_delay_sec",
        "service_delay_sec",
    ]
    for field_name in weighted_fields:
        avg_row[field_name] = _weighted_average_from_rows_by_task(rows_by_task, field_name, task_weights)
    for eta_idx in range(NUM_TRANSFER_POINTS):
        req_value = _weighted_average_from_rows_by_task(rows_by_task, "requested_eta_{}".format(eta_idx), task_weights)
        exec_value = _weighted_average_from_rows_by_task(rows_by_task, "executed_eta_{}".format(eta_idx), task_weights)
        avg_row["requested_eta_{}".format(eta_idx)] = req_value
        avg_row["eta_{}".format(eta_idx)] = req_value
        avg_row["executed_eta_{}".format(eta_idx)] = exec_value
    avg_row["requested_eta"] = ",".join(
        _format_float4(avg_row.get("requested_eta_{}".format(eta_idx), np.nan))
        for eta_idx in range(NUM_TRANSFER_POINTS)
    )
    avg_row["executed_eta"] = ",".join(
        _format_float4(avg_row.get("executed_eta_{}".format(eta_idx), np.nan))
        for eta_idx in range(NUM_TRANSFER_POINTS)
    )
    return avg_row

def _apply_weighted_non_accuracy_summary_fields(avg_row, rows_by_task, task_weights):
    weighted_fields = [
        "target_rate_hz",
        "target_time_sec",
        "avg_delay",
        "avg_excess_delay",
        "avg_delay_ratio",
        "avg_delay_actual_sec",
        "avg_delay_pred_sec",
        "avg_excess_delay_actual_sec",
        "avg_excess_delay_pred_sec",
        "avg_delay_ratio_actual",
        "avg_delay_ratio_pred",
        "avg_delay_sec",
        "violation_rate",
        "final_lambda",
        "lambda_input",
        "solver_elapsed_sec",
        "predicted_delay_sec",
    ]
    for field_name in weighted_fields:
        if any(field_name in row for row in rows_by_task.values()):
            avg_row[field_name] = _weighted_average_from_rows_by_task(rows_by_task, field_name, task_weights)
    return avg_row

def _combine_summary_avg_from_task_scopes(summary_by_scope, tasks, existing_avg_row=None):
    rows_by_task = _task_scope_rows(summary_by_scope)
    if not rows_by_task:
        return existing_avg_row
    avg_row = dict(existing_avg_row) if existing_avg_row is not None else dict(next(iter(rows_by_task.values())))
    return _apply_weighted_non_accuracy_summary_fields(avg_row, rows_by_task, _task_weight_map(tasks))

def _combine_batch_timeseries_avg(batch_rows_by_task, tasks):
    if not batch_rows_by_task:
        return []
    task_weights = _task_weight_map(tasks)
    ordered_task_ids = sorted(int(task_id) for task_id in batch_rows_by_task.keys())
    aligned_len = min((len(rows) for rows in batch_rows_by_task.values()), default=0)
    avg_rows = []
    for idx in range(aligned_len):
        rows_by_task = {
            int(task_id): dict(batch_rows_by_task[task_id][idx])
            for task_id in ordered_task_ids
        }
        source_rows = list(rows_by_task.values())
        if not source_rows:
            continue
        total_samples = int(sum(int(row.get("sample_total", 0)) for row in source_rows))
        total_correct = int(sum(int(row.get("correct_count", 0)) for row in source_rows))
        avg_row = {
            "t": int(idx),
            "logical_task_id": -1,
            "series_scope": "avg",
            "algorithm": str(source_rows[0].get("algorithm", "")),
            "policy": str(source_rows[0].get("policy", source_rows[0].get("algorithm", ""))),
            "policy_key": source_rows[0].get("policy_key"),
            "algorithm_type": source_rows[0].get("algorithm_type", source_rows[0].get("policy_key")),
            "codec_name": source_rows[0].get("codec_name"),
            "mu": _average_numeric(row.get("mu", np.nan) for row in source_rows),
            "epsilon": _average_numeric(row.get("epsilon", np.nan) for row in source_rows),
            "target_rate_hz": np.nan,
            "target_time_sec": np.nan,
            "accuracy_true": (
                float(total_correct) / float(total_samples)
                if total_samples > 0 else _average_numeric(row.get("accuracy_true", np.nan) for row in source_rows)
            ),
            "accuracy_fit": _average_numeric(row.get("accuracy_fit", np.nan) for row in source_rows),
            "delay_actual_sec": np.nan,
            "delay_pred_sec": np.nan,
            "delay_ratio_actual": np.nan,
            "delay_ratio_pred": np.nan,
            "lambda_value": np.nan,
            "accuracy": (
                float(total_correct) / float(total_samples)
                if total_samples > 0 else _average_numeric(row.get("accuracy", np.nan) for row in source_rows)
            ),
            "delay_sec": np.nan,
            "lambda": np.nan,
            "sample_correct": _average_numeric(row.get("sample_correct", np.nan) for row in source_rows),
            "correct_count": int(total_correct),
            "sample_total": int(total_samples),
            "policy_version": int(max(int(row.get("policy_version", 0)) for row in source_rows)),
            "window_id": int(max(int(row.get("window_id", 0)) for row in source_rows)),
            "batch_idx": int(idx),
            "end_to_end_delay_sec": np.nan,
            "communication_delay_sec": np.nan,
            "service_delay_sec": np.nan,
            "compressor_name": source_rows[0].get("compressor_name"),
            "requested_eta": "",
            "executed_eta": "",
        }
        if "dynamic_timeslot_size" in source_rows[0] and not pd.isna(source_rows[0].get("dynamic_timeslot_size")):
            avg_row["dynamic_timeslot_size"] = int(source_rows[0]["dynamic_timeslot_size"])
        avg_row = _apply_weighted_non_accuracy_timeseries_fields(avg_row, rows_by_task, task_weights)
        avg_rows.append(avg_row)
    return avg_rows

def _build_batch_timeseries_sets(
    completion_rows,
    tasks,
    codec_name,
    dynamic_timeslot_size=None,
    algorithm_name=None,
    algorithm_type=None,
):
    batch_rows_by_scope = {}
    batch_rows_by_task = {}
    for task in tasks:
        task_rows = _completion_rows_to_batch_timeseries_for_task(
            completion_rows,
            task,
            codec_name,
            dynamic_timeslot_size=dynamic_timeslot_size,
            algorithm_name=algorithm_name,
            algorithm_type=algorithm_type,
        )
        scope_key = "task{}".format(int(task.logical_task_id))
        batch_rows_by_scope[scope_key] = task_rows
        batch_rows_by_task[int(task.logical_task_id)] = task_rows
    batch_rows_by_scope["avg"] = _combine_batch_timeseries_avg(batch_rows_by_task, tasks)
    return batch_rows_by_scope

def _batch_timeseries_to_window_summaries(batch_rows, dynamic_timeslot_size, anchor_task_id):
    if not batch_rows:
        return []
    window_size = max(int(dynamic_timeslot_size or 1), 1)
    window_rows = []
    for window_idx, start in enumerate(range(0, len(batch_rows), window_size), start=1):
        chunk = list(batch_rows[start:start + window_size])
        if not chunk:
            continue
        total_samples = int(sum(int(item.get("sample_total", 0)) for item in chunk))
        total_correct = int(sum(int(item.get("correct_count", 0)) for item in chunk))
        avg_delay_sec = _finite_mean(item.get("delay_actual_sec", np.nan) for item in chunk)
        target_time_sec = float(chunk[-1].get("target_time_sec", np.nan))
        avg_accuracy = (
            float(total_correct) / float(total_samples)
            if total_samples > 0 else np.nan
        )
        first_row = dict(chunk[0])
        last_row = dict(chunk[-1])
        summary_row = {
            "window_id": int(window_idx),
            "dynamic_timeslot_size": int(window_size),
            "actual_window_size": int(len(chunk)),
            "anchor_task_id": int(anchor_task_id),
            "anchor_completion_count": int(len(chunk)),
            "sample_count": int(len(chunk)),
            "batch_count": int(len(chunk)),
            "window_total_samples": int(total_samples),
            "window_correct_samples": int(total_correct),
            "start_t_index": int(chunk[0].get("t", 0)),
            "end_t_index": int(chunk[-1].get("t", 0)),
            "start_batch_idx": int(chunk[0].get("batch_idx", -1)),
            "end_batch_idx": int(chunk[-1].get("batch_idx", -1)),
            "algorithm": str(first_row.get("algorithm", "")),
            "policy": str(first_row.get("policy", first_row.get("algorithm", ""))),
            "policy_key": first_row.get("policy_key"),
            "algorithm_type": first_row.get("algorithm_type", first_row.get("policy_key")),
            "codec_name": first_row.get("codec_name"),
            "mu": float(first_row["mu"]) if first_row.get("mu") is not None and not pd.isna(first_row.get("mu")) else np.nan,
            "epsilon": float(first_row["epsilon"]) if first_row.get("epsilon") is not None and not pd.isna(first_row.get("epsilon")) else np.nan,
            "avg_accuracy": float(avg_accuracy) if not pd.isna(avg_accuracy) else np.nan,
            "avg_delay_sec": float(avg_delay_sec) if not pd.isna(avg_delay_sec) else np.nan,
            "target_time_sec": float(target_time_sec) if not pd.isna(target_time_sec) else np.nan,
            "avg_excess_delay_sec": (
                float(avg_delay_sec - target_time_sec)
                if not pd.isna(avg_delay_sec) and not pd.isna(target_time_sec)
                else np.nan
            ),
            "lambda_input": float(last_row.get("lambda_value", np.nan)) if last_row.get("lambda_value") is not None else np.nan,
            "solver_elapsed_sec": np.nan,
            "predicted_accuracy": _finite_mean(item.get("accuracy_fit", np.nan) for item in chunk),
            "predicted_delay_sec": _finite_mean(item.get("delay_pred_sec", np.nan) for item in chunk),
            "policy_version": int(last_row.get("policy_version", 0)),
        }
        for eta_idx in range(NUM_TRANSFER_POINTS):
            eta_value = last_row.get("requested_eta_{}".format(eta_idx), np.nan)
            if not pd.isna(eta_value):
                summary_row["requested_eta_{}".format(eta_idx)] = float(eta_value)
        window_rows.append(summary_row)
    return window_rows

def _combine_window_summaries_avg(window_rows_by_scope, tasks, anchor_task_id):
    task_weights = _task_weight_map(tasks)
    task_window_rows = {
        int(task.logical_task_id): list(window_rows_by_scope.get("task{}".format(int(task.logical_task_id)), []))
        for task in tasks
    }
    max_len = max((len(rows) for rows in task_window_rows.values()), default=0)
    avg_rows = []
    for idx in range(max_len):
        rows_by_task = {
            int(task_id): dict(rows[idx])
            for task_id, rows in task_window_rows.items()
            if idx < len(rows)
        }
        if not rows_by_task:
            continue
        source_rows = list(rows_by_task.values())
        total_samples = int(sum(int(row.get("window_total_samples", 0)) for row in source_rows))
        total_correct = int(sum(int(row.get("window_correct_samples", 0)) for row in source_rows))
        first_row = dict(source_rows[0])
        avg_row = {
            "window_id": int(max(int(row.get("window_id", idx + 1)) for row in source_rows)),
            "dynamic_timeslot_size": int(first_row.get("dynamic_timeslot_size", 1)),
            "actual_window_size": int(sum(int(row.get("actual_window_size", 0)) for row in source_rows)),
            "anchor_task_id": int(anchor_task_id),
            "anchor_completion_count": int(sum(int(row.get("anchor_completion_count", 0)) for row in source_rows)),
            "sample_count": int(sum(int(row.get("sample_count", 0)) for row in source_rows)),
            "batch_count": int(sum(int(row.get("batch_count", 0)) for row in source_rows)),
            "window_total_samples": int(total_samples),
            "window_correct_samples": int(total_correct),
            "start_t_index": int(min(int(row.get("start_t_index", 0)) for row in source_rows)),
            "end_t_index": int(max(int(row.get("end_t_index", 0)) for row in source_rows)),
            "start_batch_idx": int(min(int(row.get("start_batch_idx", -1)) for row in source_rows)),
            "end_batch_idx": int(max(int(row.get("end_batch_idx", -1)) for row in source_rows)),
            "algorithm": str(first_row.get("algorithm", "")),
            "policy": str(first_row.get("policy", first_row.get("algorithm", ""))),
            "policy_key": first_row.get("policy_key"),
            "algorithm_type": first_row.get("algorithm_type", first_row.get("policy_key")),
            "codec_name": first_row.get("codec_name"),
            "mu": float(first_row["mu"]) if first_row.get("mu") is not None and not pd.isna(first_row.get("mu")) else np.nan,
            "epsilon": float(first_row["epsilon"]) if first_row.get("epsilon") is not None and not pd.isna(first_row.get("epsilon")) else np.nan,
            "avg_accuracy": (
                float(total_correct) / float(total_samples)
                if total_samples > 0 else np.nan
            ),
            "avg_delay_sec": np.nan,
            "target_time_sec": np.nan,
            "avg_excess_delay_sec": np.nan,
            "lambda_input": np.nan,
            "solver_elapsed_sec": np.nan,
            "predicted_accuracy": _average_numeric(row.get("predicted_accuracy", np.nan) for row in source_rows),
            "predicted_delay_sec": np.nan,
            "policy_version": int(max(int(row.get("policy_version", 0)) for row in source_rows)),
            "series_scope": "avg",
        }
        for eta_idx in range(NUM_TRANSFER_POINTS):
            avg_row["requested_eta_{}".format(eta_idx)] = _weighted_average_from_rows_by_task(
                rows_by_task,
                "requested_eta_{}".format(eta_idx),
                task_weights,
            )
        avg_row = _apply_weighted_non_accuracy_summary_fields(avg_row, rows_by_task, task_weights)
        avg_rows.append(avg_row)
    return avg_rows

def _build_window_summaries_sets(batch_rows_by_scope, dynamic_timeslot_size, anchor_task_id, tasks):
    window_rows_by_scope = {}
    for task in tasks:
        scope_key = "task{}".format(int(task.logical_task_id))
        rows = list(batch_rows_by_scope.get(scope_key, []))
        window_rows = _batch_timeseries_to_window_summaries(
            rows,
            dynamic_timeslot_size=dynamic_timeslot_size,
            anchor_task_id=int(task.logical_task_id),
        )
        for row in window_rows:
            row["series_scope"] = str(scope_key)
        window_rows_by_scope[str(scope_key)] = window_rows
    avg_rows = _combine_window_summaries_avg(window_rows_by_scope, tasks, anchor_task_id)
    for row in avg_rows:
        row["series_scope"] = "avg"
    window_rows_by_scope["avg"] = avg_rows
    return window_rows_by_scope

def _build_policy_summary_row(batch_rows, window_summaries):
    if not batch_rows:
        return None
    first_row = dict(batch_rows[0])
    last_row = dict(batch_rows[-1])
    last_window = dict(window_summaries[-1]) if window_summaries else {}
    target_rate_hz = float(first_row.get("target_rate_hz", np.nan))
    avg_delay_thisslot = float(last_window.get("avg_delay_sec", np.nan))
    acc_thisslot = float(last_window.get("avg_accuracy", np.nan))
    summary_row = {
        **({"dynamic_timeslot_size": int(first_row["dynamic_timeslot_size"])} if "dynamic_timeslot_size" in first_row and not pd.isna(first_row.get("dynamic_timeslot_size")) else {}),
        **({"target_ratio": float(first_row["target_ratio"])} if "target_ratio" in first_row and not pd.isna(first_row.get("target_ratio")) else {}),
        **({"target_label": str(first_row["target_label"])} if "target_label" in first_row else {}),
        **({"accuracy_estimator_mode": str(first_row["accuracy_estimator_mode"])} if "accuracy_estimator_mode" in first_row else {}),
        "algorithm": str(first_row.get("algorithm", "")),
        "policy": str(first_row.get("policy", first_row.get("algorithm", ""))),
        "policy_key": first_row.get("policy_key"),
        "algorithm_type": first_row.get("algorithm_type", first_row.get("policy_key")),
        "mu": float(first_row["mu"]) if first_row.get("mu") is not None and not pd.isna(first_row.get("mu")) else np.nan,
        "epsilon": float(first_row["epsilon"]) if first_row.get("epsilon") is not None and not pd.isna(first_row.get("epsilon")) else np.nan,
        "target_rate_hz": float(first_row.get("target_rate_hz", np.nan)),
        "target_time_sec": float(first_row.get("target_time_sec", np.nan)),
        "window_id": int(last_window.get("window_id", -1)) if last_window else np.nan,
        "sample_count_thisslot": int(last_window.get("sample_count", 0)) if last_window else 0,
        "batch_count_thisslot": int(last_window.get("batch_count", 0)) if last_window else 0,
        "acc_all_samples": float(last_row.get("accuracy_true", np.nan)),
        "acc_thisslot": float(acc_thisslot) if not pd.isna(acc_thisslot) else np.nan,
        "avg_utility": float(acc_thisslot) if not pd.isna(acc_thisslot) else np.nan,
        "avg_acc": float(acc_thisslot) if not pd.isna(acc_thisslot) else np.nan,
        "avg_accuracy": float(acc_thisslot) if not pd.isna(acc_thisslot) else np.nan,
        "avg_delay": float(avg_delay_thisslot) if not pd.isna(avg_delay_thisslot) else np.nan,
        "avg_excess_delay": float(last_window.get("avg_excess_delay_sec", np.nan)) if last_window else np.nan,
        "avg_delay_ratio": (
            float(avg_delay_thisslot) * float(target_rate_hz)
            if not pd.isna(avg_delay_thisslot) and not pd.isna(target_rate_hz)
            else np.nan
        ),
        "lambda_input": float(last_window.get("lambda_input", np.nan)) if last_window else np.nan,
        "solver_elapsed_sec": float(last_window.get("solver_elapsed_sec", np.nan)) if last_window else np.nan,
        "predicted_accuracy": float(last_window.get("predicted_accuracy", np.nan)) if last_window else np.nan,
        "predicted_delay_sec": float(last_window.get("predicted_delay_sec", np.nan)) if last_window else np.nan,
        "avg_accuracy_fit": _finite_mean(item.get("accuracy_fit", np.nan) for item in batch_rows),
        "avg_accuracy_true": _finite_mean(item.get("accuracy_true", np.nan) for item in batch_rows),
        "avg_delay_actual_sec": _finite_mean(item.get("delay_actual_sec", np.nan) for item in batch_rows),
        "avg_delay_pred_sec": _finite_mean(item.get("delay_pred_sec", np.nan) for item in batch_rows),
        "avg_excess_delay_actual_sec": _finite_mean(
            (
                float(item.get("delay_actual_sec", np.nan)) - float(item.get("target_time_sec", np.nan))
                if not pd.isna(item.get("delay_actual_sec", np.nan)) and not pd.isna(item.get("target_time_sec", np.nan))
                else np.nan
            )
            for item in batch_rows
        ),
        "avg_excess_delay_pred_sec": _finite_mean(
            (
                float(item.get("delay_pred_sec", np.nan)) - float(item.get("target_time_sec", np.nan))
                if not pd.isna(item.get("delay_pred_sec", np.nan)) and not pd.isna(item.get("target_time_sec", np.nan))
                else np.nan
            )
            for item in batch_rows
        ),
        "avg_delay_ratio_actual": _finite_mean(item.get("delay_ratio_actual", np.nan) for item in batch_rows),
        "avg_delay_ratio_pred": _finite_mean(item.get("delay_ratio_pred", np.nan) for item in batch_rows),
        "num_batches": int(len(batch_rows)),
        "num_control_steps": int(len(window_summaries)),
    }
    for key, value in first_row.items():
        if str(key).startswith("target_ratio_task") and not pd.isna(value):
            summary_row[str(key)] = float(value)
    return summary_row

def _build_batch_policy_summary_row(batch_rows):
    if not batch_rows:
        return None
    first_row = dict(batch_rows[0])
    target_rate_hz = float(first_row.get("target_rate_hz", np.nan))
    delay_arr = np.asarray([item.get("delay_actual_sec", np.nan) for item in batch_rows], dtype=float)
    pred_delay_arr = np.asarray([item.get("delay_pred_sec", np.nan) for item in batch_rows], dtype=float)
    acc_arr = np.asarray([item.get("accuracy_true", np.nan) for item in batch_rows], dtype=float)
    pred_acc_arr = np.asarray([item.get("accuracy_fit", np.nan) for item in batch_rows], dtype=float)
    target_time_sec = float(first_row.get("target_time_sec", np.nan))
    excess_delay_arr = delay_arr - target_time_sec if not pd.isna(target_time_sec) else np.full_like(delay_arr, np.nan)
    violation_rate = (
        float(np.mean(delay_arr > target_time_sec))
        if delay_arr.size > 0 and not pd.isna(target_time_sec)
        else 0.0
    )
    summary_row = {
        **({"dynamic_timeslot_size": int(first_row["dynamic_timeslot_size"])} if "dynamic_timeslot_size" in first_row and not pd.isna(first_row.get("dynamic_timeslot_size")) else {}),
        **({"target_ratio": float(first_row["target_ratio"])} if "target_ratio" in first_row and not pd.isna(first_row.get("target_ratio")) else {}),
        **({"target_label": str(first_row["target_label"])} if "target_label" in first_row else {}),
        **({"accuracy_estimator_mode": str(first_row["accuracy_estimator_mode"])} if "accuracy_estimator_mode" in first_row else {}),
        "algorithm": str(first_row.get("algorithm", "")),
        "policy": str(first_row.get("policy", first_row.get("algorithm", ""))),
        "policy_key": first_row.get("policy_key"),
        "algorithm_type": first_row.get("algorithm_type", first_row.get("policy_key")),
        "mu": float(first_row["mu"]) if first_row.get("mu") is not None and not pd.isna(first_row.get("mu")) else np.nan,
        "epsilon": float(first_row["epsilon"]) if first_row.get("epsilon") is not None and not pd.isna(first_row.get("epsilon")) else np.nan,
        "target_rate_hz": float(target_rate_hz) if not pd.isna(target_rate_hz) else np.nan,
        "target_time_sec": float(target_time_sec) if not pd.isna(target_time_sec) else np.nan,
        "avg_utility": _finite_mean(acc_arr),
        "avg_acc": _finite_mean(acc_arr),
        "avg_delay": _finite_mean(delay_arr),
        "avg_excess_delay": _finite_mean(excess_delay_arr),
        "avg_delay_ratio": _finite_mean(item.get("delay_ratio_actual", np.nan) for item in batch_rows),
        "avg_accuracy_fit": _finite_mean(pred_acc_arr),
        "avg_accuracy_true": _finite_mean(acc_arr),
        "avg_delay_actual_sec": _finite_mean(delay_arr),
        "avg_delay_pred_sec": _finite_mean(pred_delay_arr),
        "avg_delay_ratio_actual": _finite_mean(item.get("delay_ratio_actual", np.nan) for item in batch_rows),
        "avg_delay_ratio_pred": _finite_mean(item.get("delay_ratio_pred", np.nan) for item in batch_rows),
        "avg_accuracy": _finite_mean(acc_arr),
        "avg_delay_sec": _finite_mean(delay_arr),
        "violation_rate": float(violation_rate),
        "final_lambda": float(batch_rows[-1].get("lambda_value", np.nan)) if batch_rows[-1].get("lambda_value") is not None else np.nan,
        "num_batches": int(len(batch_rows)),
    }
    for key, value in first_row.items():
        if str(key).startswith("target_ratio_task") and not pd.isna(value):
            summary_row[str(key)] = float(value)
    return summary_row

def _build_weighted_avg_policy_summaries(summary_by_scope, tasks):
    existing_avg_row = summary_by_scope.get("avg")
    combined_avg_row = _combine_summary_avg_from_task_scopes(summary_by_scope, tasks, existing_avg_row=existing_avg_row)
    updated = dict(summary_by_scope)
    updated["avg"] = combined_avg_row
    return updated

def _aggregate_policy_summary_rows(summary_rows):
    if not summary_rows:
        return {
            "policy_count": 0,
            "avg_accuracy_over_policies": np.nan,
            "best_accuracy_policy": None,
            "best_avg_accuracy": np.nan,
        }
    frame = pd.DataFrame(summary_rows)
    if frame.empty or "avg_accuracy" not in frame.columns:
        return {
            "policy_count": 0,
            "avg_accuracy_over_policies": np.nan,
            "best_accuracy_policy": None,
            "best_avg_accuracy": np.nan,
        }
    summary = {
        "policy_count": int(frame.shape[0]),
        "avg_accuracy_over_policies": float(frame["avg_accuracy"].mean()),
        "best_accuracy_policy": None,
        "best_avg_accuracy": np.nan,
    }
    if not frame["avg_accuracy"].isna().all():
        best_idx = frame["avg_accuracy"].idxmax()
        summary["best_accuracy_policy"] = str(frame.loc[best_idx, "algorithm"])
        summary["best_avg_accuracy"] = float(frame.loc[best_idx, "avg_accuracy"])
    return summary

def _summarize_policy_rows(rows):
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    summary_rows = []
    group_cols = ["algorithm"]
    if "target_label" in frame.columns:
        group_cols = ["target_label", "algorithm"]
    if "dynamic_timeslot_size" in frame.columns:
        group_cols = ["dynamic_timeslot_size"] + list(group_cols)
    for group_key, group in frame.groupby(group_cols, sort=False):
        dynamic_timeslot_size = None
        target_label = None
        algorithm = None
        if isinstance(group_key, tuple):
            values = list(group_key)
            if "dynamic_timeslot_size" in group_cols:
                dynamic_timeslot_size = values.pop(0)
            if "target_label" in group_cols:
                target_label = values.pop(0)
            algorithm = values.pop(0) if values else None
        else:
            algorithm = group_key
        summary_rows.append(
            {
                **({"dynamic_timeslot_size": int(dynamic_timeslot_size)} if dynamic_timeslot_size is not None else {}),
                **({"target_label": str(target_label)} if target_label is not None else {}),
                **(
                    {"accuracy_estimator_mode": str(group["accuracy_estimator_mode"].iloc[0])}
                    if "accuracy_estimator_mode" in group.columns else {}
                ),
                "algorithm": str(algorithm),
                **({"algorithm_type": str(group["algorithm_type"].iloc[0])} if "algorithm_type" in group.columns else {}),
                **({"mu": float(group["mu"].iloc[0])} if "mu" in group.columns else {}),
                **({"epsilon": float(group["epsilon"].iloc[0])} if "epsilon" in group.columns else {}),
                "avg_utility": float(group["accuracy_true"].mean()) if "accuracy_true" in group.columns else float(group["weighted_utility_true"].mean()),
                "avg_acc": float(group["accuracy_true"].mean()) if "accuracy_true" in group.columns else float(group["weighted_accuracy_true"].mean()),
                "avg_delay": float(group["delay_actual_sec"].mean()) if "delay_actual_sec" in group.columns else float(group["avg_delay_actual_sec"].mean()),
                "avg_excess_delay": float((group["delay_actual_sec"] - group["target_time_sec"]).mean()) if "delay_actual_sec" in group.columns and "target_time_sec" in group.columns else float(group["avg_excess_delay_actual_sec"].mean()),
                "avg_delay_ratio": float(group["delay_ratio_actual"].mean()) if "delay_ratio_actual" in group.columns else float(group["avg_delay_ratio_actual"].mean()),
                "avg_accuracy_fit": float(group["accuracy_fit"].mean()) if "accuracy_fit" in group.columns else np.nan,
                "avg_accuracy_true": float(group["accuracy_true"].mean()) if "accuracy_true" in group.columns else np.nan,
                "avg_weighted_accuracy_fit": float(group["weighted_accuracy_fit"].mean()) if "weighted_accuracy_fit" in group.columns else np.nan,
                "avg_weighted_accuracy_true": float(group["weighted_accuracy_true"].mean()),
                "avg_weighted_utility_fit": float(group["weighted_utility_fit"].mean()) if "weighted_utility_fit" in group.columns else np.nan,
                "avg_weighted_utility_true": float(group["weighted_utility_true"].mean()) if "weighted_utility_true" in group.columns else np.nan,
                "avg_delay_actual_sec": float(group["delay_actual_sec"].mean()) if "delay_actual_sec" in group.columns else float(group["avg_delay_actual_sec"].mean()),
                "avg_delay_pred_sec": float(group["delay_pred_sec"].mean()) if "delay_pred_sec" in group.columns else (float(group["avg_delay_pred_sec"].mean()) if "avg_delay_pred_sec" in group.columns else np.nan),
                "avg_excess_delay_actual_sec": float(group["avg_excess_delay_actual_sec"].mean()),
                "avg_excess_delay_pred_sec": float(group["avg_excess_delay_pred_sec"].mean()) if "avg_excess_delay_pred_sec" in group.columns else np.nan,
                "avg_delay_ratio_actual": float(group["delay_ratio_actual"].mean()) if "delay_ratio_actual" in group.columns else float(group["avg_delay_ratio_actual"].mean()),
                "avg_delay_ratio_pred": float(group["delay_ratio_pred"].mean()) if "delay_ratio_pred" in group.columns else (float(group["avg_delay_ratio_pred"].mean()) if "avg_delay_ratio_pred" in group.columns else np.nan),
                "num_control_steps": int(group.shape[0]),
            }
        )
    return summary_rows

def _annotate_rows_for_target(rows, target_ratio, accuracy_estimator_mode=None, anchor_task_id=None):
    target_label = _format_target_dir_name(target_ratio)
    annotated = []
    for row in rows:
        item = dict(row)
        if isinstance(target_ratio, dict):
            if anchor_task_id is not None and int(anchor_task_id) in target_ratio:
                item["target_ratio"] = float(target_ratio[int(anchor_task_id)])
            for task_id, ratio_value in sorted(target_ratio.items()):
                item["target_ratio_task{}".format(int(task_id))] = float(ratio_value)
        else:
            item["target_ratio"] = float(target_ratio)
        item["target_label"] = str(target_label)
        if accuracy_estimator_mode is not None:
            item["accuracy_estimator_mode"] = str(accuracy_estimator_mode)
        annotated.append(item)
    return annotated

def _write_plot_context(
    output_dir,
    tasks,
    codec_name,
    channel_limit_mbps,
    target_time_ratio,
    accuracy_estimator_mode="fitting_model",
    stein_sigma=None,
    stein_N=None,
    dynamic_timeslot_size=None,
    max_dynamic_timeslot_count=None,
    anchor_task_id=None,
):
    task_entries = []
    for task in tasks:
        target_time_sec = (1.0 / float(task.target_rate_hz)) if float(task.target_rate_hz) > 0.0 else None
        task_entries.append(
            {
                "task_id": int(task.logical_task_id),
                "model": task.model,
                "dataset": task.dataset,
                "batch_size": int(task.batch_size),
                "w_k": float(task.weight),
                "target_rate_hz": float(task.target_rate_hz),
                "target_time_sec": (float(target_time_sec) if target_time_sec is not None else None),
                "scenario_tx_limit_mbps": float(channel_limit_mbps),
            }
        )
    context = {
        "experiment": "online",
        "platform": "jetson",
        "model": "resnet_flan_t5_multi_task",
        "dataset": "mixed",
        "codec_name": codec_name,
        "accuracy_estimator_mode": str(accuracy_estimator_mode),
        "batch_size": "mixed",
        "channel_limit_mbps": float(channel_limit_mbps),
        "per_link_initial_bandwidth_mbps": float(channel_limit_mbps),
        "channel_limit_semantics": "per_link",
        "channel_speed_tag": _format_tx_limit_tag(channel_limit_mbps),
        "delay_definition": "bottleneck_max_stage",
        "tasks": task_entries,
    }
    if isinstance(target_time_ratio, dict):
        context["target_time_ratio_by_task"] = {
            str(int(task_id)): float(ratio_value) for task_id, ratio_value in sorted(target_time_ratio.items())
        }
    elif target_time_ratio is not None:
        context["target_time_ratio"] = float(target_time_ratio)
    if anchor_task_id is not None:
        anchor_task_id = int(anchor_task_id)
        context["anchor_task_id"] = int(anchor_task_id)
        anchor_entry = next((item for item in task_entries if int(item["task_id"]) == int(anchor_task_id)), None)
        if anchor_entry is not None:
            context["target_rate_hz"] = float(anchor_entry["target_rate_hz"])
            context["target_time_sec"] = (
                float(anchor_entry["target_time_sec"]) if anchor_entry.get("target_time_sec") is not None else None
            )
            if isinstance(target_time_ratio, dict) and int(anchor_task_id) in target_time_ratio:
                context["target_time_ratio"] = float(target_time_ratio[int(anchor_task_id)])
    if dynamic_timeslot_size is not None:
        context["dynamic_timeslot_size"] = int(dynamic_timeslot_size)
        context["dynamic_timeslot_mode"] = "anchor_task_min_count"
    if max_dynamic_timeslot_count is not None:
        context["max_dynamic_timeslot_count"] = int(max_dynamic_timeslot_count)
    if str(accuracy_estimator_mode) == "stein_estimator":
        context["stein_sigma"] = float(stein_sigma) if stein_sigma is not None else None
        context["stein_N"] = int(stein_N) if stein_N is not None else None
    with open(os.path.join(output_dir, "plot_context.json"), "w", encoding="utf-8") as handle:
        json.dump(context, handle, indent=2)
    return context

def _maybe_render_plots(trial_dir, plot_context):
    try:
        render_multi_trial_plots(trial_dir, plot_context=plot_context)
    except Exception:
        logging.exception("Plot rendering failed for %s", trial_dir)
