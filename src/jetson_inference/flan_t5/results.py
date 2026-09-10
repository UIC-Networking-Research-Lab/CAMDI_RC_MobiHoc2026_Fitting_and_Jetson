"""Result aggregation and saved figure context."""

import json
import numpy as np
import os
import pandas as pd

from jetson_inference.flan_t5.config import MODEL_TAG


def _safe_nanmean(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.isnan(arr).all():
        return np.nan
    return float(np.nanmean(arr))

def _summarize_policy_results(all_results):
    rows = []
    for result in all_results:
        acc_arr = np.asarray(result["accuracy_list"], dtype=float)
        delay_arr = np.asarray(result["delay_list"], dtype=float)
        pred_acc_arr = np.asarray(result.get("predicted_accuracy_history", []), dtype=float)
        pred_delay_arr = np.asarray(result.get("predicted_delay_history", []), dtype=float)
        deadline = float(result["t0_deadline_sec"])
        target_rate_hz = (1.0 / deadline) if deadline > 0.0 else np.nan
        excess_delay_arr = delay_arr - deadline
        rows.append(
            {
                "compressor_profile": result["compressor_profile"],
                "compressor_display_name": result["compressor_display_name"],
                "accuracy_estimator_mode": result.get("accuracy_estimator_mode"),
                "dynamic_timeslot_size": int(result.get("dynamic_timeslot_size", 0)),
                "target_rate_hz": float(target_rate_hz) if not np.isnan(target_rate_hz) else np.nan,
                "algorithm": result["policy_name"],
                "policy": result["policy_name"],
                "policy_key": result.get("policy_key"),
                "mu": (float(result["mu_value"]) if result.get("mu_value") is not None else np.nan),
                "epsilon": (float(result["epsilon_value"]) if result.get("epsilon_value") is not None else np.nan),
                "avg_utility": _safe_nanmean(acc_arr),
                "avg_acc": _safe_nanmean(acc_arr),
                "avg_delay": _safe_nanmean(delay_arr),
                "avg_excess_delay": _safe_nanmean(excess_delay_arr),
                "avg_delay_ratio": _safe_nanmean(delay_arr * target_rate_hz) if not np.isnan(target_rate_hz) else np.nan,
                "avg_accuracy_fit": _safe_nanmean(pred_acc_arr),
                "avg_accuracy_true": _safe_nanmean(acc_arr),
                "avg_delay_actual_sec": _safe_nanmean(delay_arr),
                "avg_delay_pred_sec": _safe_nanmean(pred_delay_arr),
                "avg_delay_ratio_actual": _safe_nanmean(delay_arr * target_rate_hz) if not np.isnan(target_rate_hz) else np.nan,
                "avg_delay_ratio_pred": _safe_nanmean(pred_delay_arr * target_rate_hz) if not np.isnan(target_rate_hz) else np.nan,
                "avg_accuracy": _safe_nanmean(acc_arr),
                "avg_delay_sec": _safe_nanmean(delay_arr),
                "violation_rate": float(np.mean(delay_arr > deadline)) if len(delay_arr) > 0 else 0.0,
                "final_lambda": float(result["lambda_list"][-1]) if result["lambda_list"] else 0.0,
                "t0_deadline_sec": deadline,
            }
        )
    return rows

def _summarize_last_timeslot(all_results):
    rows = []
    for result in all_results:
        window_summaries = list(result.get("window_summaries", []))
        if not window_summaries:
            continue
        last_window = dict(window_summaries[-1])
        task_profiles = list(result.get("task_profiles", []))
        acc_all_samples = np.nan
        if task_profiles:
            acc_all_samples = float(task_profiles[-1].get("running_accuracy", np.nan))
        deadline = float(result["t0_deadline_sec"])
        target_rate_hz = (1.0 / deadline) if deadline > 0.0 else np.nan
        last_delay = float(last_window.get("avg_delay_sec", np.nan))
        rows.append(
            {
                "compressor_profile": result["compressor_profile"],
                "compressor_display_name": result["compressor_display_name"],
                "accuracy_estimator_mode": result.get("accuracy_estimator_mode"),
                "dynamic_timeslot_size": int(result.get("dynamic_timeslot_size", 0)),
                "target_rate_hz": float(target_rate_hz) if not np.isnan(target_rate_hz) else np.nan,
                "algorithm": result["policy_name"],
                "policy": result["policy_name"],
                "policy_key": result.get("policy_key"),
                "mu": (float(result["mu_value"]) if result.get("mu_value") is not None else np.nan),
                "epsilon": (float(result["epsilon_value"]) if result.get("epsilon_value") is not None else np.nan),
                "window_id": int(last_window.get("window_id", -1)),
                "sample_count_thisslot": int(last_window.get("sample_count", 0)),
                "batch_count_thisslot": int(last_window.get("batch_count", 0)),
                "acc_all_samples": float(acc_all_samples) if not np.isnan(acc_all_samples) else np.nan,
                "acc_thisslot": float(last_window.get("avg_accuracy", np.nan)),
                "avg_utility": float(last_window.get("avg_accuracy", np.nan)),
                "avg_acc": float(last_window.get("avg_accuracy", np.nan)),
                "avg_delay": last_delay,
                "avg_excess_delay": float(last_window.get("avg_excess_delay_sec", np.nan)),
                "avg_delay_ratio": (
                    last_delay * float(target_rate_hz)
                    if not np.isnan(target_rate_hz) and not np.isnan(last_delay)
                    else np.nan
                ),
                "lambda_input": float(last_window.get("lambda_input", np.nan)),
                "solver_elapsed_sec": float(last_window.get("solver_elapsed_sec", np.nan)),
                "predicted_accuracy": float(last_window.get("predicted_accuracy", np.nan)),
                "predicted_delay_sec": float(last_window.get("predicted_delay_sec", np.nan)),
                "solver_z": float(last_window.get("solver_z", np.nan)),
                "t0_deadline_sec": deadline,
            }
        )
    return rows

def _aggregate_accuracy_summary(summary_rows):
    if not summary_rows:
        return {
            "policy_count": 0,
            "avg_accuracy_over_policies": np.nan,
            "best_policy_by_accuracy": None,
            "best_avg_accuracy": np.nan,
        }
    frame = pd.DataFrame.from_records(summary_rows)
    summary = {
        "policy_count": int(len(frame)),
        "avg_accuracy_over_policies": float(frame["avg_accuracy"].mean()) if "avg_accuracy" in frame else np.nan,
        "best_policy_by_accuracy": None,
        "best_avg_accuracy": np.nan,
    }
    if "avg_accuracy" in frame and not frame["avg_accuracy"].isna().all():
        best_idx = frame["avg_accuracy"].idxmax()
        summary["best_policy_by_accuracy"] = str(frame.loc[best_idx, "policy"])
        summary["best_avg_accuracy"] = float(frame.loc[best_idx, "avg_accuracy"])
    return summary

def _write_plot_context(
    output_dir,
    profile_name,
    profile_spec,
    target_rate_hz,
    t0_deadline,
    batch_size,
    dynamic_timeslot_size,
    max_dynamic_timeslot_count=None,
    accuracy_estimator_mode=None,
    stein_sigma=None,
    stein_N=None,
):
    plot_context = {
        "experiment": "online",
        "platform": "jetson",
        "model": MODEL_TAG,
        "dataset": "sst2",
        "codec_name": profile_spec["codec_name"],
        "codec_profile": profile_name,
        "accuracy_estimator_mode": accuracy_estimator_mode,
        "target_rate_hz": float(target_rate_hz),
        "target_time_sec": float(t0_deadline),
        "batch_size": int(batch_size),
        "dynamic_timeslot_size": int(dynamic_timeslot_size),
        "dynamic_timeslot_mode": "tumbling_window_count",
        "max_dynamic_timeslot_count": (
            int(max_dynamic_timeslot_count) if max_dynamic_timeslot_count is not None else None
        ),
    }
    if accuracy_estimator_mode == "stein_estimator":
        plot_context["stein_sigma"] = float(stein_sigma) if stein_sigma is not None else None
        plot_context["stein_N"] = int(stein_N) if stein_N is not None else None
    with open(os.path.join(output_dir, "plot_context.json"), "w", encoding="utf-8") as handle:
        json.dump(plot_context, handle, ensure_ascii=False, indent=2)
    return plot_context
