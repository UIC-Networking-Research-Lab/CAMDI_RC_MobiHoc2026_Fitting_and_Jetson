#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import sys
from glob import glob
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


DEFAULT_SWEEP_OUTPUT_DIR = os.path.join("outputs")


def _log(message: str):
    print("[multi-plot] {}".format(message), file=sys.stdout, flush=True)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _load_plot_context(trial_dir: str, plot_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if plot_context is not None:
        return dict(plot_context)
    path = os.path.join(trial_dir, "plot_context.json")
    return _load_json(path) if os.path.exists(path) else {}


def _candidate_trial_dirs(root_dir: str) -> List[str]:
    matches = []
    for current_root, _, files in os.walk(root_dir):
        if any(name.startswith("policy_summary") and name.endswith(".csv") for name in files):
            matches.append(current_root)
    return sorted(matches)


def _load_csv_records(csv_paths: List[str], sort_column: Optional[str] = None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in sorted(csv_paths):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        if sort_column and sort_column in frame.columns:
            frame = frame.sort_values(sort_column).reset_index(drop=True)
        records.append({"path": path, "frame": frame})
    return records


def _load_timeseries_records(trial_dir: str) -> List[Dict[str, Any]]:
    csv_paths = sorted(glob(os.path.join(trial_dir, "timeseries_*.csv")))
    if not csv_paths:
        csv_paths = sorted(glob(os.path.join(trial_dir, "ts_*.csv")))
    return _load_csv_records(csv_paths, sort_column="t")


def _load_window_summary_records(trial_dir: str) -> List[Dict[str, Any]]:
    return _load_csv_records(sorted(glob(os.path.join(trial_dir, "window_summary_*.csv"))), sort_column="window_id")


def _task_ids_from_context(ctx: Dict[str, Any]) -> List[int]:
    task_items = ctx.get("tasks", [])
    task_ids = []
    for item in task_items:
        try:
            task_ids.append(int(item["task_id"]))
        except Exception:
            continue
    return sorted(set(task_ids))


def _task_rate_from_context(ctx: Dict[str, Any], task_id: int) -> float:
    for item in ctx.get("tasks", []):
        try:
            if int(item.get("task_id")) == int(task_id) and item.get("target_rate_hz") is not None:
                return float(item["target_rate_hz"])
        except Exception:
            continue
    return np.nan


def _coalesce_series(frame: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
    for column in candidates:
        if column in frame.columns:
            return frame[column]
    return None


def _copy_series(dst: pd.DataFrame, dst_column: str, frame: pd.DataFrame, candidates: List[str]):
    series = _coalesce_series(frame, candidates)
    if series is not None:
        dst[dst_column] = series


def _extract_algorithm_name(frame: pd.DataFrame, fallback: str = "", prefer_policy: bool = False) -> str:
    columns = ("policy", "algorithm") if prefer_policy else ("algorithm", "policy")
    for column in columns:
        if column in frame.columns and not frame[column].dropna().empty:
            return str(frame[column].dropna().iloc[0])
    return str(fallback)


def _series_scope(frame: pd.DataFrame, path: str, default_scope: str = "avg") -> str:
    if "series_scope" in frame.columns and not frame["series_scope"].dropna().empty:
        return str(frame["series_scope"].dropna().iloc[0]).strip()
    stem = os.path.splitext(os.path.basename(path))[0]
    task_match = re.search(r"_task(\d+)$", stem)
    if task_match:
        return "task{}".format(task_match.group(1))
    if stem.endswith("_avg"):
        return "avg"
    return str(default_scope)


def _task_scope_id(scope: str) -> Optional[int]:
    match = re.fullmatch(r"task(\d+)", str(scope).strip())
    if not match:
        return None
    return int(match.group(1))


def _record_priority_for_scope(path: str, scope: str) -> int:
    stem = os.path.splitext(os.path.basename(path))[0]
    if str(scope) == "avg" and stem.endswith("_avg"):
        return 1
    return 0


def _sort_frames_by_algorithm(frames: List[pd.DataFrame]) -> List[pd.DataFrame]:
    return sorted(frames, key=lambda frame: str(frame["algorithm"].iloc[0]) if "algorithm" in frame.columns and not frame.empty else "")


def _normalize_aggregate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = pd.DataFrame()
    if "t" in frame.columns:
        normalized["t"] = frame["t"].astype(float)
    else:
        normalized["t"] = np.arange(len(frame), dtype=float)
    if "completion_total" in frame.columns:
        normalized["completion_total"] = frame["completion_total"].astype(float)
    normalized["algorithm"] = _extract_algorithm_name(frame)
    _copy_series(normalized, "accuracy_true", frame, ["accuracy_true", "weighted_accuracy_true"])
    _copy_series(normalized, "accuracy_fit", frame, ["accuracy_fit", "weighted_accuracy_fit"])
    _copy_series(normalized, "delay_actual_sec", frame, ["delay_actual_sec", "avg_delay_actual_sec"])
    _copy_series(normalized, "delay_pred_sec", frame, ["delay_pred_sec", "avg_delay_pred_sec"])
    _copy_series(normalized, "delay_ratio_actual", frame, ["delay_ratio_actual", "avg_delay_ratio_actual"])
    _copy_series(normalized, "delay_ratio_pred", frame, ["delay_ratio_pred", "avg_delay_ratio_pred"])
    _copy_series(normalized, "target_rate_hz", frame, ["target_rate_hz"])
    _copy_series(normalized, "target_time_sec", frame, ["target_time_sec"])
    return normalized


def _normalize_task_frame(frame: pd.DataFrame, ctx: Dict[str, Any], task_id: int) -> pd.DataFrame:
    normalized = pd.DataFrame()
    if "t" in frame.columns:
        normalized["t"] = frame["t"].astype(float)
    else:
        normalized["t"] = np.arange(len(frame), dtype=float)
    completion_col = "completion_task{}".format(int(task_id))
    if completion_col in frame.columns:
        normalized[completion_col] = frame[completion_col].astype(float)
    normalized["algorithm"] = _extract_algorithm_name(frame)
    _copy_series(normalized, "accuracy_true", frame, ["accuracy_true_task{}".format(task_id), "accuracy_true"])
    _copy_series(normalized, "accuracy_fit", frame, ["accuracy_fit_task{}".format(task_id), "accuracy_fit"])
    _copy_series(normalized, "delay_actual_sec", frame, ["delay_actual_sec_task{}".format(task_id), "delay_actual_sec"])
    _copy_series(normalized, "delay_pred_sec", frame, ["delay_pred_sec_task{}".format(task_id), "delay_pred_sec"])
    _copy_series(normalized, "delay_ratio_actual", frame, ["delay_ratio_actual_task{}".format(task_id), "delay_ratio_actual"])
    _copy_series(normalized, "delay_ratio_pred", frame, ["delay_ratio_pred_task{}".format(task_id), "delay_ratio_pred"])
    _copy_series(normalized, "lambda_value", frame, ["lambda_task{}".format(task_id), "lambda_value"])
    _copy_series(normalized, "service_delay_sec", frame, ["service_delay_sec_task{}".format(task_id), "service_delay_sec"])
    _copy_series(normalized, "communication_delay_sec", frame, ["communication_delay_sec_task{}".format(task_id), "communication_delay_sec"])
    _copy_series(normalized, "miscellaneous_delay_sec", frame, ["miscellaneous_delay_sec_task{}".format(task_id), "miscellaneous_delay_sec"])
    target_rate = _coalesce_series(frame, ["target_rate_hz_task{}".format(task_id), "target_rate_hz"])
    if target_rate is not None:
        normalized["target_rate_hz"] = target_rate.astype(float)
    else:
        rate_value = _task_rate_from_context(ctx, task_id)
        if not pd.isna(rate_value):
            normalized["target_rate_hz"] = np.full(len(normalized), float(rate_value), dtype=float)
    if "target_rate_hz" in normalized.columns:
        normalized["target_time_sec"] = np.divide(
            1.0,
            normalized["target_rate_hz"].astype(float).to_numpy(),
            out=np.full(len(normalized), np.nan, dtype=float),
            where=normalized["target_rate_hz"].astype(float).to_numpy() > 0.0,
        )
    eta_task_prefix = "eta_task{}_".format(int(task_id))
    eta_columns = [column for column in frame.columns if column.startswith(eta_task_prefix)]
    if eta_columns:
        for column in sorted(eta_columns, key=lambda name: int(name.split("_")[-1])):
            link_idx = int(column.split("_")[-1])
            normalized["requested_eta_{}".format(link_idx)] = frame[column].astype(float)
    else:
        requested_cols = [column for column in frame.columns if column.startswith("requested_eta_")]
        for column in sorted(requested_cols, key=lambda name: int(name.split("_")[-1])):
            normalized[column] = frame[column].astype(float)
    return normalized


def _normalize_window_frame(frame: pd.DataFrame, algorithm_fallback: str = "") -> pd.DataFrame:
    normalized = pd.DataFrame()
    if "window_id" not in frame.columns:
        return normalized
    normalized["window_id"] = frame["window_id"].astype(float)
    normalized["algorithm"] = _extract_algorithm_name(frame, fallback=algorithm_fallback, prefer_policy=True)
    _copy_series(normalized, "avg_accuracy", frame, ["avg_accuracy"])
    _copy_series(normalized, "avg_delay_sec", frame, ["avg_delay_sec"])
    _copy_series(normalized, "target_time_sec", frame, ["target_time_sec"])
    _copy_series(normalized, "lambda_input", frame, ["lambda_input"])
    for column in sorted([name for name in frame.columns if name.startswith("requested_eta_")], key=lambda name: int(name.split("_")[-1])):
        normalized[column] = frame[column].astype(float)
    return normalized


def _window_task_ids_from_context_or_frames(ctx: Dict[str, Any], window_frames_by_task: Dict[int, List[pd.DataFrame]]) -> List[int]:
    task_ids = _task_ids_from_context(ctx)
    if task_ids:
        return task_ids
    return sorted(int(task_id) for task_id in window_frames_by_task.keys())


def _map_offline_records(
    timeseries_records: List[Dict[str, Any]],
    window_records: List[Dict[str, Any]],
    ctx: Dict[str, Any],
) -> Dict[str, Any]:
    task_ids = _task_ids_from_context(ctx)
    aggregate_frames: List[pd.DataFrame] = []
    task_frames_by_task: Dict[int, List[pd.DataFrame]] = {task_id: [] for task_id in task_ids}
    for record in timeseries_records:
        frame = record["frame"]
        aggregate_frames.append(_normalize_aggregate_frame(frame))
        for task_id in task_ids:
            task_frame = _normalize_task_frame(frame, ctx, task_id)
            if not task_frame.empty and any(
                column in task_frame.columns
                for column in ("accuracy_true", "delay_actual_sec", "lambda_value")
            ):
                task_frames_by_task.setdefault(task_id, []).append(task_frame)
    return {
        "aggregate_frames": _sort_frames_by_algorithm(aggregate_frames),
        "task_frames_by_task": {
            task_id: _sort_frames_by_algorithm(frames)
            for task_id, frames in task_frames_by_task.items()
            if frames
        },
        "window_frames": [],
    }


def _map_online_records(
    timeseries_records: List[Dict[str, Any]],
    window_records: List[Dict[str, Any]],
    ctx: Dict[str, Any],
) -> Dict[str, Any]:
    aggregate_frames_by_algorithm: Dict[str, Dict[str, Any]] = {}
    task_frames_by_task: Dict[int, Dict[str, Dict[str, Any]]] = {}
    window_task_frames_by_task: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for record in timeseries_records:
        frame = record["frame"]
        scope = _series_scope(frame, record["path"], default_scope="avg")
        algorithm = _extract_algorithm_name(frame, fallback=os.path.basename(record["path"]))
        task_id = _task_scope_id(scope)
        if task_id is None:
            normalized = _normalize_aggregate_frame(frame)
            priority = _record_priority_for_scope(record["path"], scope)
            current = aggregate_frames_by_algorithm.get(algorithm)
            if current is None or priority < current["priority"]:
                aggregate_frames_by_algorithm[algorithm] = {"priority": priority, "frame": normalized}
            continue
        normalized_task = _normalize_task_frame(frame, ctx, task_id)
        if normalized_task.empty:
            continue
        bucket = task_frames_by_task.setdefault(task_id, {})
        current = bucket.get(algorithm)
        priority = _record_priority_for_scope(record["path"], scope)
        if current is None or priority < current["priority"]:
            bucket[algorithm] = {"priority": priority, "frame": normalized_task}

    window_frames_by_algorithm: Dict[str, Dict[str, Any]] = {}
    for record in window_records:
        frame = record["frame"]
        scope = _series_scope(frame, record["path"], default_scope="avg")
        algorithm = _extract_algorithm_name(frame, fallback=os.path.basename(record["path"]))
        task_id = _task_scope_id(scope)
        normalized = _normalize_window_frame(frame, algorithm_fallback=algorithm)
        if normalized.empty:
            continue
        priority = _record_priority_for_scope(record["path"], scope)
        if task_id is not None:
            bucket = window_task_frames_by_task.setdefault(task_id, {})
            current = bucket.get(algorithm)
            if current is None or priority < current["priority"]:
                bucket[algorithm] = {"priority": priority, "frame": normalized}
            continue
        priority = _record_priority_for_scope(record["path"], scope)
        current = window_frames_by_algorithm.get(algorithm)
        if current is None or priority < current["priority"]:
            window_frames_by_algorithm[algorithm] = {"priority": priority, "frame": normalized}

    return {
        "aggregate_frames": _sort_frames_by_algorithm([item["frame"] for item in aggregate_frames_by_algorithm.values()]),
        "task_frames_by_task": {
            task_id: _sort_frames_by_algorithm([item["frame"] for item in frames_by_algorithm.values()])
            for task_id, frames_by_algorithm in task_frames_by_task.items()
            if frames_by_algorithm
        },
        "window_frames": _sort_frames_by_algorithm([item["frame"] for item in window_frames_by_algorithm.values()]),
        "window_frames_by_task": {
            task_id: _sort_frames_by_algorithm([item["frame"] for item in frames_by_algorithm.values()])
            for task_id, frames_by_algorithm in window_task_frames_by_task.items()
            if frames_by_algorithm
        },
    }


def _build_semantic_trial(
    trial_dir: str,
    ctx: Dict[str, Any],
    timeseries_records: List[Dict[str, Any]],
    window_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    experiment = str(ctx.get("experiment", "")).strip().lower()
    if experiment == "online":
        return _map_online_records(timeseries_records, window_records, ctx)
    if experiment == "offline":
        return _map_offline_records(timeseries_records, window_records, ctx)
    if any("series_scope" in record["frame"].columns for record in timeseries_records) or window_records:
        return _map_online_records(timeseries_records, window_records, ctx)
    return _map_offline_records(timeseries_records, window_records, ctx)


def _task_ids_from_context_or_task_frames(ctx: Dict[str, Any], task_frames_by_task: Dict[int, List[pd.DataFrame]]) -> List[int]:
    task_ids = _task_ids_from_context(ctx)
    if task_ids:
        return task_ids
    return sorted(int(task_id) for task_id in task_frames_by_task.keys())


def _task_x_values(frame: pd.DataFrame, task_id: int) -> np.ndarray:
    completion_col = "completion_task{}".format(int(task_id))
    if completion_col in frame.columns:
        return frame[completion_col].astype(float).to_numpy()
    if "t" in frame.columns:
        return frame["t"].astype(float).to_numpy()
    return np.arange(len(frame), dtype=float)


def _clear_plot_outputs(trial_dir: str, ctx: Dict[str, Any]):
    tagged_stems = [
        "accuracy_over_time",
        "delay_over_invR_over_time",
        "delay_over_time",
        "rate_over_time",
        "tradeoff_accuracy_vs_delay",
        "lambda_over_time",
        "eta_over_time",
        "delay_breakdown_distribution",
    ]
    untagged_names = [
        "accuracy_over_timeslot.png",
        "delay_over_timeslot.png",
        "rate_over_timeslot.png",
        "delay_over_invR_over_timeslot.png",
        "lambda_over_timeslot.png",
        "eta_over_timeslot.png",
        "weighted_accuracy_over_time.png",
    ]
    for stem in tagged_stems:
        path = _output_path(trial_dir, stem, ctx)
        if os.path.exists(path):
            os.remove(path)
        tagged_pattern = _output_path(trial_dir, "{}_task*".format(stem), ctx)
        for extra_path in glob(tagged_pattern):
            if os.path.exists(extra_path):
                os.remove(extra_path)
    for name in untagged_names:
        path = os.path.join(trial_dir, name)
        if os.path.exists(path):
            os.remove(path)
        stem = os.path.splitext(name)[0]
        for extra_path in glob(os.path.join(trial_dir, "{}_task*.png".format(stem))):
            if os.path.exists(extra_path):
                os.remove(extra_path)


def _x_values(frame: pd.DataFrame, task_id: Optional[int] = None) -> np.ndarray:
    if task_id is not None:
        task_col = "completion_task{}".format(int(task_id))
        if task_col in frame.columns:
            return frame[task_col].astype(float).to_numpy()
    if "t" in frame.columns:
        return frame["t"].astype(float).to_numpy()
    if "completion_total" in frame.columns:
        return frame["completion_total"].astype(float).to_numpy()
    return np.arange(len(frame), dtype=float)


def _frames_have_columns(frames: List[pd.DataFrame], columns: List[str]) -> bool:
    if not frames:
        return False
    return all(all(column in frame.columns for column in columns) for frame in frames)


def _subtitle_lines(ctx: Dict[str, Any]) -> List[str]:
    experiment = ctx.get("experiment", "offline")
    platform = ctx.get("platform", "jetson")
    model = ctx.get("model", "mixed")
    dataset = ctx.get("dataset", "mixed")
    codec = ctx.get("codec_name", "unknown")
    batch_size = ctx.get("batch_size", "unknown")
    channel_limit = ctx.get("channel_limit_mbps", "unknown")
    task_items = ctx.get("tasks", [])
    line1 = "experiment={experiment} | platform={platform} | model={model} | dataset={dataset} | codec={codec}".format(
        experiment=experiment,
        platform=platform,
        model=model,
        dataset=dataset,
        codec=codec,
    )
    line2 = "channel_limit={channel_limit} Mbps | batch_size={batch_size}".format(
        channel_limit=channel_limit,
        batch_size=batch_size,
    )
    lines = [line1, line2]
    if not task_items:
        lines.append("tasks=unknown")
        return lines
    for item in task_items:
        rate = item.get("target_rate_hz", "?")
        try:
            target_time = "{:.6g}".format(1.0 / float(rate))
        except Exception:
            target_time = "?"
        lines.append(
            "task{tid}: model={model} | dataset={dataset} | batch_size={bs} | w_k={w} | target_rate={r} Hz | target_time={tt} s | scenario_tx={tx}".format(
                tid=item.get("task_id", "?"),
                model=item.get("model", "?"),
                dataset=item.get("dataset", "?"),
                bs=item.get("batch_size", "?"),
                w=item.get("w_k", "?"),
                r=rate,
                tt=target_time,
                tx=item.get("scenario_tx_limit_mbps", "?"),
            )
        )
    return lines


def _apply_title(fig, title: str, ctx: Dict[str, Any]):
    experiment = str(ctx.get("experiment", "offline")).strip().lower()
    if experiment == "offline":
        title = "{} [Offline]".format(title)
    elif experiment == "online":
        title = "{} [Online]".format(title)
    fig.suptitle(title, fontsize=13, y=0.98)
    fig.text(0.5, 0.94, "\n".join(_subtitle_lines(ctx)), ha="center", va="top", fontsize=9)


def _legend_outside(ax):
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        frameon=False,
    )


def _figure_legend_outside(fig, axes, ncol: int = 1):
    if isinstance(axes, np.ndarray):
        axes = axes.ravel().tolist()
    elif not isinstance(axes, (list, tuple)):
        axes = [axes]
    handles = []
    labels = []
    for ax in axes:
        current_handles, current_labels = ax.get_legend_handles_labels()
        for handle, label in zip(current_handles, current_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if not handles:
        return
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.84, 0.5),
        borderaxespad=0.0,
        frameon=False,
        ncol=ncol,
    )


def _output_path(trial_dir: str, stem: str, ctx: Dict[str, Any]) -> str:
    tag = str(ctx.get("channel_speed_tag", "")).strip()
    if tag:
        return os.path.join(trial_dir, "{}_{}.png".format(stem, tag))
    return os.path.join(trial_dir, "{}.png".format(stem))


def _task_output_stem(stem: str, task_id: int) -> str:
    return "{}_task{}".format(stem, int(task_id))


def _task_plot_title(title: str, task_id: int) -> str:
    return "{} [Task {}]".format(title, int(task_id))


def _untagged_output_path(trial_dir: str, stem: str) -> str:
    return os.path.join(trial_dir, "{}.png".format(stem))


def _plot_accuracy_over_time(
    trial_dir: str,
    frames: List[pd.DataFrame],
    ctx: Dict[str, Any],
    stem: str = "accuracy_over_time",
    title: str = "Accuracy Over Time",
    xlabel: str = "t",
    x_getter=None,
):
    if not _frames_have_columns(frames, ["accuracy_true"]):
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for frame in frames:
        algo = str(frame["algorithm"].iloc[0])
        x = x_getter(frame) if x_getter is not None else _x_values(frame)
        ax.plot(x, frame["accuracy_true"].astype(float).to_numpy(), label=algo)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy (true)")
    ax.set_ylim(bottom=0.0)
    ymax = max(1.0, max(float(frame["accuracy_true"].astype(float).max()) for frame in frames) + 0.02)
    ax.set_ylim(top=min(ymax, 1.05))
    _legend_outside(ax)
    _apply_title(fig, title, ctx)
    fig.tight_layout(rect=[0, 0, 0.82, 0.9])
    fig.savefig(_output_path(trial_dir, stem, ctx), dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_delay_ratio(
    trial_dir: str,
    frames: List[pd.DataFrame],
    ctx: Dict[str, Any],
    stem: str = "delay_over_invR_over_time",
    title: str = "Delay Ratio Over Time",
    xlabel: str = "t",
    x_getter=None,
):
    fig, ax = plt.subplots(figsize=(10, 5))
    for frame in frames:
        algo = str(frame["algorithm"].iloc[0])
        if "delay_ratio_actual" in frame.columns:
            y = frame["delay_ratio_actual"].astype(float).to_numpy()
        elif "delay_actual_sec" in frame.columns and "target_time_sec" in frame.columns:
            delay = frame["delay_actual_sec"].astype(float).to_numpy()
            target = frame["target_time_sec"].astype(float).to_numpy()
            y = np.divide(delay, target, out=np.full_like(delay, np.nan, dtype=float), where=target > 0.0)
        else:
            y = frame["avg_delay_ratio_actual"].astype(float).to_numpy()
        x = x_getter(frame) if x_getter is not None else _x_values(frame)
        ax.plot(x, y, label=algo)
    ax.axhline(1.0, linestyle="--", color="black", linewidth=1.5, label="deadline ratio=1")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("delay / (1/R)")
    _legend_outside(ax)
    _apply_title(fig, title, ctx)
    fig.tight_layout(rect=[0, 0, 0.82, 0.9])
    fig.savefig(_output_path(trial_dir, stem, ctx), dpi=160, bbox_inches="tight")
    plt.close(fig)


def _task_ids_from_context_or_frame(ctx: Dict[str, Any], frames: List[pd.DataFrame]) -> List[int]:
    if "tasks" in ctx:
        return [int(item["task_id"]) for item in ctx["tasks"]]
    task_ids = set()
    for col in frames[0].columns:
        if col.startswith("lambda_task"):
            task_ids.add(int(col.split("lambda_task", 1)[1]))
        elif col.startswith("delay_actual_sec_task"):
            task_ids.add(int(col.split("delay_actual_sec_task", 1)[1]))
        elif col.startswith("target_rate_hz_task"):
            task_ids.add(int(col.split("target_rate_hz_task", 1)[1]))
    return sorted(task_ids)


def _plot_delay_over_time(
    trial_dir: str,
    frames: List[pd.DataFrame],
    ctx: Dict[str, Any],
    stem: str = "delay_over_time",
    title: str = "Delay Over Time",
    xlabel: str = "t",
    x_getter=None,
):
    if not _frames_have_columns(frames, ["delay_actual_sec"]):
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    deadline_drawn = False
    for frame in frames:
        algo = str(frame["algorithm"].iloc[0])
        x = x_getter(frame) if x_getter is not None else _x_values(frame)
        delay = frame["delay_actual_sec"].astype(float).to_numpy()
        ax.plot(x, delay, label=algo)
        if "target_time_sec" in frame.columns:
            deadline = frame["target_time_sec"].astype(float).to_numpy()
            if np.isfinite(deadline).any():
                ax.plot(
                    x,
                    deadline,
                    linestyle="--",
                    color="black",
                    linewidth=1.5,
                    label=("deadline=1/R" if not deadline_drawn else None),
                )
                deadline_drawn = True
    ax.set_ylabel("Delay (sec)")
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(xlabel)
    _legend_outside(ax)
    _apply_title(fig, title, ctx)
    fig.tight_layout(rect=[0, 0, 0.82, 0.9])
    fig.savefig(_output_path(trial_dir, stem, ctx), dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_rate_over_time(
    trial_dir: str,
    frames: List[pd.DataFrame],
    ctx: Dict[str, Any],
    stem: str = "rate_over_time",
    title: str = "Actual Rate Over Time",
    xlabel: str = "t",
    x_getter=None,
):
    if not _frames_have_columns(frames, ["delay_actual_sec"]):
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    target_drawn = False
    for frame in frames:
        algo = str(frame["algorithm"].iloc[0])
        x = x_getter(frame) if x_getter is not None else _x_values(frame)
        delay_actual = frame["delay_actual_sec"].astype(float).to_numpy()
        actual_rate = np.divide(1.0, delay_actual, out=np.full_like(delay_actual, np.nan, dtype=float), where=delay_actual > 0.0)
        ax.plot(x, actual_rate, label=algo)
        if "target_rate_hz" in frame.columns:
            target_rate = frame["target_rate_hz"].astype(float).to_numpy()
            ax.plot(
                x,
                target_rate,
                linestyle="--",
                color="black",
                linewidth=1.5,
                label=("target_rate_hz" if not target_drawn else None),
            )
            target_drawn = True
    ax.set_ylabel("Rate (Hz)")
    ax.set_xlabel(xlabel)
    _legend_outside(ax)
    _apply_title(fig, title, ctx)
    fig.tight_layout(rect=[0, 0, 0.82, 0.9])
    fig.savefig(_output_path(trial_dir, stem, ctx), dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_tradeoff(
    trial_dir: str,
    frames: List[pd.DataFrame],
    ctx: Dict[str, Any],
    stem: str = "tradeoff_accuracy_vs_delay",
    title: str = "Accuracy-Delay Tradeoff",
):
    fig, ax = plt.subplots(figsize=(7, 5))
    for frame in frames:
        algo = str(frame["algorithm"].iloc[0])
        if "delay_ratio_actual" in frame.columns:
            x = float(np.nanmean(frame["delay_ratio_actual"].astype(float).to_numpy()))
        else:
            x = float(np.nanmean(frame["avg_delay_ratio_actual"].astype(float).to_numpy()))
        if "accuracy_true" in frame.columns:
            y = float(np.nanmean(frame["accuracy_true"].astype(float).to_numpy()))
        else:
            y = float(np.nanmean(frame["weighted_accuracy_true"].astype(float).to_numpy()))
        ax.scatter(x, y, label=algo)
    ax.set_xlabel("Average delay / (1/R)")
    ax.set_ylabel("Average true accuracy")
    ax.set_ylim(bottom=0.0)
    _legend_outside(ax)
    _apply_title(fig, title, ctx)
    fig.tight_layout(rect=[0, 0, 0.82, 0.9])
    fig.savefig(_output_path(trial_dir, stem, ctx), dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_task_level_main_plots(trial_dir: str, task_frames_by_task: Dict[int, List[pd.DataFrame]], ctx: Dict[str, Any]):
    for task_id in _task_ids_from_context_or_task_frames(ctx, task_frames_by_task):
        frames = task_frames_by_task.get(task_id, [])
        if not frames:
            continue
        x_getter = lambda frame, tid=task_id: _task_x_values(frame, tid)
        xlabel = "completed task{} samples".format(task_id)
        _plot_accuracy_over_time(
            trial_dir,
            frames,
            ctx,
            stem=_task_output_stem("accuracy_over_time", task_id),
            title=_task_plot_title("Accuracy Over Time", task_id),
            xlabel=xlabel,
            x_getter=x_getter,
        )
        _plot_delay_over_time(
            trial_dir,
            frames,
            ctx,
            stem=_task_output_stem("delay_over_time", task_id),
            title=_task_plot_title("Delay Over Time", task_id),
            xlabel=xlabel,
            x_getter=x_getter,
        )
        _plot_rate_over_time(
            trial_dir,
            frames,
            ctx,
            stem=_task_output_stem("rate_over_time", task_id),
            title=_task_plot_title("Actual Rate Over Time", task_id),
            xlabel=xlabel,
            x_getter=x_getter,
        )
        _plot_delay_ratio(
            trial_dir,
            frames,
            ctx,
            stem=_task_output_stem("delay_over_invR_over_time", task_id),
            title=_task_plot_title("Delay Ratio Over Time", task_id),
            xlabel=xlabel,
            x_getter=x_getter,
        )
        _plot_tradeoff(
            trial_dir,
            frames,
            ctx,
            stem=_task_output_stem("tradeoff_accuracy_vs_delay", task_id),
            title=_task_plot_title("Accuracy-Delay Tradeoff", task_id),
        )


def _plot_lambda(
    trial_dir: str,
    aggregate_frames: List[pd.DataFrame],
    task_frames_by_task: Dict[int, List[pd.DataFrame]],
    ctx: Dict[str, Any],
):
    task_ids = _task_ids_from_context_or_task_frames(ctx, task_frames_by_task)
    if task_ids and any(task_frames_by_task.get(task_id) for task_id in task_ids):
        fig, axes = plt.subplots(len(task_ids), 1, figsize=(10, max(4, 3.5 * len(task_ids))), sharex=False)
        if len(task_ids) == 1:
            axes = [axes]
        for ax, task_id in zip(axes, task_ids):
            frames = task_frames_by_task.get(task_id, [])
            for frame in frames:
                if "lambda_value" not in frame.columns:
                    continue
                algo = str(frame["algorithm"].iloc[0])
                ax.plot(_task_x_values(frame, task_id), frame["lambda_value"].astype(float).to_numpy(), label=algo)
            ax.set_ylabel("lambda task{}".format(task_id))
            ax.set_xlabel("completed task{} samples".format(task_id))
        _figure_legend_outside(fig, axes)
        _apply_title(fig, "Per-Task Dual Variables Over Time", ctx)
        fig.tight_layout(rect=[0, 0, 0.82, 0.9])
        fig.savefig(_output_path(trial_dir, "lambda_over_time", ctx), dpi=160, bbox_inches="tight")
        plt.close(fig)
        return
    if _frames_have_columns(aggregate_frames, ["lambda_value"]):
        fig, ax = plt.subplots(figsize=(10, 5))
        for frame in aggregate_frames:
            algo = str(frame["algorithm"].iloc[0])
            ax.plot(_x_values(frame), frame["lambda_value"].astype(float).to_numpy(), label=algo)
        ax.set_ylabel("lambda")
        ax.set_xlabel("t")
        _legend_outside(ax)
        _apply_title(fig, "Dual Variable Over Time", ctx)
        fig.tight_layout(rect=[0, 0, 0.82, 0.9])
        fig.savefig(_output_path(trial_dir, "lambda_over_time", ctx), dpi=160, bbox_inches="tight")
        plt.close(fig)
        return
def _plot_eta(
    trial_dir: str,
    aggregate_frames: List[pd.DataFrame],
    task_frames_by_task: Dict[int, List[pd.DataFrame]],
    ctx: Dict[str, Any],
):
    task_ids = _task_ids_from_context_or_task_frames(ctx, task_frames_by_task)
    if task_ids and any(task_frames_by_task.get(task_id) for task_id in task_ids):
        all_eta_cols = set()
        for frames in task_frames_by_task.values():
            for frame in frames:
                for column in frame.columns:
                    if column.startswith("requested_eta_"):
                        all_eta_cols.add(column)
        if not all_eta_cols:
            return
        eta_cols = sorted(all_eta_cols, key=lambda name: int(name.split("_")[-1]))
        fig, axes = plt.subplots(len(task_ids) * len(eta_cols), 1, figsize=(10, max(4, 2.6 * len(task_ids) * len(eta_cols))), sharex=False)
        if not isinstance(axes, np.ndarray):
            axes = np.asarray([axes], dtype=object)
        axes = axes.ravel().tolist()
        axis_idx = 0
        for task_id in task_ids:
            frames = task_frames_by_task.get(task_id, [])
            for eta_col in eta_cols:
                ax = axes[axis_idx]
                axis_idx += 1
                for frame in frames:
                    if eta_col not in frame.columns:
                        continue
                    algo = str(frame["algorithm"].iloc[0])
                    ax.plot(_task_x_values(frame, task_id), frame[eta_col].astype(float).to_numpy(), label=algo)
                ax.set_ylabel("task{} {}".format(task_id, eta_col.replace("requested_", "")))
                ax.set_xlabel("completed task{} samples".format(task_id))
        _figure_legend_outside(fig, axes)
        _apply_title(fig, "Per-Task Compression Over Time", ctx)
        fig.tight_layout(rect=[0, 0, 0.82, 0.9])
        fig.savefig(_output_path(trial_dir, "eta_over_time", ctx), dpi=160, bbox_inches="tight")
        plt.close(fig)
        return
    if not aggregate_frames:
        return
    eta_cols = [col for col in aggregate_frames[0].columns if col.startswith("requested_eta_")]
    if eta_cols:
        eta_cols = sorted(eta_cols, key=lambda x: int(x.split("_")[-1]))
        fig, axes = plt.subplots(len(eta_cols), 1, figsize=(10, max(4, 2.8 * len(eta_cols))), sharex=True)
        if len(eta_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, eta_cols):
            for frame in aggregate_frames:
                algo = str(frame["algorithm"].iloc[0])
                ax.plot(_x_values(frame), frame[col].astype(float).to_numpy(), label=algo)
            ax.set_ylabel(col.replace("requested_", ""))
        axes[-1].set_xlabel("t")
        _figure_legend_outside(fig, axes)
        _apply_title(fig, "Eta Over Time", ctx)
        fig.tight_layout(rect=[0, 0, 0.82, 0.9])
        fig.savefig(_output_path(trial_dir, "eta_over_time", ctx), dpi=160, bbox_inches="tight")
        plt.close(fig)
        return


def _plot_accuracy_over_timeslot(
    trial_dir: str,
    frames: List[pd.DataFrame],
    ctx: Dict[str, Any],
    stem: str = "accuracy_over_timeslot",
    title: str = "Accuracy Over Timeslot",
):
    if not _frames_have_columns(frames, ["window_id", "avg_accuracy"]):
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for frame in frames:
        algo = str(frame["algorithm"].iloc[0])
        ax.plot(frame["window_id"].astype(float).to_numpy(), frame["avg_accuracy"].astype(float).to_numpy(), marker="o", label=algo)
    ax.set_xlabel("timeslot")
    ax.set_ylabel("Accuracy (timeslot avg)")
    ax.set_ylim(bottom=0.0)
    _legend_outside(ax)
    _apply_title(fig, title, ctx)
    fig.tight_layout(rect=[0, 0, 0.82, 0.9])
    fig.savefig(_untagged_output_path(trial_dir, stem), dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_delay_over_timeslot(
    trial_dir: str,
    frames: List[pd.DataFrame],
    ctx: Dict[str, Any],
    stem: str = "delay_over_timeslot",
    title: str = "Delay Over Timeslot",
):
    if not _frames_have_columns(frames, ["window_id", "avg_delay_sec"]):
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    deadline_drawn = False
    for frame in frames:
        algo = str(frame["algorithm"].iloc[0])
        window_ids = frame["window_id"].astype(float).to_numpy()
        delay = frame["avg_delay_sec"].astype(float).to_numpy()
        ax.plot(window_ids, delay, marker="o", label=algo)
        if "target_time_sec" in frame.columns:
            deadline = frame["target_time_sec"].astype(float).to_numpy()
            if np.isfinite(deadline).any():
                ax.plot(
                    window_ids,
                    deadline,
                    linestyle="--",
                    color="black",
                    linewidth=1.5,
                    label=("deadline=1/R" if not deadline_drawn else None),
                )
                deadline_drawn = True
    ax.set_xlabel("timeslot")
    ax.set_ylabel("Delay (sec)")
    ax.set_ylim(bottom=0.0)
    _legend_outside(ax)
    _apply_title(fig, title, ctx)
    fig.tight_layout(rect=[0, 0, 0.82, 0.9])
    fig.savefig(_untagged_output_path(trial_dir, stem), dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_rate_over_timeslot(
    trial_dir: str,
    frames: List[pd.DataFrame],
    ctx: Dict[str, Any],
    stem: str = "rate_over_timeslot",
    title: str = "Actual Rate Over Timeslot",
):
    if not _frames_have_columns(frames, ["window_id", "avg_delay_sec"]):
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    target_drawn = False
    for frame in frames:
        algo = str(frame["algorithm"].iloc[0])
        window_ids = frame["window_id"].astype(float).to_numpy()
        avg_delay = frame["avg_delay_sec"].astype(float).to_numpy()
        actual_rate = np.divide(1.0, avg_delay, out=np.full_like(avg_delay, np.nan, dtype=float), where=avg_delay > 0.0)
        ax.plot(window_ids, actual_rate, marker="o", label=algo)
        if "target_time_sec" in frame.columns:
            target_time = frame["target_time_sec"].astype(float).to_numpy()
            target_rate = np.divide(1.0, target_time, out=np.full_like(target_time, np.nan, dtype=float), where=target_time > 0.0)
            ax.plot(
                window_ids,
                target_rate,
                linestyle="--",
                color="black",
                linewidth=1.5,
                label=("target_rate_hz" if not target_drawn else None),
            )
            target_drawn = True
    ax.set_xlabel("timeslot")
    ax.set_ylabel("Rate (Hz)")
    _legend_outside(ax)
    _apply_title(fig, title, ctx)
    fig.tight_layout(rect=[0, 0, 0.82, 0.9])
    fig.savefig(_untagged_output_path(trial_dir, stem), dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_delay_ratio_over_timeslot(
    trial_dir: str,
    frames: List[pd.DataFrame],
    ctx: Dict[str, Any],
    stem: str = "delay_over_invR_over_timeslot",
    title: str = "Delay Ratio Over Timeslot",
):
    if not _frames_have_columns(frames, ["window_id", "avg_delay_sec", "target_time_sec"]):
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for frame in frames:
        algo = str(frame["algorithm"].iloc[0])
        window_ids = frame["window_id"].astype(float).to_numpy()
        delay = frame["avg_delay_sec"].astype(float).to_numpy()
        target = frame["target_time_sec"].astype(float).to_numpy()
        ratio = np.divide(delay, target, out=np.full_like(delay, np.nan, dtype=float), where=target > 0.0)
        ax.plot(window_ids, ratio, marker="o", label=algo)
    ax.axhline(1.0, linestyle="--", color="black", linewidth=1.5, label="deadline ratio=1")
    ax.set_xlabel("timeslot")
    ax.set_ylabel("delay / (1/R)")
    _legend_outside(ax)
    _apply_title(fig, title, ctx)
    fig.tight_layout(rect=[0, 0, 0.82, 0.9])
    fig.savefig(_untagged_output_path(trial_dir, stem), dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_lambda_over_timeslot(
    trial_dir: str,
    frames: List[pd.DataFrame],
    ctx: Dict[str, Any],
    stem: str = "lambda_over_timeslot",
    title: str = "Dual Variable Over Timeslot",
):
    if not _frames_have_columns(frames, ["window_id", "lambda_input"]):
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for frame in frames:
        algo = str(frame["algorithm"].iloc[0])
        ax.plot(frame["window_id"].astype(float).to_numpy(), frame["lambda_input"].astype(float).to_numpy(), marker="o", label=algo)
    ax.set_xlabel("timeslot")
    ax.set_ylabel("lambda")
    _legend_outside(ax)
    _apply_title(fig, title, ctx)
    fig.tight_layout(rect=[0, 0, 0.82, 0.9])
    fig.savefig(_untagged_output_path(trial_dir, stem), dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_eta_over_timeslot(
    trial_dir: str,
    frames: List[pd.DataFrame],
    ctx: Dict[str, Any],
    stem: str = "eta_over_timeslot",
    title: str = "Eta Over Timeslot",
):
    eta_cols = sorted([col for col in frames[0].columns if col.startswith("requested_eta_")], key=lambda x: int(x.split("_")[-1]))
    if not eta_cols:
        return
    fig, axes = plt.subplots(len(eta_cols), 1, figsize=(10, max(4, 3.0 * len(eta_cols))), sharex=True)
    if len(eta_cols) == 1:
        axes = [axes]
    for idx, eta_col in enumerate(eta_cols):
        for frame in frames:
            algo = str(frame["algorithm"].iloc[0])
            axes[idx].plot(frame["window_id"].astype(float).to_numpy(), frame[eta_col].astype(float).to_numpy(), marker="o", label=algo)
        axes[idx].set_ylabel(eta_col.replace("requested_", ""))
    axes[-1].set_xlabel("timeslot")
    _figure_legend_outside(fig, axes)
    _apply_title(fig, title, ctx)
    fig.tight_layout(rect=[0, 0, 0.82, 0.9])
    fig.savefig(_untagged_output_path(trial_dir, stem), dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_task_level_timeslot_plots(trial_dir: str, window_frames_by_task: Dict[int, List[pd.DataFrame]], ctx: Dict[str, Any]):
    for task_id in _window_task_ids_from_context_or_frames(ctx, window_frames_by_task):
        frames = window_frames_by_task.get(task_id, [])
        if not frames:
            continue
        _plot_accuracy_over_timeslot(
            trial_dir,
            frames,
            ctx,
            stem=_task_output_stem("accuracy_over_timeslot", task_id),
            title=_task_plot_title("Accuracy Over Timeslot", task_id),
        )
        _plot_delay_over_timeslot(
            trial_dir,
            frames,
            ctx,
            stem=_task_output_stem("delay_over_timeslot", task_id),
            title=_task_plot_title("Delay Over Timeslot", task_id),
        )
        _plot_rate_over_timeslot(
            trial_dir,
            frames,
            ctx,
            stem=_task_output_stem("rate_over_timeslot", task_id),
            title=_task_plot_title("Actual Rate Over Timeslot", task_id),
        )
        _plot_delay_ratio_over_timeslot(
            trial_dir,
            frames,
            ctx,
            stem=_task_output_stem("delay_over_invR_over_timeslot", task_id),
            title=_task_plot_title("Delay Ratio Over Timeslot", task_id),
        )
        _plot_lambda_over_timeslot(
            trial_dir,
            frames,
            ctx,
            stem=_task_output_stem("lambda_over_timeslot", task_id),
            title=_task_plot_title("Dual Variable Over Timeslot", task_id),
        )
        _plot_eta_over_timeslot(
            trial_dir,
            frames,
            ctx,
            stem=_task_output_stem("eta_over_timeslot", task_id),
            title=_task_plot_title("Eta Over Timeslot", task_id),
        )


def _plot_delay_breakdown_distribution(
    trial_dir: str,
    task_frames_by_task: Dict[int, List[pd.DataFrame]],
    ctx: Dict[str, Any],
):
    task_ids = _task_ids_from_context_or_task_frames(ctx, task_frames_by_task)
    if not task_ids:
        return
    if not any(
        any(
            column in frame.columns
            for column in ("service_delay_sec", "communication_delay_sec", "miscellaneous_delay_sec")
        )
        for frames in task_frames_by_task.values()
        for frame in frames
    ):
        return
    fig, axes = plt.subplots(len(task_ids), 1, figsize=(10, max(4, 3.8 * len(task_ids))), sharex=True)
    if len(task_ids) == 1:
        axes = [axes]
    for ax, task_id in zip(axes, task_ids):
        frames = task_frames_by_task.get(task_id, [])
        if not frames:
            continue
        frames = _sort_frames_by_algorithm(frames)
        algos = [str(frame["algorithm"].iloc[0]) for frame in frames]
        x = np.arange(len(algos), dtype=float)
        service = []
        communication = []
        misc = []
        for frame in frames:
            service.append(float(frame.get("service_delay_sec", pd.Series([np.nan])).astype(float).mean()))
            communication.append(float(frame.get("communication_delay_sec", pd.Series([np.nan])).astype(float).mean()))
            misc.append(float(frame.get("miscellaneous_delay_sec", pd.Series([np.nan])).astype(float).mean()))
        ax.bar(x, service, label="service")
        ax.bar(x, communication, bottom=service, label="communication")
        ax.bar(x, misc, bottom=np.asarray(service) + np.asarray(communication), label="misc")
        ax.set_ylabel("task{} delay (s)".format(task_id))
        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=20, ha="right")
    _figure_legend_outside(fig, axes)
    _apply_title(fig, "Delay Breakdown Distribution", ctx)
    fig.tight_layout(rect=[0, 0, 0.82, 0.9])
    fig.savefig(_output_path(trial_dir, "delay_breakdown_distribution", ctx), dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_trial_plots(trial_dir: str, plot_context: Optional[Dict[str, Any]] = None) -> bool:
    if plt is None:
        _log("skip {}: matplotlib is not available".format(trial_dir))
        return False
    timeseries_records = _load_timeseries_records(trial_dir)
    if not timeseries_records:
        _log("skip {}: no compatible timeseries_*.csv or ts_*.csv found".format(trial_dir))
        return False
    ctx = _load_plot_context(trial_dir, plot_context)
    window_records = _load_window_summary_records(trial_dir)
    semantic = _build_semantic_trial(trial_dir, ctx, timeseries_records, window_records)
    aggregate_frames = semantic["aggregate_frames"]
    task_frames_by_task = semantic["task_frames_by_task"]
    window_frames = semantic["window_frames"]
    window_frames_by_task = semantic.get("window_frames_by_task", {})
    _clear_plot_outputs(trial_dir, ctx)
    _plot_accuracy_over_time(trial_dir, aggregate_frames, ctx)
    _plot_delay_over_time(trial_dir, aggregate_frames, ctx)
    _plot_rate_over_time(trial_dir, aggregate_frames, ctx)
    _plot_delay_ratio(trial_dir, aggregate_frames, ctx)
    _plot_tradeoff(trial_dir, aggregate_frames, ctx)
    _plot_task_level_main_plots(trial_dir, task_frames_by_task, ctx)
    _plot_lambda(trial_dir, aggregate_frames, task_frames_by_task, ctx)
    _plot_eta(trial_dir, aggregate_frames, task_frames_by_task, ctx)
    _plot_delay_breakdown_distribution(trial_dir, task_frames_by_task, ctx)
    if window_frames:
        _plot_accuracy_over_timeslot(trial_dir, window_frames, ctx)
        _plot_delay_over_timeslot(trial_dir, window_frames, ctx)
        _plot_rate_over_timeslot(trial_dir, window_frames, ctx)
        _plot_delay_ratio_over_timeslot(trial_dir, window_frames, ctx)
        _plot_lambda_over_timeslot(trial_dir, window_frames, ctx)
        _plot_eta_over_timeslot(trial_dir, window_frames, ctx)
    if window_frames_by_task:
        _plot_task_level_timeslot_plots(trial_dir, window_frames_by_task, ctx)
    _log("rendered {}".format(trial_dir))
    return True


def render_sweep_plots(root_dir: str):
    if not os.path.isdir(root_dir):
        raise FileNotFoundError("Sweep directory not found: {}".format(root_dir))
    trial_dirs = _candidate_trial_dirs(root_dir)
    _log("scan root={} found {} candidate trial directories".format(root_dir, len(trial_dirs)))
    rendered = 0
    for trial_dir in trial_dirs:
        rendered += int(render_trial_plots(trial_dir))
    _log("finished: rendered {} / {} trial directories".format(rendered, len(trial_dirs)))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render plots for multi-task offline experiment outputs.")
    parser.add_argument("--trial_dir", default=None, help="Optional single trial directory containing timeseries_*.csv or ts_*.csv.")
    parser.add_argument("--sweep_dir", default=DEFAULT_SWEEP_OUTPUT_DIR, help="Root directory for recursive redraw. The script finds experiment folders containing policy_summary*.csv. Defaults to outputs/.")
    parser.add_argument("--plot_context", default=None, help="Optional explicit plot_context.json path.")
    return parser


def main():
    args = _build_parser().parse_args()
    if args.trial_dir:
        context = _load_json(args.plot_context) if args.plot_context else None
        render_trial_plots(os.path.abspath(args.trial_dir), plot_context=context)
    else:
        render_sweep_plots(os.path.abspath(args.sweep_dir))


if __name__ == "__main__":
    main()
