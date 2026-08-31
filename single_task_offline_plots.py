#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
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


def _default_sweep_output_dir() -> str:
    return os.path.join("outputs")


DEFAULT_SWEEP_OUTPUT_DIR = _default_sweep_output_dir()


def _log(message: str):
    print("[single-plot] {}".format(message), file=sys.stdout, flush=True)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _load_plot_context(trial_dir: str, plot_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if plot_context is not None:
        return dict(plot_context)
    path = os.path.join(trial_dir, "plot_context.json")
    return _load_json(path) if os.path.exists(path) else {}


def _load_timeseries_frames(trial_dir: str) -> List[pd.DataFrame]:
    csv_paths = sorted(glob(os.path.join(trial_dir, "timeseries_*.csv")))
    if not csv_paths:
        csv_paths = sorted(glob(os.path.join(trial_dir, "ts_*.csv")))
    frames: List[pd.DataFrame] = []
    for path in csv_paths:
        frame = pd.read_csv(path)
        if not frame.empty:
            frame.attrs["source_path"] = path
        try:
            frame = frame.copy()
            frame["t"] = frame["t"].astype(int)
        except Exception:
            continue
        frame = frame.sort_values("t").reset_index(drop=True)
        t_values = frame["t"].to_numpy()
        if len(np.unique(t_values)) != len(t_values):
            continue
        frames.append(frame)
    return frames


def _candidate_trial_dirs(root_dir: str) -> List[str]:
    matches: List[str] = []
    for current_root, _, files in os.walk(root_dir):
        if any(name.startswith("policy_summary") and name.endswith(".csv") for name in files):
            matches.append(current_root)
    return sorted(matches)


def _load_task_profile_frames(trial_dir: str) -> List[pd.DataFrame]:
    csv_paths = sorted(glob(os.path.join(trial_dir, "task_profiles_*.csv")))
    frames: List[pd.DataFrame] = []
    for path in csv_paths:
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame.attrs["source_path"] = path
        frames.append(frame)
    return frames


def _load_window_summary_frames(trial_dir: str) -> List[pd.DataFrame]:
    csv_paths = sorted(glob(os.path.join(trial_dir, "window_summary_*.csv")))
    frames: List[pd.DataFrame] = []
    for path in csv_paths:
        frame = pd.read_csv(path)
        if frame.empty or "window_id" not in frame.columns:
            continue
        frame = frame.sort_values("window_id").reset_index(drop=True)
        frame.attrs["source_path"] = path
        if "algorithm" not in frame.columns and "policy" in frame.columns:
            frame["algorithm"] = frame["policy"]
        if "algorithm" not in frame.columns:
            stem = os.path.splitext(os.path.basename(path))[0]
            frame["algorithm"] = stem.replace("window_summary_", "")
        frames.append(frame)
    return frames


def _resolve_rate_series(reference: pd.DataFrame, ctx: Dict[str, Any]) -> np.ndarray:
    if "target_rate_hz" in reference.columns:
        return reference["target_rate_hz"].astype(float).to_numpy()
    target_rate = _normalize_optional_scalar(ctx.get("target_rate_hz"))
    if target_rate is not None:
        try:
            value = float(target_rate)
            return np.full(len(reference), value, dtype=float)
        except Exception:
            pass
    target_time = _normalize_optional_scalar(ctx.get("target_time_sec"))
    if target_time is not None:
        try:
            value = float(target_time)
            rate = (1.0 / value) if value > 0.0 else np.nan
            return np.full(len(reference), rate, dtype=float)
        except Exception:
            pass
    raise KeyError("target_rate_hz")


def _normalize_optional_scalar(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return None
        return str(value[0])
    text = str(value).strip()
    return text if text else None


def _scalar_matches(left: Optional[str], right: Optional[str], tol: float = 1e-9) -> bool:
    if left is None or right is None:
        return left is None or right is None
    if left == right:
        return True
    try:
        return abs(float(left) - float(right)) <= tol
    except Exception:
        return False


def _filter_frames_for_codec(frames: List[pd.DataFrame], ctx: Dict[str, Any]) -> List[pd.DataFrame]:
    original_frames = list(frames)
    codec_name = _normalize_optional_scalar(ctx.get("codec_name"))
    target_rate = _normalize_optional_scalar(ctx.get("target_rate_hz"))
    mu_value = _normalize_optional_scalar(ctx.get("mu"))

    filtered = list(frames)

    if codec_name is not None:
        kept = []
        for frame in filtered:
            codec_series = frame["codec_name"] if "codec_name" in frame.columns else None
            if codec_series is None:
                kept.append(frame)
                continue
            codec_scalar = _normalize_optional_scalar(codec_series.iloc[0])
            if codec_scalar is None or _scalar_matches(codec_scalar, codec_name):
                kept.append(frame)
        filtered = kept

    if target_rate is not None:
        kept = []
        for frame in filtered:
            rate_series = frame["target_rate_hz"] if "target_rate_hz" in frame.columns else None
            if rate_series is None:
                kept.append(frame)
                continue
            rate_scalar = _normalize_optional_scalar(rate_series.iloc[0])
            if rate_scalar is None or _scalar_matches(rate_scalar, target_rate, tol=1e-6):
                kept.append(frame)
        filtered = kept

    if mu_value is not None:
        kept = []
        for frame in filtered:
            frame_mu = frame["mu"] if "mu" in frame.columns else None
            if frame_mu is None:
                kept.append(frame)
                continue
            mu_scalar = _normalize_optional_scalar(frame_mu.iloc[0])
            if mu_scalar is None or _scalar_matches(mu_scalar, mu_value, tol=1e-9):
                kept.append(frame)
        filtered = kept

    return filtered if filtered else original_frames


def _subtitle_lines(ctx: Dict[str, Any]) -> List[str]:
    experiment = ctx.get("experiment", None)
    platform = ctx.get("platform", "jetson")
    model = ctx.get("model", "unknown")
    dataset = ctx.get("dataset", "unknown")
    codec = ctx.get("codec_name", "unknown")
    estimator_mode = ctx.get("accuracy_estimator_mode", None)
    rate = ctx.get("target_rate_hz", "unknown")
    target_time = ctx.get("target_time_sec", None)
    batch_size = ctx.get("batch_size", "unknown")
    stein_sigma = ctx.get("stein_sigma", None)
    stein_n = ctx.get("stein_N", None)
    if target_time is None:
        try:
            rate_float = float(rate)
            target_time = (1.0 / rate_float) if rate_float > 0.0 else float("inf")
        except Exception:
            target_time = "unknown"
    line1_parts = []
    if experiment is not None:
        line1_parts.append("experiment={}".format(experiment))
    line1_parts.extend(
        [
            "platform={}".format(platform),
            "model={}".format(model),
            "dataset={}".format(dataset),
            "codec={}".format(codec),
        ]
    )
    if estimator_mode is not None:
        line1_parts.append("acc_estimator={}".format(estimator_mode))
    line1 = " | ".join(line1_parts)
    line2 = "target_rate={rate} Hz | target_time={target_time} s | batch_size={batch_size}".format(
        rate=rate,
        target_time=("{:.6g}".format(float(target_time)) if isinstance(target_time, (int, float)) else target_time),
        batch_size=batch_size,
    )
    if stein_sigma is not None and stein_n is not None:
        line2 += " | stein_sigma={} | stein_N={}".format(stein_sigma, stein_n)
    return [line1, line2]


def _apply_title(fig, title: str, ctx: Dict[str, Any]):
    experiment = _normalize_optional_scalar(ctx.get("experiment"))
    if experiment is not None and str(experiment).strip().lower() == "online":
        title = "{} [Online]".format(title)
    fig.suptitle(title, fontsize=13, y=0.98)
    subtitle = "\n".join(_subtitle_lines(ctx))
    fig.text(0.5, 0.94, subtitle, ha="center", va="top", fontsize=9)


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


def _series_by_algorithm(frames: List[pd.DataFrame], column: str) -> Dict[str, np.ndarray]:
    return {
        str(frame["algorithm"].iloc[0]): frame[column].astype(float).to_numpy()
        for frame in frames
    }


def _plot_accuracy(trial_dir: str, frames: List[pd.DataFrame], times: np.ndarray, ctx: Dict[str, Any]):
    fig, ax = plt.subplots(figsize=(10, 5))
    for frame in frames:
        algo = str(frame["algorithm"].iloc[0])
        ax.plot(times, frame["accuracy_true"].astype(float).to_numpy(), label=algo)
    ax.set_xlabel("t")
    ax.set_ylabel("Accuracy (true)")
    ax.set_ylim(bottom=0.0)
    ymax = max(1.0, max(float(frame["accuracy_true"].astype(float).max()) for frame in frames) + 0.02)
    ax.set_ylim(top=min(ymax, 1.05))
    _legend_outside(ax)
    _apply_title(fig, "Accuracy Over Time", ctx)
    fig.tight_layout(rect=[0, 0, 0.82, 0.9])
    fig.savefig(os.path.join(trial_dir, "accuracy_over_time.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_delay(trial_dir: str, frames: List[pd.DataFrame], times: np.ndarray, deadline: np.ndarray, ctx: Dict[str, Any]):
    fig, ax = plt.subplots(figsize=(10, 5))
    for frame in frames:
        algo = str(frame["algorithm"].iloc[0])
        ax.plot(times, frame["delay_actual_sec"].astype(float).to_numpy(), label=algo)
    ax.plot(times, deadline, linestyle="--", color="black", linewidth=1.5, label="deadline=1/R")
    ax.set_xlabel("t")
    ax.set_ylabel("Delay (sec)")
    ax.set_ylim(bottom=0.0)
    _legend_outside(ax)
    _apply_title(fig, "Delay Over Time", ctx)
    fig.tight_layout(rect=[0, 0, 0.82, 0.9])
    fig.savefig(os.path.join(trial_dir, "delay_over_time.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_rate(trial_dir: str, frames: List[pd.DataFrame], times: np.ndarray, rate: np.ndarray, ctx: Dict[str, Any]):
    fig, ax = plt.subplots(figsize=(10, 5))
    for frame in frames:
        algo = str(frame["algorithm"].iloc[0])
        delay_actual = frame["delay_actual_sec"].astype(float).to_numpy()
        actual_rate = np.divide(
            1.0,
            delay_actual,
            out=np.full_like(delay_actual, np.nan, dtype=float),
            where=delay_actual > 0.0,
        )
        ax.plot(times, actual_rate, label=algo)
    ax.plot(times, rate, color="black", linestyle="--", linewidth=1.5, label="target_rate_hz")
    ax.set_xlabel("t")
    ax.set_ylabel("Rate (Hz)")
    _legend_outside(ax)
    _apply_title(fig, "Actual Rate Over Time", ctx)
    fig.tight_layout(rect=[0, 0, 0.82, 0.9])
    fig.savefig(os.path.join(trial_dir, "rate_over_time.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_delay_ratio(trial_dir: str, frames: List[pd.DataFrame], times: np.ndarray, deadline: np.ndarray, ctx: Dict[str, Any]):
    fig, ax = plt.subplots(figsize=(10, 5))
    for frame in frames:
        algo = str(frame["algorithm"].iloc[0])
        ratio = frame["delay_actual_sec"].astype(float).to_numpy() / deadline
        ax.plot(times, ratio, label=algo)
    ax.axhline(1.0, linestyle="--", color="black", linewidth=1.5, label="deadline ratio=1")
    ax.set_xlabel("t")
    ax.set_ylabel("delay / (1/R)")
    _legend_outside(ax)
    _apply_title(fig, "Delay Ratio Over Time", ctx)
    fig.tight_layout(rect=[0, 0, 0.82, 0.9])
    fig.savefig(os.path.join(trial_dir, "delay_over_invR_over_time.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_lambda(trial_dir: str, frames: List[pd.DataFrame], times: np.ndarray, ctx: Dict[str, Any]):
    fig, ax = plt.subplots(figsize=(10, 5))
    for frame in frames:
        algo = str(frame["algorithm"].iloc[0])
        ax.plot(times, frame["lambda_value"].astype(float).to_numpy(), label=algo)
    ax.set_xlabel("t")
    ax.set_ylabel("lambda")
    _legend_outside(ax)
    _apply_title(fig, "Dual Variable Over Time", ctx)
    fig.tight_layout(rect=[0, 0, 0.82, 0.9])
    fig.savefig(os.path.join(trial_dir, "lambda_over_time.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_eta(trial_dir: str, frames: List[pd.DataFrame], times: np.ndarray, ctx: Dict[str, Any]):
    eta_cols = sorted([col for col in frames[0].columns if col.startswith("eta_")], key=lambda x: int(x.split("_")[1]))
    num_links = len(eta_cols)
    fig, axes = plt.subplots(num_links, 1, figsize=(10, max(3.5 * num_links, 4)), sharex=True)
    if num_links == 1:
        axes = [axes]
    for idx, eta_col in enumerate(eta_cols):
        for frame in frames:
            algo = str(frame["algorithm"].iloc[0])
            axes[idx].plot(times, frame[eta_col].astype(float).to_numpy(), label=algo)
        axes[idx].set_ylabel(eta_col)
    axes[-1].set_xlabel("t")
    _figure_legend_outside(fig, axes)
    _apply_title(fig, "Eta Over Time", ctx)
    fig.tight_layout(rect=[0, 0, 0.82, 0.9])
    fig.savefig(os.path.join(trial_dir, "eta_over_time.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_tradeoff(trial_dir: str, frames: List[pd.DataFrame], deadline: np.ndarray, ctx: Dict[str, Any]):
    fig, ax = plt.subplots(figsize=(7, 5))
    for frame in frames:
        algo = str(frame["algorithm"].iloc[0])
        mean_delay_ratio = float(np.nanmean(frame["delay_actual_sec"].astype(float).to_numpy() / deadline))
        mean_acc = float(np.nanmean(frame["accuracy_true"].astype(float).to_numpy()))
        ax.scatter(mean_delay_ratio, mean_acc, label=algo)
    ax.set_xlabel("Average delay / (1/R)")
    ax.set_ylabel("Average true accuracy")
    ax.set_ylim(bottom=0.0)
    _legend_outside(ax)
    _apply_title(fig, "Accuracy-Delay Tradeoff", ctx)
    fig.tight_layout(rect=[0, 0, 0.82, 0.9])
    fig.savefig(os.path.join(trial_dir, "tradeoff_accuracy_vs_delay.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_accuracy_over_timeslot(trial_dir: str, frames: List[pd.DataFrame], ctx: Dict[str, Any]):
    fig, ax = plt.subplots(figsize=(10, 5))
    for frame in frames:
        algo = str(frame["algorithm"].iloc[0])
        ax.plot(frame["window_id"].astype(float).to_numpy(), frame["avg_accuracy"].astype(float).to_numpy(), marker="o", label=algo)
    ax.set_xlabel("timeslot")
    ax.set_ylabel("Accuracy (timeslot avg)")
    ax.set_ylim(bottom=0.0)
    ymax = max(1.0, max(float(frame["avg_accuracy"].astype(float).max()) for frame in frames) + 0.02)
    ax.set_ylim(top=min(ymax, 1.05))
    _legend_outside(ax)
    _apply_title(fig, "Accuracy Over Timeslot", ctx)
    fig.tight_layout(rect=[0, 0, 0.82, 0.9])
    fig.savefig(os.path.join(trial_dir, "accuracy_over_timeslot.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_delay_over_timeslot(trial_dir: str, frames: List[pd.DataFrame], ctx: Dict[str, Any]):
    fig, ax = plt.subplots(figsize=(10, 5))
    deadline_drawn = False
    for frame in frames:
        algo = str(frame["algorithm"].iloc[0])
        window_ids = frame["window_id"].astype(float).to_numpy()
        delay = frame["avg_delay_sec"].astype(float).to_numpy()
        deadline = frame["target_time_sec"].astype(float).to_numpy() if "target_time_sec" in frame.columns else np.full_like(delay, np.nan)
        ax.plot(window_ids, delay, marker="o", label=algo)
        if np.isfinite(deadline).any():
            ax.plot(
                window_ids,
                deadline,
                linestyle="--",
                linewidth=1.2,
                color="black",
                label=("deadline=1/R" if not deadline_drawn else None),
            )
            deadline_drawn = True
    ax.set_xlabel("timeslot")
    ax.set_ylabel("Delay (sec)")
    ax.set_ylim(bottom=0.0)
    _legend_outside(ax)
    _apply_title(fig, "Delay Over Timeslot", ctx)
    fig.tight_layout(rect=[0, 0, 0.82, 0.9])
    fig.savefig(os.path.join(trial_dir, "delay_over_timeslot.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_rate_over_timeslot(trial_dir: str, frames: List[pd.DataFrame], ctx: Dict[str, Any]):
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
                linewidth=1.2,
                color="black",
                label=("target_rate_hz" if not target_drawn else None),
            )
            target_drawn = True
    ax.set_xlabel("timeslot")
    ax.set_ylabel("Rate (Hz)")
    _legend_outside(ax)
    _apply_title(fig, "Actual Rate Over Timeslot", ctx)
    fig.tight_layout(rect=[0, 0, 0.82, 0.9])
    fig.savefig(os.path.join(trial_dir, "rate_over_timeslot.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_delay_ratio_over_timeslot(trial_dir: str, frames: List[pd.DataFrame], ctx: Dict[str, Any]):
    fig, ax = plt.subplots(figsize=(10, 5))
    for frame in frames:
        algo = str(frame["algorithm"].iloc[0])
        window_ids = frame["window_id"].astype(float).to_numpy()
        delay = frame["avg_delay_sec"].astype(float).to_numpy()
        target_time = frame["target_time_sec"].astype(float).to_numpy() if "target_time_sec" in frame.columns else np.full_like(delay, np.nan)
        ratio = np.divide(delay, target_time, out=np.full_like(delay, np.nan, dtype=float), where=target_time > 0.0)
        ax.plot(window_ids, ratio, marker="o", label=algo)
    ax.axhline(1.0, linestyle="--", color="black", linewidth=1.5, label="deadline ratio=1")
    ax.set_xlabel("timeslot")
    ax.set_ylabel("delay / (1/R)")
    _legend_outside(ax)
    _apply_title(fig, "Delay Ratio Over Timeslot", ctx)
    fig.tight_layout(rect=[0, 0, 0.82, 0.9])
    fig.savefig(os.path.join(trial_dir, "delay_over_invR_over_timeslot.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_lambda_over_timeslot(trial_dir: str, frames: List[pd.DataFrame], ctx: Dict[str, Any]):
    if not _frames_have_columns(frames, ["lambda_input"]):
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for frame in frames:
        algo = str(frame["algorithm"].iloc[0])
        ax.plot(frame["window_id"].astype(float).to_numpy(), frame["lambda_input"].astype(float).to_numpy(), marker="o", label=algo)
    ax.set_xlabel("timeslot")
    ax.set_ylabel("lambda")
    _legend_outside(ax)
    _apply_title(fig, "Dual Variable Over Timeslot", ctx)
    fig.tight_layout(rect=[0, 0, 0.82, 0.9])
    fig.savefig(os.path.join(trial_dir, "lambda_over_timeslot.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_eta_over_timeslot(trial_dir: str, frames: List[pd.DataFrame], ctx: Dict[str, Any]):
    eta_cols = sorted([col for col in frames[0].columns if col.startswith("requested_eta_")], key=lambda x: int(x.split("_")[-1]))
    if not eta_cols:
        return
    num_links = len(eta_cols)
    fig, axes = plt.subplots(num_links, 1, figsize=(10, max(3.5 * num_links, 4)), sharex=True)
    if num_links == 1:
        axes = [axes]
    for idx, eta_col in enumerate(eta_cols):
        for frame in frames:
            algo = str(frame["algorithm"].iloc[0])
            axes[idx].plot(frame["window_id"].astype(float).to_numpy(), frame[eta_col].astype(float).to_numpy(), marker="o", label=algo)
        axes[idx].set_ylabel(eta_col.replace("requested_", ""))
    axes[-1].set_xlabel("timeslot")
    _figure_legend_outside(fig, axes)
    _apply_title(fig, "Eta Over Timeslot", ctx)
    fig.tight_layout(rect=[0, 0, 0.82, 0.9])
    fig.savefig(os.path.join(trial_dir, "eta_over_timeslot.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def _frames_have_columns(frames: List[pd.DataFrame], columns: List[str]) -> bool:
    return all(all(column in frame.columns for column in columns) for frame in frames)


def _plot_metric_over_time(
    trial_dir: str,
    frames: List[pd.DataFrame],
    times: np.ndarray,
    column: str,
    ylabel: str,
    title: str,
    filename: str,
    ctx: Dict[str, Any],
):
    if not _frames_have_columns(frames, [column]):
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for frame in frames:
        algo = str(frame["algorithm"].iloc[0])
        ax.plot(times, frame[column].astype(float).to_numpy(), label=algo)
    ax.set_xlabel("t")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0.0)
    _legend_outside(ax)
    _apply_title(fig, title, ctx)
    fig.tight_layout(rect=[0, 0, 0.82, 0.9])
    fig.savefig(os.path.join(trial_dir, filename), dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_optimization_breakdown_over_time(trial_dir: str, frames: List[pd.DataFrame], times: np.ndarray, ctx: Dict[str, Any]):
    cols = [
        "optimization_total_sec",
        "objective_eval_sec_total",
        "accuracy_predict_sec_total",
        "gradient_eval_sec_total",
        "gradient_inner_accuracy_sec_total",
        "true_accuracy_eval_sec",
    ]
    if not _frames_have_columns(frames, cols):
        return
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    plot_specs = [
        ("optimization_total_sec", "optimization_total_sec", "Optimization Total"),
        ("objective_eval_sec_total", "objective_eval_sec_total", "Objective Eval Total"),
        ("accuracy_predict_sec_total", "accuracy_predict_sec_total", "Accuracy Estimation Total"),
        ("gradient_eval_sec_total", "gradient_eval_sec_total", "Gradient Eval Total"),
        ("gradient_inner_accuracy_sec_total", "gradient_inner_accuracy_sec_total", "Gradient Inner Accuracy Total"),
        ("true_accuracy_eval_sec", "true_accuracy_eval_sec", "True Accuracy Eval"),
    ]
    axes_flat = list(axes.reshape(-1))
    for ax, (column, ylabel, subtitle) in zip(axes_flat, plot_specs):
        for frame in frames:
            algo = str(frame["algorithm"].iloc[0])
            ax.plot(times, frame[column].astype(float).to_numpy(), label=algo)
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle, fontsize=10)
        ax.set_ylim(bottom=0.0)
    for ax in axes[-1]:
        ax.set_xlabel("t")
    _figure_legend_outside(fig, axes_flat, ncol=1)
    _apply_title(fig, "Optimization Profiling Over Time", ctx)
    fig.tight_layout(rect=[0, 0, 0.82, 0.9])
    fig.savefig(os.path.join(trial_dir, "optimization_breakdown_over_time.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_accuracy_calls_over_time(trial_dir: str, frames: List[pd.DataFrame], times: np.ndarray, ctx: Dict[str, Any]):
    cols = [
        "num_objective_calls",
        "num_accuracy_predict_calls",
        "num_gradient_calls",
        "num_accuracy_calls_total",
        "num_accuracy_calls_in_gradient",
    ]
    if not _frames_have_columns(frames, cols):
        return
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    plot_specs = [
        ("num_objective_calls", "num_objective_calls", "Objective Calls"),
        ("num_accuracy_predict_calls", "num_accuracy_predict_calls", "Accuracy Predict Calls"),
        ("num_gradient_calls", "num_gradient_calls", "Gradient Calls"),
        ("num_accuracy_calls_total", "num_accuracy_calls_total", "Accuracy Calls Total"),
        ("num_accuracy_calls_in_gradient", "num_accuracy_calls_in_gradient", "Accuracy Calls In Gradient"),
    ]
    axes_flat = list(axes.reshape(-1))
    for idx, (column, ylabel, subtitle) in enumerate(plot_specs):
        ax = axes_flat[idx]
        for frame in frames:
            algo = str(frame["algorithm"].iloc[0])
            ax.plot(times, frame[column].astype(float).to_numpy(), label=algo)
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle, fontsize=10)
        ax.set_ylim(bottom=0.0)
    axes_flat[-1].axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("t")
    _figure_legend_outside(fig, axes_flat, ncol=1)
    _apply_title(fig, "Optimization Call Counts Over Time", ctx)
    fig.tight_layout(rect=[0, 0, 0.82, 0.9])
    fig.savefig(os.path.join(trial_dir, "accuracy_calls_over_time.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_optimization_breakdown_average(trial_dir: str, frames: List[pd.DataFrame], ctx: Dict[str, Any]):
    cols = [
        "accuracy_predict_sec_total",
        "gradient_eval_sec_total",
        "objective_eval_sec_total",
        "true_accuracy_eval_sec",
    ]
    if not _frames_have_columns(frames, cols):
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    algos = [str(frame["algorithm"].iloc[0]) for frame in frames]
    x = np.arange(len(algos), dtype=float)
    accuracy_predict = np.asarray([float(frame["accuracy_predict_sec_total"].astype(float).mean()) for frame in frames], dtype=float)
    gradient_eval = np.asarray([float(frame["gradient_eval_sec_total"].astype(float).mean()) for frame in frames], dtype=float)
    objective_eval = np.asarray([float(frame["objective_eval_sec_total"].astype(float).mean()) for frame in frames], dtype=float)
    true_eval = np.asarray([float(frame["true_accuracy_eval_sec"].astype(float).mean()) for frame in frames], dtype=float)
    ax.bar(x, accuracy_predict, label="acc_estimation")
    ax.bar(x, gradient_eval, bottom=accuracy_predict, label="gradient")
    ax.bar(x, objective_eval, bottom=accuracy_predict + gradient_eval, label="objective_total")
    ax.bar(x, true_eval, bottom=accuracy_predict + gradient_eval + objective_eval, label="true_accuracy_eval")
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=20, ha="right")
    ax.set_ylabel("Average Time (sec)")
    ax.set_ylim(bottom=0.0)
    _legend_outside(ax)
    _apply_title(fig, "Average Optimization Breakdown", ctx)
    fig.tight_layout(rect=[0, 0, 0.82, 0.9])
    fig.savefig(os.path.join(trial_dir, "optimization_breakdown_average.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_delay_breakdown_distribution(trial_dir: str, frames: List[pd.DataFrame], ctx: Dict[str, Any]):
    profile_frames = _load_task_profile_frames(trial_dir)
    if not profile_frames:
        return
    records = []
    for frame in profile_frames:
        if "policy" in frame.columns:
            algo = str(frame["policy"].iloc[0])
        elif "algorithm" in frame.columns:
            algo = str(frame["algorithm"].iloc[0])
        else:
            algo = os.path.splitext(os.path.basename(frame.attrs.get("source_path", "task_profiles")))[0].replace("task_profiles_", "")
        tp_cols = ["tp0_elapsed", "tp1_elapsed", "tp2_elapsed"]
        if not all(col in frame.columns for col in tp_cols):
            continue
        service_cols = [col for col in frame.columns if col.endswith("_service_total_sec")]
        if not service_cols:
            continue
        total_col = "total_delay_sec" if "total_delay_sec" in frame.columns else ("actual_delay_sec" if "actual_delay_sec" in frame.columns else None)
        if total_col is None:
            continue
        network = frame[tp_cols].astype(float).sum(axis=1)
        service = frame[service_cols].astype(float).sum(axis=1)
        total = frame[total_col].astype(float)
        misc = total - network - service
        records.append(
            {
                "algorithm": algo,
                "service": float(service.mean()),
                "communication": float(network.mean()),
                "misc": float(misc.mean()),
            }
        )
    if not records:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    algos = [row["algorithm"] for row in records]
    x = np.arange(len(algos), dtype=float)
    service = np.asarray([row["service"] for row in records], dtype=float)
    communication = np.asarray([row["communication"] for row in records], dtype=float)
    misc = np.asarray([row["misc"] for row in records], dtype=float)
    ax.bar(x, service, label="service")
    ax.bar(x, communication, bottom=service, label="communication")
    ax.bar(x, misc, bottom=service + communication, label="misc")
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=20, ha="right")
    ax.set_ylabel("Average Delay (sec)")
    _legend_outside(ax)
    _apply_title(fig, "Delay Breakdown Distribution", ctx)
    fig.tight_layout(rect=[0, 0, 0.82, 0.9])
    fig.savefig(os.path.join(trial_dir, "delay_breakdown_distribution.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_trial_plots(trial_dir: str, plot_context: Optional[Dict[str, Any]] = None) -> bool:
    if plt is None:
        _log("skip {}: matplotlib is not available".format(trial_dir))
        return False
    frames = _load_timeseries_frames(trial_dir)
    if not frames:
        _log("skip {}: no compatible timeseries_*.csv or ts_*.csv found".format(trial_dir))
        return False
    ctx = _load_plot_context(trial_dir, plot_context)
    frames = _filter_frames_for_codec(frames, ctx)
    if not frames:
        _log("skip {}: all frames filtered out by plot_context".format(trial_dir))
        return False
    reference = frames[0]
    times = reference["t"].astype(float).to_numpy()
    try:
        rate = _resolve_rate_series(reference, ctx)
    except Exception as exc:
        _log("skip {}: unable to resolve target rate ({})".format(trial_dir, exc))
        return False
    deadline = np.divide(1.0, rate, out=np.full_like(rate, np.inf), where=rate > 0.0)

    _plot_accuracy(trial_dir, frames, times, ctx)
    _plot_delay(trial_dir, frames, times, deadline, ctx)
    _plot_rate(trial_dir, frames, times, rate, ctx)
    _plot_delay_ratio(trial_dir, frames, times, deadline, ctx)
    _plot_lambda(trial_dir, frames, times, ctx)
    _plot_eta(trial_dir, frames, times, ctx)
    _plot_tradeoff(trial_dir, frames, deadline, ctx)
    _plot_metric_over_time(
        trial_dir,
        frames,
        times,
        "optimization_total_sec",
        "optimization_total_sec",
        "Optimization Total Over Time",
        "optimization_total_over_time.png",
        ctx,
    )
    _plot_metric_over_time(
        trial_dir,
        frames,
        times,
        "accuracy_predict_sec_total",
        "accuracy_predict_sec_total",
        "Accuracy Estimation Time Over Time",
        "accuracy_estimator_time_over_time.png",
        ctx,
    )
    _plot_metric_over_time(
        trial_dir,
        frames,
        times,
        "gradient_eval_sec_total",
        "gradient_eval_sec_total",
        "Gradient Time Over Time",
        "gradient_time_over_time.png",
        ctx,
    )
    _plot_optimization_breakdown_over_time(trial_dir, frames, times, ctx)
    _plot_accuracy_calls_over_time(trial_dir, frames, times, ctx)
    _plot_optimization_breakdown_average(trial_dir, frames, ctx)
    _plot_delay_breakdown_distribution(trial_dir, frames, ctx)
    window_frames = _load_window_summary_frames(trial_dir)
    window_frames = _filter_frames_for_codec(window_frames, ctx) if window_frames else []
    if window_frames:
        _plot_accuracy_over_timeslot(trial_dir, window_frames, ctx)
        _plot_delay_over_timeslot(trial_dir, window_frames, ctx)
        _plot_rate_over_timeslot(trial_dir, window_frames, ctx)
        _plot_delay_ratio_over_timeslot(trial_dir, window_frames, ctx)
        _plot_lambda_over_timeslot(trial_dir, window_frames, ctx)
        _plot_eta_over_timeslot(trial_dir, window_frames, ctx)
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
    parser = argparse.ArgumentParser(description="Render standalone plots for a single-task offline experiment trial or sweep.")
    parser.add_argument(
        "--trial_dir",
        default=None,
        help="Optional trial output directory containing timeseries_*.csv or ts_*.csv. If omitted, the script redraws every trial under --sweep_dir.",
    )
    parser.add_argument(
        "--sweep_dir",
        default=DEFAULT_SWEEP_OUTPUT_DIR,
        help="Root directory used when --trial_dir is omitted. The script recursively finds experiment folders containing policy_summary*.csv. Defaults to outputs/.",
    )
    parser.add_argument("--plot_context", default=None, help="Optional path to plot_context.json. Defaults to <trial_dir>/plot_context.json.")
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
