"""channel-estimator construction and warmup initialization."""

import logging
import numpy as np

from jetson_inference.common.channel_estimator import ChannelEstimator
from jetson_inference.multi_task.config import (
    DEFAULT_BANDWIDTH_WARNING_RATIO,
    DEFAULT_CHANNEL_MODEL_TYPE,
    DEFAULT_CHANNEL_UPDATE_MODE,
    DEFAULT_CHANNEL_WINDOW_SIZE,
    NUM_TRANSFER_POINTS,
)


def build_channel_estimator(device="cpu",
                            model_type=DEFAULT_CHANNEL_MODEL_TYPE,
                            update_mode=DEFAULT_CHANNEL_UPDATE_MODE,
                            window_size=DEFAULT_CHANNEL_WINDOW_SIZE,
                            default_bandwidth_bps=10e6):
    # ChannelEstimator stores/returns throughput in bytes/sec.
    return ChannelEstimator(
        n_links=NUM_TRANSFER_POINTS,
        model_type=model_type,
        update_mode=update_mode,
        window_size=window_size,
        default_bandwidth=float(default_bandwidth_bps) / 8.0,
        device=device,
    )

def _build_seeded_channel_estimator(warmup_tp_stats,
                                    device,
                                    model_type,
                                    update_mode,
                                    window_size,
                                    default_bandwidth_bps):
    estimator = build_channel_estimator(
        device=device,
        model_type=model_type,
        update_mode=update_mode,
        window_size=window_size,
        default_bandwidth_bps=default_bandwidth_bps,
    )
    for tp_stats in warmup_tp_stats:
        estimator.observe_task(tp_stats, refit=False)
    estimator.fit()
    return estimator

def _probe_mean_total_bandwidth_bps(tp_stats_history, fallback_total_bps):
    fallback = np.full(NUM_TRANSFER_POINTS, float(fallback_total_bps), dtype=float)
    if not tp_stats_history:
        return fallback
    rows = []
    for tp_stats in tp_stats_history:
        per_link = []
        for bytes_sent, elapsed_sec in tp_stats:
            elapsed = max(float(elapsed_sec), 1e-9)
            per_link.append((float(bytes_sent) * 8.0) / elapsed)
        if len(per_link) == NUM_TRANSFER_POINTS:
            rows.append(np.asarray(per_link, dtype=float))
    if not rows:
        return fallback
    mean_values = np.mean(np.stack(rows, axis=0), axis=0)
    return np.asarray(mean_values, dtype=float)

def _current_total_bandwidth_bps(channel_estimator, shared_bandwidth):
    if channel_estimator is None:
        return shared_bandwidth.total_bandwidth_bps()
    try:
        has_history = any(len(link.history) > 0 for link in channel_estimator.links)
    except Exception:
        has_history = False
    if not has_history:
        return shared_bandwidth.total_bandwidth_bps()
    predicted_bytes_per_sec = np.asarray(channel_estimator.predict_all(), dtype=float)
    return predicted_bytes_per_sec * 8.0

def _warn_if_per_link_bandwidth_low(link_bw_bps, tx_limit_mbps, context_label, ratio=DEFAULT_BANDWIDTH_WARNING_RATIO):
    if tx_limit_mbps is None:
        return
    link_bw_bps = np.asarray(link_bw_bps, dtype=float)
    threshold_bps = float(tx_limit_mbps) * float(ratio) * 1e6
    low_links = [
        {
            "link_idx": int(idx),
            "predicted_mbps": float(value) / 1e6,
            "threshold_mbps": threshold_bps / 1e6,
        }
        for idx, value in enumerate(link_bw_bps)
        if float(value) < threshold_bps
    ]
    if not low_links:
        return
    logging.warning(
        "[bandwidth-check][%s] Per-link bandwidth fell below %.0f%% of tx_limit_mbps=%.4f. details=%s",
        str(context_label),
        float(ratio) * 100.0,
        float(tx_limit_mbps),
        low_links,
    )
