"""Experiment orchestration and output generation."""

import gc
import json
import logging
import numpy as np
import os
import time
import torch

from collections import deque
from jetson_inference.multi_task.channel import (
    _current_total_bandwidth_bps,
    _probe_mean_total_bandwidth_bps,
    _warn_if_per_link_bandwidth_low,
    build_channel_estimator,
)
from jetson_inference.multi_task.config import (
    DEFAULT_ACCURACY_ESTIMATOR_MODES,
    DEFAULT_CHANNEL_MODEL_TYPE,
    DEFAULT_CHANNEL_PROBE_STEPS,
    DEFAULT_CHANNEL_UPDATE_MODE,
    DEFAULT_CHANNEL_WINDOW_SIZE,
    DEFAULT_CODEC_NAMES,
    DEFAULT_DEVICE,
    DEFAULT_DYNAMIC_TIMESLOT_SIZES,
    DEFAULT_MAX_DYNAMIC_TIMESLOT_COUNT,
    DEFAULT_MAX_INFLIGHT_PER_TASK,
    DEFAULT_MU_VALUES,
    DEFAULT_NON_MU_POLICY_COMPLETION_LIMIT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_STEIN_FAST_MAX_BATCHES,
    DEFAULT_STEIN_FAST_MAX_SAMPLES,
    DEFAULT_STEIN_N,
    DEFAULT_STEIN_SIGMA,
    DEFAULT_TX_BUCKET_CAPACITY_BYTES,
    DEFAULT_TX_LIMIT_MBPS,
    NUM_NODES,
    NUM_TRANSFER_POINTS,
    TRANSPORT_BACKEND,
    TRANSPORT_READY_TIMEOUT_SEC,
    _dynamic_timeslot_output_dir,
    _format_float4,
    _format_target_dir_name,
    _make_run_output_dir,
    _parse_accuracy_estimator_modes,
    _parse_dynamic_timeslot_sizes,
    _parse_mu_values,
    _per_link_bandwidth_bps,
    _resolve_target_ratio_groups,
    _sanitize_tag,
    _save_rows,
    _target_time_by_task,
    _tp_stats_rows_from_history,
)
from jetson_inference.multi_task.policies import (
    NoCompressionMultiTaskBaseline,
    _expand_algorithms,
    _policy_uses_estimator,
    _policy_uses_fixed_channel,
    _policy_uses_mu,
    create_policy,
)
from jetson_inference.multi_task.results import (
    _aggregate_policy_summary_rows,
    _annotate_rows_for_target,
    _build_batch_policy_summary_row,
    _build_batch_timeseries_sets,
    _build_policy_summary_row,
    _build_weighted_avg_policy_summaries,
    _build_window_summaries_sets,
    _maybe_render_plots,
    _summarize_policy_rows,
    _write_plot_context,
)
from jetson_inference.multi_task.runtime import (
    _retarget_control_records,
    _run_stage_round_robin,
)
from jetson_inference.multi_task.state import SharedBandwidthState
from jetson_inference.multi_task.tasks import _build_task_defs
from jetson_inference.multi_task.transport import (
    ThreadSafeTransportSender,
    _create_multitask_transport,
    _send_shutdown,
)
from jetson_inference.multi_task.workers import _build_runtime_map


def _run_codec_experiment(
    transport,
    sender,
    runtime_config,
    config_base_dir,
    codec_name,
    accuracy_estimator_mode,
    device,
    output_root,
    tx_limit_mbps,
    max_inflight_per_task,
    channel_predictor_mode,
    channel_model_type,
    channel_window_size,
    channel_probe_steps,
    stein_sigma,
    stein_N,
    stein_fast_max_batches,
    stein_fast_max_samples,
):
    tasks = _build_task_defs(
        config_base_dir,
        runtime_config,
        codec_name=codec_name,
        device=device,
        accuracy_estimator_mode=accuracy_estimator_mode,
        stein_sigma=stein_sigma,
        stein_N=stein_N,
        stein_fast_max_batches=stein_fast_max_batches,
        stein_fast_max_samples=stein_fast_max_samples,
    )
    runtime_map = _build_runtime_map(tasks, node_id="A", device=device)
    task_ids = [int(task.logical_task_id) for task in tasks]
    tasks_by_id = {int(task.logical_task_id): task for task in tasks}
    codec_dir = os.path.join(output_root, _sanitize_tag(codec_name))
    estimator_dir = os.path.join(codec_dir, _sanitize_tag(accuracy_estimator_mode))
    os.makedirs(estimator_dir, exist_ok=True)
    initial_per_link_bps = _per_link_bandwidth_bps(tx_limit_mbps)
    configured_total_bw_bps = np.full(NUM_TRANSFER_POINTS, float(initial_per_link_bps), dtype=float)

    probe_task_id = int(task_ids[0])
    probe_task = tasks_by_id[probe_task_id]
    probe_runtime_map = {probe_task_id: runtime_map[probe_task_id]}
    probe_shared_bandwidth = SharedBandwidthState(
        task_ids=[probe_task_id],
        num_links=NUM_TRANSFER_POINTS,
        initial_total_bps=initial_per_link_bps,
    )
    probe_allocations = {
        probe_task_id: {
            "eta": np.ones(NUM_TRANSFER_POINTS, dtype=float),
            "s_comm": np.ones(NUM_TRANSFER_POINTS, dtype=float),
            "s_comp": np.ones(NUM_NODES, dtype=float),
        }
    }
    probe_items = {
        probe_task_id: deque(list(tasks_by_id[probe_task_id].warmup_items[: int(channel_probe_steps)]))
    }
    logging.info(
        "[A][%s] Channel probe start: task=%s steps=%s",
        "{}|{}".format(codec_name, accuracy_estimator_mode),
        probe_task_id,
        int(len(probe_items[probe_task_id])),
    )
    logging.info(
        "[A][%s] Channel probe config: fixed_per_link_bw_bps=%s max_inflight_per_task=%s",
        "{}|{}".format(codec_name, accuracy_estimator_mode),
        [float(x) for x in np.asarray(probe_shared_bandwidth.total_bandwidth_bps(), dtype=float)],
        4,
    )
    channel_probe_result = _run_stage_round_robin(
        transport,
        sender,
        [probe_task],
        probe_runtime_map,
        probe_items,
        probe_allocations,
        probe_shared_bandwidth,
        max_inflight_per_task=4,
        device=device,
        policy=None,
        algorithm_name="Channel probe",
        algorithm_type="channel_probe",
        terminate_on_task_id=probe_task_id,
        loop_non_terminating_tasks=False,
        record_every_completion=False,
        freeze_bandwidth_during_stage=True,
        tx_limit_mbps_for_warning=tx_limit_mbps,
    )
    channel_probe_history = list(channel_probe_result.get("total_tp_stats_history", []))
    _save_rows(
        os.path.join(estimator_dir, "channel_probe_tp_stats.csv"),
        _tp_stats_rows_from_history(channel_probe_history),
    )
    probe_mean_total_bw_bps = _probe_mean_total_bandwidth_bps(
        channel_probe_history,
        fallback_total_bps=initial_per_link_bps,
    )
    _warn_if_per_link_bandwidth_low(
        probe_mean_total_bw_bps,
        tx_limit_mbps=tx_limit_mbps,
        context_label="{}|{}:channel_probe_mean".format(codec_name, accuracy_estimator_mode),
    )
    if channel_probe_history:
        logging.info(
            "[A][%s] Channel probe complete: mean_per_link_bw_bps=%s from %s task1-only samples",
            "{}|{}".format(codec_name, accuracy_estimator_mode),
            [float(x) for x in np.asarray(probe_mean_total_bw_bps, dtype=float)],
            int(len(channel_probe_history)),
        )
    else:
        logging.warning(
            "[A][%s] Channel probe produced no observations; falling back to fixed initial bandwidth=%s",
            "{}|{}".format(codec_name, accuracy_estimator_mode),
            [float(x) for x in np.asarray(probe_mean_total_bw_bps, dtype=float)],
        )
    logging.info(
        "[A][%s] Policy initialization bandwidth source: using configured per-link limit=%s for warmup and adaptive policy initialization",
        "{}|{}".format(codec_name, accuracy_estimator_mode),
        [float(x) for x in np.asarray(configured_total_bw_bps, dtype=float)],
    )

    shared_bandwidth = SharedBandwidthState(
        task_ids=task_ids,
        num_links=NUM_TRANSFER_POINTS,
        initial_total_bps=initial_per_link_bps,
    )
    warmup_allocations = {
        tid: {
            "eta": np.ones(NUM_TRANSFER_POINTS, dtype=float),
            "s_comm": np.full(NUM_TRANSFER_POINTS, 1.0 / max(len(task_ids), 1), dtype=float),
            "s_comp": np.full(NUM_NODES, 1.0 / max(len(task_ids), 1), dtype=float),
        }
        for tid in task_ids
    }
    warmup_policy = NoCompressionMultiTaskBaseline(tasks)
    warmup_items = {tid: deque(tasks_by_id[tid].warmup_items) for tid in task_ids}
    warmup_result = _run_stage_round_robin(
        transport,
        sender,
        tasks,
        runtime_map,
        warmup_items,
        warmup_allocations,
        shared_bandwidth,
        max_inflight_per_task=max_inflight_per_task,
        device=device,
        policy=warmup_policy,
        algorithm_name="Warmup: no compression",
        algorithm_type="no_compression_multi_baseline",
        terminate_on_task_id=None,
        loop_non_terminating_tasks=False,
        record_every_completion=True,
        channel_estimator=None,
        fixed_stage_total_bw_bps=configured_total_bw_bps,
        tx_limit_mbps_for_warning=tx_limit_mbps,
    )
    warmup_completion_rows = warmup_result["completion_rows"]
    warmup_delay_by_task = {}
    for tid in task_ids:
        delays = [row["delay_sec"] for row in warmup_completion_rows if int(row["logical_task_id"]) == tid]
        warmup_delay_by_task[tid] = float(np.median(delays)) if delays else 0.0
    anchor_task_id = max(task_ids, key=lambda tid: warmup_delay_by_task.get(tid, 0.0))
    logging.info(
        "[A][%s] Warmup complete. anchor_task_id=%s warmup_delay_by_task=%s",
        "{}|{}".format(codec_name, accuracy_estimator_mode),
        anchor_task_id,
        {int(k): _format_float4(v) for k, v in warmup_delay_by_task.items()},
    )
    logging.info(
        "[A][%s] Warmup semantics: full independent no-compression run completed before adaptive policies",
        "{}|{}".format(codec_name, accuracy_estimator_mode),
    )
    logging.info(
        "[A][%s] Initial shared bandwidth semantics: per_link_limit=%.4f Mbps on each of %s links",
        "{}|{}".format(codec_name, accuracy_estimator_mode),
        float(tx_limit_mbps),
        int(NUM_TRANSFER_POINTS),
    )

    baseline_target_time_by_task = {
        int(task.logical_task_id): float(warmup_delay_by_task.get(int(task.logical_task_id), 0.0))
        for task in tasks
    }
    dynamic_target_rate_by_task = {}
    for task in tasks:
        target_time_sec = float(baseline_target_time_by_task[int(task.logical_task_id)])
        task.target_rate_hz = (1.0 / target_time_sec) if target_time_sec > 0.0 else float(task.target_rate_hz)
        dynamic_target_rate_by_task[int(task.logical_task_id)] = float(task.target_rate_hz)
    logging.info(
        "[A][%s] Baseline target times from no-compression medians (sec)=%s -> target rates=%s",
        "{}|{}".format(codec_name, accuracy_estimator_mode),
        {int(k): _format_float4(v) for k, v in baseline_target_time_by_task.items()},
        {int(k): _format_float4(v) for k, v in dynamic_target_rate_by_task.items()},
    )
    warmup_policy_rows = _retarget_control_records(
        warmup_result.get("control_records", []),
        tasks,
        algorithm_name="Baseline: no compression",
        algorithm_type="no_compression_multi_baseline",
        anchor_task_id=anchor_task_id,
    )
    if str(channel_predictor_mode).strip().lower() != "online":
        logging.info(
            "[A][%s] Overriding channel_predictor_mode=%s -> online for multi-task experiment",
            "{}|{}".format(codec_name, accuracy_estimator_mode),
            channel_predictor_mode,
        )

    mu_values = _parse_mu_values(runtime_config.get("mu_values", DEFAULT_MU_VALUES))
    all_algorithms = _expand_algorithms(runtime_config, mu_values)
    skipped_online_algorithm_types = {
        "no_compression_multi_baseline",
        "csi_aware_multi",
        "proportional_multi_baseline",
        "strict_priority_multi_baseline",
    }
    runnable_algorithms = [
        cfg for cfg in all_algorithms
        if str(cfg["type"]).lower() not in skipped_online_algorithm_types
    ]
    all_rows = []
    logging.info(
        "[A][%s] Using configured per-link bandwidth as c_hat basis for adaptive policies: %s",
        "{}|{}".format(codec_name, accuracy_estimator_mode),
        [float(x) for x in np.asarray(configured_total_bw_bps, dtype=float)],
    )
    target_ratio_groups = _resolve_target_ratio_groups(runtime_config, tasks)
    dynamic_timeslot_sizes = _parse_dynamic_timeslot_sizes(
        runtime_config.get("dynamic_timeslot_sizes", DEFAULT_DYNAMIC_TIMESLOT_SIZES)
    )
    max_dynamic_timeslot_count = runtime_config.get("max_dynamic_timeslot_count", DEFAULT_MAX_DYNAMIC_TIMESLOT_COUNT)
    max_total_batches = runtime_config.get("max_total_batches")
    policy_summaries = []
    lastslot_policy_summaries = []
    policy_summaries_by_scope = {}
    lastslot_policy_summaries_by_scope = {}
    for ratio_by_task in target_ratio_groups:
        per_target_rate_by_task = {}
        for task in tasks:
            warmup_delay = float(warmup_delay_by_task.get(int(task.logical_task_id), 0.0))
            target_ratio = float(ratio_by_task[int(task.logical_task_id)])
            target_time_sec = warmup_delay * float(target_ratio) if warmup_delay > 0.0 else 0.0
            task.target_rate_hz = (1.0 / target_time_sec) if target_time_sec > 0.0 else float(task.target_rate_hz)
            per_target_rate_by_task[int(task.logical_task_id)] = float(task.target_rate_hz)
        target_label = _format_target_dir_name(ratio_by_task)
        target_dir = os.path.join(estimator_dir, target_label)
        os.makedirs(target_dir, exist_ok=True)
        logging.info(
            "[A][%s] Target group ratios=%s -> target rates=%s output_dir=%s",
            "{}|{}".format(codec_name, accuracy_estimator_mode),
            {int(k): _format_float4(v) for k, v in sorted(ratio_by_task.items())},
            {int(k): _format_float4(v) for k, v in per_target_rate_by_task.items()},
            target_dir,
        )
        for dynamic_timeslot_size in dynamic_timeslot_sizes:
            dynamic_dir = _dynamic_timeslot_output_dir(target_dir, dynamic_timeslot_size)
            os.makedirs(dynamic_dir, exist_ok=True)
            target_plot_context = _write_plot_context(
                dynamic_dir,
                tasks,
                codec_name,
                tx_limit_mbps,
                ratio_by_task,
                accuracy_estimator_mode=accuracy_estimator_mode,
                stein_sigma=stein_sigma,
                stein_N=stein_N,
                dynamic_timeslot_size=int(dynamic_timeslot_size),
                max_dynamic_timeslot_count=max_dynamic_timeslot_count,
                anchor_task_id=anchor_task_id,
            )

            baseline_rows_by_scope = {
                scope_key: _annotate_rows_for_target(
                    rows,
                    ratio_by_task,
                    accuracy_estimator_mode=accuracy_estimator_mode,
                    anchor_task_id=anchor_task_id,
                )
                for scope_key, rows in _build_batch_timeseries_sets(
                    warmup_completion_rows,
                    tasks,
                    codec_name=codec_name,
                    dynamic_timeslot_size=int(dynamic_timeslot_size),
                    algorithm_name="Baseline: no compression",
                    algorithm_type="no_compression_multi_baseline",
                ).items()
            }
            baseline_window_rows_by_scope = {
                scope_key: _annotate_rows_for_target(
                    rows,
                    ratio_by_task,
                    accuracy_estimator_mode=accuracy_estimator_mode,
                    anchor_task_id=anchor_task_id,
                )
                for scope_key, rows in _build_window_summaries_sets(
                    baseline_rows_by_scope,
                    dynamic_timeslot_size=int(dynamic_timeslot_size),
                    anchor_task_id=int(anchor_task_id),
                    tasks=tasks,
                ).items()
            }
            baseline_rows_for_target = list(baseline_rows_by_scope.get("avg", []))
            baseline_window_rows = list(baseline_window_rows_by_scope.get("avg", []))
            for row in baseline_rows_for_target:
                row["dynamic_timeslot_size"] = int(dynamic_timeslot_size)
            baseline_batch_summary_by_scope = {
                scope_key: _build_batch_policy_summary_row(rows)
                for scope_key, rows in baseline_rows_by_scope.items()
            }
            baseline_batch_summary_by_scope = _build_weighted_avg_policy_summaries(
                baseline_batch_summary_by_scope,
                tasks,
            )
            baseline_lastslot_summary_by_scope = {
                scope_key: _build_policy_summary_row(
                    baseline_rows_by_scope.get(scope_key, []),
                    baseline_window_rows_by_scope.get(scope_key, []),
                )
                for scope_key in baseline_rows_by_scope.keys()
            }
            baseline_lastslot_summary_by_scope = _build_weighted_avg_policy_summaries(
                baseline_lastslot_summary_by_scope,
                tasks,
            )
            warmup_tag = _sanitize_tag("Baseline: no compression")
            _save_rows(os.path.join(dynamic_dir, "timeseries_{}.csv".format(warmup_tag)), baseline_rows_for_target)
            _save_rows(os.path.join(dynamic_dir, "window_summary_{}.csv".format(warmup_tag)), baseline_window_rows)
            for scope_key, rows in baseline_rows_by_scope.items():
                _save_rows(os.path.join(dynamic_dir, "timeseries_{}_{}.csv".format(warmup_tag, scope_key)), rows)
            for scope_key, rows in baseline_window_rows_by_scope.items():
                _save_rows(os.path.join(dynamic_dir, "window_summary_{}_{}.csv".format(warmup_tag, scope_key)), rows)
            dynamic_rows = list(baseline_rows_for_target)
            target_summary = [baseline_batch_summary_by_scope["avg"]] if baseline_batch_summary_by_scope.get("avg") is not None else []
            target_lastslot_summary = [baseline_lastslot_summary_by_scope["avg"]] if baseline_lastslot_summary_by_scope.get("avg") is not None else []
            target_summary_by_scope = {
                scope_key: ([summary_row] if summary_row is not None else [])
                for scope_key, summary_row in baseline_batch_summary_by_scope.items()
            }
            target_lastslot_summary_by_scope = {
                scope_key: ([summary_row] if summary_row is not None else [])
                for scope_key, summary_row in baseline_lastslot_summary_by_scope.items()
            }

            for algo_cfg in runnable_algorithms:
                algo_type = str(algo_cfg["type"])
                policy = create_policy(algo_cfg, tasks)
                algo_name = str(algo_cfg.get("name", policy.policy_name))
                logging.info(
                    "[A][%s][%s][dyn=%d] Policy start",
                    "{}|{}".format(codec_name, accuracy_estimator_mode),
                    algo_name,
                    int(dynamic_timeslot_size),
                )
                shared_bandwidth_policy = SharedBandwidthState(
                    task_ids=task_ids,
                    num_links=NUM_TRANSFER_POINTS,
                    initial_total_bps=initial_per_link_bps,
                )
                stage_channel_estimator = None
                stage_fixed_total_bw_bps = None
                if _policy_uses_fixed_channel(policy=policy, algorithm_type=algo_type):
                    stage_fixed_total_bw_bps = configured_total_bw_bps.copy()
                elif _policy_uses_estimator(policy=policy, algorithm_type=algo_type):
                    stage_channel_estimator = build_channel_estimator(
                        device=device,
                        model_type=channel_model_type,
                        update_mode=channel_predictor_mode,
                        window_size=channel_window_size,
                        default_bandwidth_bps=initial_per_link_bps,
                    )
                t_solve = time.perf_counter()
                initial_allocations = policy.current_allocations(
                    warmup_result["latest_snapshots"],
                    (
                        _current_total_bandwidth_bps(stage_channel_estimator, shared_bandwidth_policy)
                        if stage_channel_estimator is not None else
                        configured_total_bw_bps
                    ),
                )
                initial_control_solve_sec = time.perf_counter() - t_solve
                non_mu_policy_completion_limit = (
                    int(DEFAULT_NON_MU_POLICY_COMPLETION_LIMIT)
                    if not _policy_uses_mu(policy=policy, algorithm_type=algo_type)
                    else None
                )
                logging.info(
                    "[A][%s][%s][dyn=%d] Initial allocation solve time: %.4fs",
                    "{}|{}".format(codec_name, accuracy_estimator_mode),
                    algo_name,
                    int(dynamic_timeslot_size),
                    float(initial_control_solve_sec),
                )
                experiment_items = {tid: deque(tasks_by_id[tid].experiment_items) for tid in task_ids}
                trial_result = _run_stage_round_robin(
                    transport,
                    sender,
                    tasks,
                    runtime_map,
                    experiment_items,
                    initial_allocations,
                    shared_bandwidth_policy,
                    max_inflight_per_task=max_inflight_per_task,
                    device=device,
                    policy=policy,
                    anchor_task_id=anchor_task_id,
                    output_dir=dynamic_dir,
                    plot_context=target_plot_context,
                    algorithm_name=algo_name,
                    algorithm_type=algo_type,
                    channel_estimator=stage_channel_estimator,
                    terminate_on_task_id=anchor_task_id,
                    loop_non_terminating_tasks=True,
                    fixed_stage_total_bw_bps=stage_fixed_total_bw_bps,
                    tx_limit_mbps_for_warning=tx_limit_mbps,
                    dynamic_timeslot_size=int(dynamic_timeslot_size),
                    max_dynamic_timeslot_count=max_dynamic_timeslot_count,
                    max_total_batches=max_total_batches,
                    target_completion_count_override=non_mu_policy_completion_limit,
                )
                target_rows_by_scope = {
                    scope_key: _annotate_rows_for_target(
                        rows,
                        ratio_by_task,
                        accuracy_estimator_mode=accuracy_estimator_mode,
                        anchor_task_id=anchor_task_id,
                    )
                    for scope_key, rows in _build_batch_timeseries_sets(
                        trial_result["completion_rows"],
                        tasks,
                        codec_name=codec_name,
                        dynamic_timeslot_size=int(dynamic_timeslot_size),
                        algorithm_name=algo_name,
                        algorithm_type=algo_type,
                    ).items()
                }
                window_rows_by_scope = {
                    scope_key: _annotate_rows_for_target(
                        rows,
                        ratio_by_task,
                        accuracy_estimator_mode=accuracy_estimator_mode,
                        anchor_task_id=anchor_task_id,
                    )
                    for scope_key, rows in _build_window_summaries_sets(
                        target_rows_by_scope,
                        dynamic_timeslot_size=int(dynamic_timeslot_size),
                        anchor_task_id=int(anchor_task_id),
                        tasks=tasks,
                    ).items()
                }
                target_rows = list(target_rows_by_scope.get("avg", []))
                window_rows = list(window_rows_by_scope.get("avg", []))
                for row in target_rows:
                    row["dynamic_timeslot_size"] = int(dynamic_timeslot_size)
                algo_tag = _sanitize_tag(algo_name)
                _save_rows(os.path.join(dynamic_dir, "timeseries_{}.csv".format(algo_tag)), target_rows)
                _save_rows(os.path.join(dynamic_dir, "window_summary_{}.csv".format(algo_tag)), window_rows)
                for scope_key, rows in target_rows_by_scope.items():
                    _save_rows(os.path.join(dynamic_dir, "timeseries_{}_{}.csv".format(algo_tag, scope_key)), rows)
                for scope_key, rows in window_rows_by_scope.items():
                    _save_rows(os.path.join(dynamic_dir, "window_summary_{}_{}.csv".format(algo_tag, scope_key)), rows)
                dynamic_rows.extend(target_rows)
                batch_summary_by_scope = {
                    scope_key: _build_batch_policy_summary_row(rows)
                    for scope_key, rows in target_rows_by_scope.items()
                }
                batch_summary_by_scope = _build_weighted_avg_policy_summaries(
                    batch_summary_by_scope,
                    tasks,
                )
                lastslot_summary_by_scope = {
                    scope_key: _build_policy_summary_row(
                        target_rows_by_scope.get(scope_key, []),
                        window_rows_by_scope.get(scope_key, []),
                    )
                    for scope_key in target_rows_by_scope.keys()
                }
                lastslot_summary_by_scope = _build_weighted_avg_policy_summaries(
                    lastslot_summary_by_scope,
                    tasks,
                )
                if batch_summary_by_scope.get("avg") is not None:
                    target_summary.append(batch_summary_by_scope["avg"])
                if lastslot_summary_by_scope.get("avg") is not None:
                    target_lastslot_summary.append(lastslot_summary_by_scope["avg"])
                for scope_key, summary_row in batch_summary_by_scope.items():
                    if summary_row is not None:
                        target_summary_by_scope.setdefault(scope_key, []).append(summary_row)
                for scope_key, summary_row in lastslot_summary_by_scope.items():
                    if summary_row is not None:
                        target_lastslot_summary_by_scope.setdefault(scope_key, []).append(summary_row)
                logging.info(
                    "[A][%s][%s][dyn=%d] Policy finished. terminated_task_id=%s termination_reason=%s",
                    "{}|{}".format(codec_name, accuracy_estimator_mode),
                    algo_name,
                    int(dynamic_timeslot_size),
                    trial_result.get("terminated_task_id"),
                    trial_result.get("termination_reason"),
                )

            _save_rows(os.path.join(dynamic_dir, "policy_summary.csv"), target_summary)
            _save_rows(os.path.join(dynamic_dir, "summary_lasttimeslot.csv"), target_lastslot_summary)
            for scope_key, rows in target_summary_by_scope.items():
                _save_rows(os.path.join(dynamic_dir, "policy_summary_{}.csv".format(scope_key)), rows)
            for scope_key, rows in target_lastslot_summary_by_scope.items():
                _save_rows(os.path.join(dynamic_dir, "summary_lasttimeslot_{}.csv".format(scope_key)), rows)
            target_run_summary = {
                "codec_name": codec_name,
                "accuracy_estimator_mode": str(accuracy_estimator_mode),
                "target_ratio": float(ratio_by_task[int(anchor_task_id)]),
                "target_ratio_by_task": {int(k): float(v) for k, v in sorted(ratio_by_task.items())},
                "dynamic_timeslot_size": int(dynamic_timeslot_size),
                "max_dynamic_timeslot_count": (
                    None if max_dynamic_timeslot_count is None else int(max_dynamic_timeslot_count)
                ),
                "max_total_batches": (None if max_total_batches is None else int(max_total_batches)),
                "target_time_sec_by_task": _target_time_by_task(tasks),
                "target_rate_by_task_hz": {int(task.logical_task_id): float(task.target_rate_hz) for task in tasks},
                "probe_mean_bandwidth_mbps_per_link": [float(x) / 1e6 for x in np.asarray(probe_mean_total_bw_bps, dtype=float)],
                "warmup_delay_by_task_sec": warmup_delay_by_task,
                "policy_summary": target_summary,
                "summary_lasttimeslot": target_lastslot_summary,
                "accuracy_summary": _aggregate_policy_summary_rows(target_summary),
            }
            with open(os.path.join(dynamic_dir, "run_summary.json"), "w", encoding="utf-8") as handle:
                json.dump(target_run_summary, handle, indent=2)
            policy_summaries.extend(target_summary)
            lastslot_policy_summaries.extend(target_lastslot_summary)
            for scope_key, rows in target_summary_by_scope.items():
                policy_summaries_by_scope.setdefault(scope_key, []).extend(rows)
            for scope_key, rows in target_lastslot_summary_by_scope.items():
                lastslot_policy_summaries_by_scope.setdefault(scope_key, []).extend(rows)
            all_rows.extend(dynamic_rows)
            _maybe_render_plots(dynamic_dir, target_plot_context)

    policy_summary = policy_summaries if policy_summaries else _summarize_policy_rows(all_rows)
    _save_rows(os.path.join(estimator_dir, "policy_summary.csv"), policy_summary)
    _save_rows(os.path.join(estimator_dir, "summary_lasttimeslot.csv"), lastslot_policy_summaries)
    for scope_key, rows in policy_summaries_by_scope.items():
        _save_rows(os.path.join(estimator_dir, "policy_summary_{}.csv".format(scope_key)), rows)
    for scope_key, rows in lastslot_policy_summaries_by_scope.items():
        _save_rows(os.path.join(estimator_dir, "summary_lasttimeslot_{}.csv".format(scope_key)), rows)
    run_summary = {
        "codec_name": codec_name,
        "accuracy_estimator_mode": str(accuracy_estimator_mode),
        "anchor_task_id": int(anchor_task_id),
        "channel_probe_task_id": int(probe_task_id),
        "channel_probe_steps": int(len(channel_probe_history)),
        "probe_mean_bandwidth_mbps_per_link": [float(x) / 1e6 for x in np.asarray(probe_mean_total_bw_bps, dtype=float)],
        "warmup_delay_by_task_sec": warmup_delay_by_task,
        "baseline_target_time_sec_by_task": baseline_target_time_by_task,
        "baseline_target_rate_by_task_hz": dynamic_target_rate_by_task,
        "target_ratio_groups": [
            {int(k): float(v) for k, v in sorted(group.items())} for group in target_ratio_groups
        ],
        "dynamic_timeslot_sizes": [int(x) for x in dynamic_timeslot_sizes],
        "max_dynamic_timeslot_count": (
            None if max_dynamic_timeslot_count is None else int(max_dynamic_timeslot_count)
        ),
        "max_total_batches": (None if max_total_batches is None else int(max_total_batches)),
        "tx_limit_mbps": float(tx_limit_mbps),
        "initial_per_link_bandwidth_mbps": float(tx_limit_mbps),
        "channel_limit_semantics": "per_link",
        "last_terminated_task_id": (
            trial_result.get("terminated_task_id") if 'trial_result' in locals() else None
        ),
        "policy_summary": policy_summary,
        "summary_lasttimeslot": lastslot_policy_summaries,
        "accuracy_summary": _aggregate_policy_summary_rows(policy_summary),
    }
    with open(os.path.join(estimator_dir, "run_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2)
    logging.info(
        "[A][%s] Codec-scoped cleanup start: releasing tasks/runtime/probe/warmup state before next codec or estimator mode",
        "{}|{}".format(codec_name, accuracy_estimator_mode),
    )
    del tasks
    del runtime_map
    del task_ids
    del tasks_by_id
    del probe_task
    del probe_runtime_map
    del probe_shared_bandwidth
    del probe_allocations
    del probe_items
    del channel_probe_result
    del channel_probe_history
    del shared_bandwidth
    del warmup_allocations
    del warmup_policy
    del warmup_items
    del warmup_result
    del warmup_completion_rows
    del warmup_delay_by_task
    del baseline_target_time_by_task
    del dynamic_target_rate_by_task
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return run_summary

def run_node_a(
    runtime_config,
    config_base_dir,
    device=DEFAULT_DEVICE,
    base_output_dir=DEFAULT_OUTPUT_DIR,
    transport_backend=TRANSPORT_BACKEND,
    tx_limit_mbps=DEFAULT_TX_LIMIT_MBPS,
    tx_bucket_capacity_bytes=DEFAULT_TX_BUCKET_CAPACITY_BYTES,
    max_inflight_per_task=DEFAULT_MAX_INFLIGHT_PER_TASK,
    channel_predictor_mode=DEFAULT_CHANNEL_UPDATE_MODE,
    channel_model_type=DEFAULT_CHANNEL_MODEL_TYPE,
    channel_window_size=DEFAULT_CHANNEL_WINDOW_SIZE,
    channel_probe_steps=DEFAULT_CHANNEL_PROBE_STEPS,
    stein_sigma=DEFAULT_STEIN_SIGMA,
    stein_N=DEFAULT_STEIN_N,
    stein_fast_max_batches=DEFAULT_STEIN_FAST_MAX_BATCHES,
    stein_fast_max_samples=DEFAULT_STEIN_FAST_MAX_SAMPLES,
):
    output_root = _make_run_output_dir("A", base_output_dir, device, tx_limit_mbps)
    transport = _create_multitask_transport(
        node_id="A",
        transport_backend=transport_backend,
        tx_bucket_capacity_bytes=tx_bucket_capacity_bytes,
    )
    transport.start()
    transport.wait_for_peers(["B", "C", "D"], timeout=TRANSPORT_READY_TIMEOUT_SEC)
    sender = ThreadSafeTransportSender(transport)
    codec_names = list(runtime_config.get("codec_names", DEFAULT_CODEC_NAMES))
    accuracy_estimator_modes = _parse_accuracy_estimator_modes(
        runtime_config.get("accuracy_estimator_modes", DEFAULT_ACCURACY_ESTIMATOR_MODES)
    )
    run_summary = {
        "experiment": "online",
        "device": device,
        "tx_limit_mbps": float(tx_limit_mbps),
        "channel_limit_semantics": "per_link",
        "codec_runs": [],
    }
    try:
        for codec_name in codec_names:
            for accuracy_estimator_mode in accuracy_estimator_modes:
                codec_summary = _run_codec_experiment(
                    transport,
                    sender,
                    runtime_config,
                    config_base_dir,
                    codec_name=str(codec_name),
                    accuracy_estimator_mode=str(accuracy_estimator_mode),
                    device=device,
                    output_root=output_root,
                    tx_limit_mbps=tx_limit_mbps,
                    max_inflight_per_task=max_inflight_per_task,
                    channel_predictor_mode=channel_predictor_mode,
                    channel_model_type=channel_model_type,
                    channel_window_size=channel_window_size,
                    channel_probe_steps=channel_probe_steps,
                    stein_sigma=stein_sigma,
                    stein_N=stein_N,
                    stein_fast_max_batches=stein_fast_max_batches,
                    stein_fast_max_samples=stein_fast_max_samples,
                )
                run_summary["codec_runs"].append(codec_summary)
    finally:
        _send_shutdown(transport)
        transport.close()
    with open(os.path.join(output_root, "run_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2)
    return output_root
