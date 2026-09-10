"""Experiment orchestration and output generation."""

import json
import logging
import numpy as np
import os
import pandas as pd
import time

from jetson_inference.analysis.single_task import render_trial_plots
from jetson_inference.common.http_transport import create_transport
from jetson_inference.resnet.compression import (
    _codec_profiles,
    _normalize_profile_name,
)
from jetson_inference.resnet.config import (
    DEFAULT_ACCURACY_ESTIMATOR_MODES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHANNEL_MODEL_TYPE,
    DEFAULT_CHANNEL_UPDATE_MODE,
    DEFAULT_CHANNEL_WINDOW_SIZE,
    DEFAULT_DATA_ROOT,
    DEFAULT_DOWNLOAD_DATA,
    DEFAULT_EPSILON,
    DEFAULT_MAX_BATCHES,
    DEFAULT_MAX_DYNAMIC_TIMESLOT_COUNT,
    DEFAULT_STEIN_FAST_MAX_BATCHES,
    DEFAULT_STEIN_N,
    DEFAULT_STEIN_SIGMA,
    DEFAULT_T0_RATIO,
    DEFAULT_TX_BUCKET_CAPACITY_BYTES,
    DEFAULT_TX_LIMIT_MBPS,
    DEFAULT_WARMUP_STEPS,
    MODEL_CHECKPOINT,
    MODEL_TAG,
    NODE_IPS,
    NODE_PORTS,
    T0_RATIOS,
    TRANSPORT_BACKEND,
    TRANSPORT_READY_TIMEOUT_SEC,
    _dynamic_timeslot_output_dir,
    _format_float4,
    _make_run_output_dir,
    _parse_accuracy_estimator_modes,
    _parse_dynamic_timeslot_sizes,
    _parse_mu_values,
    _sanitize_tag,
    _save_rows,
    _t0_output_dir,
)
from jetson_inference.resnet.estimation import _build_accuracy_estimator
from jetson_inference.resnet.model import (
    ResNet56PartitionFactory,
    load_cifar10_batches,
)
from jetson_inference.resnet.policies import (
    ConservativeSinglePolicy,
    MaxCompressionBaselinePolicy,
    MovingAverageSinglePolicy,
    MyopicSinglePolicy,
    NoCSISinglePolicy,
    NoCompressionBaselinePolicy,
)
from jetson_inference.resnet.results import (
    _aggregate_accuracy_summary,
    _summarize_last_timeslot,
    _summarize_policy_results,
    _write_plot_context,
)
from jetson_inference.resnet.runtime import (
    _prepare_profile_context,
    _run_policy,
)
from jetson_inference.resnet.workers import _send_shutdown


def run_node_a(
    device="cuda",
    checkpoint_path=MODEL_CHECKPOINT,
    batch_size=DEFAULT_BATCH_SIZE,
    max_batches=DEFAULT_MAX_BATCHES,
    data_root=DEFAULT_DATA_ROOT,
    download_data=DEFAULT_DOWNLOAD_DATA,
    channel_predictor_mode=DEFAULT_CHANNEL_UPDATE_MODE,
    channel_model_type=DEFAULT_CHANNEL_MODEL_TYPE,
    channel_window_size=DEFAULT_CHANNEL_WINDOW_SIZE,
    warmup_steps=DEFAULT_WARMUP_STEPS,
    t0_ratio=DEFAULT_T0_RATIO,
    mu_values_arg=None,
    dynamic_timeslot_sizes_arg=None,
    accuracy_estimator_modes_arg=DEFAULT_ACCURACY_ESTIMATOR_MODES,
    epsilon=DEFAULT_EPSILON,
    stein_sigma=DEFAULT_STEIN_SIGMA,
    stein_N=DEFAULT_STEIN_N,
    stein_fast_max_batches=DEFAULT_STEIN_FAST_MAX_BATCHES,
    transport_backend=TRANSPORT_BACKEND,
    tx_limit_mbps=DEFAULT_TX_LIMIT_MBPS,
    tx_bucket_capacity_bytes=DEFAULT_TX_BUCKET_CAPACITY_BYTES,
    compressor_profiles_arg="topk,quantization,llmint8_fp16_int4",
):
    output_dir = _make_run_output_dir("A", "experiment", device=device, tx_limit_mbps=tx_limit_mbps)
    logging.info("[A] Output directory: %s", output_dir)
    factory = ResNet56PartitionFactory(checkpoint_path=checkpoint_path, device=device)
    partition = factory.build_partition_for_node("A")
    batches = load_cifar10_batches(batch_size=batch_size, max_batches=max_batches, data_root=data_root, download=download_data)
    if not batches:
        raise RuntimeError("No CIFAR-10 batches were loaded; aborting experiment")

    requested_profiles = [
        _normalize_profile_name(item) for item in str(compressor_profiles_arg).split(",") if item.strip()
    ]
    profile_specs = _codec_profiles()
    for item in requested_profiles:
        if item not in profile_specs:
            raise ValueError("Unknown compressor profile '{}'. Choose from {}".format(item, list(profile_specs.keys())))

    warmup_total = min(max(0, int(warmup_steps)), len(batches))
    stein_fast_batches = batches[: max(1, int(stein_fast_max_batches))]
    mu_values = _parse_mu_values(mu_values_arg)
    dynamic_timeslot_sizes = _parse_dynamic_timeslot_sizes(dynamic_timeslot_sizes_arg)
    accuracy_estimator_modes = _parse_accuracy_estimator_modes(accuracy_estimator_modes_arg)
    t0_ratio_values = [float(t0_ratio)] if t0_ratio is not None else list(T0_RATIOS)

    transport = create_transport(
        "A",
        NODE_IPS,
        NODE_PORTS,
        backend=transport_backend,
        tx_limit_mbps=tx_limit_mbps,
        tx_bucket_capacity_bytes=tx_bucket_capacity_bytes,
    )
    transport.start()
    transport.wait_for_peers(["B", "C", "D"], timeout=TRANSPORT_READY_TIMEOUT_SEC)
    logging.info("[A] All workers are ready")

    run_start = time.perf_counter()
    run_summary = {
        "model_tag": MODEL_TAG,
        "checkpoint_path": checkpoint_path,
        "batch_size": batch_size,
        "experiment_batches": len(batches),
        "warmup_batches": warmup_total,
        "mu_values": list(mu_values),
        "dynamic_timeslot_sizes": list(dynamic_timeslot_sizes),
        "epsilon": float(epsilon),
        "accuracy_estimator_modes": list(accuracy_estimator_modes),
        "stein_sigma": float(stein_sigma),
        "stein_N": int(stein_N),
        "stein_fast_max_batches": int(len(stein_fast_batches)),
        "t0_ratios": list(t0_ratio_values),
        "partition_plan": factory.partition_plan,
        "profiles": [],
    }

    try:
        for profile_name in requested_profiles:
            profile_spec = profile_specs[profile_name]
            profile_dir = os.path.join(output_dir, _sanitize_tag(profile_name))
            os.makedirs(profile_dir, exist_ok=True)
            profile_ctx = _prepare_profile_context(
                transport=transport,
                partition=partition,
                batches=batches,
                device=device,
                output_dir=profile_dir,
                profile_name=profile_name,
                warmup_total=warmup_total,
                channel_model_type=channel_model_type,
                channel_predictor_mode=channel_predictor_mode,
                channel_window_size=channel_window_size,
            )

            profile_summary = {
                "compressor_profile": profile_name,
                "compressor_display_name": profile_spec["display_name"],
                "eta_min": [float(item) for item in profile_spec["eta_min"]],
                "warmup_median_delay_sec": float(profile_ctx["warmup_median_delay"]),
                "a_ref": [float(item) for item in profile_ctx["a_ref"]],
                "tau_list": [float(item) for item in profile_ctx["tau_list"]],
                "accuracy_estimator_runs": [],
            }

            for accuracy_estimator_mode in accuracy_estimator_modes:
                estimator_dir = os.path.join(profile_dir, _sanitize_tag(accuracy_estimator_mode))
                os.makedirs(estimator_dir, exist_ok=True)
                estimator = _build_accuracy_estimator(
                    profile_spec=profile_spec,
                    estimator_mode=accuracy_estimator_mode,
                    checkpoint_path=checkpoint_path,
                    device=device,
                    batches=stein_fast_batches,
                    stein_sigma=stein_sigma,
                    stein_N=stein_N,
                )
                if estimator is None:
                    logging.warning(
                        "[A][%s][%s] Estimator unavailable; no_csi_single will be skipped",
                        profile_name,
                        accuracy_estimator_mode,
                    )

                estimator_summary = {
                    "accuracy_estimator_mode": accuracy_estimator_mode,
                    "estimator_path": profile_spec["estimator_path"] if accuracy_estimator_mode == "fitting_model" else None,
                    "t0_runs": [],
                }

                for current_t0_ratio in t0_ratio_values:
                    t0_deadline = float(profile_ctx["warmup_median_delay"]) * float(current_t0_ratio) if profile_ctx["warmup_median_delay"] > 0 else 0.0
                    t0_output_dir = _t0_output_dir(estimator_dir, current_t0_ratio)
                    target_rate_hz = (1.0 / float(t0_deadline)) if float(t0_deadline) > 0.0 else 0.0
                    policies = []
                    if estimator is not None:
                        for mu_value in mu_values:
                            policies.append(NoCSISinglePolicy(profile_spec["eta_min"], estimator, mu=mu_value, epsilon=epsilon))
                    policies.extend(
                        [
                            MyopicSinglePolicy(profile_spec["eta_min"], estimator),
                            ConservativeSinglePolicy(profile_spec["eta_min"], estimator),
                            MovingAverageSinglePolicy(profile_spec["eta_min"], estimator, window_size=5),
                            NoCompressionBaselinePolicy(profile_spec["eta_min"]),
                            MaxCompressionBaselinePolicy(profile_spec["eta_min"]),
                        ]
                    )

                    policy_results = []
                    for policy in policies:
                        result = _run_policy(
                            transport=transport,
                            partition=partition,
                            batches=batches,
                            device=device,
                            profile_name=profile_name,
                            profile_spec=profile_spec,
                            t0_ratio=current_t0_ratio,
                            t0_deadline=t0_deadline,
                            seeded_channel_estimator=profile_ctx["seeded_channel_estimator"],
                            tau_list=profile_ctx["tau_list"],
                            a_ref=profile_ctx["a_ref"],
                            policy=policy,
                        )
                        result["accuracy_estimator_mode"] = accuracy_estimator_mode
                        policy_results.append(result)
                        _save_rows(os.path.join(t0_output_dir, "task_profiles_{}_{}.csv".format(_sanitize_tag(profile_name), _sanitize_tag(policy.policy_name))), result["task_profiles"])
                        timeseries_rows = []
                        for idx in range(len(result["accuracy_list"])):
                            plot_accuracy = float(
                                result["task_profiles"][idx].get("running_accuracy", result["accuracy_list"][idx])
                            )
                            requested_eta = [float(x) for x in result["requested_eta_history"][idx]]
                            executed_eta = [float(x) for x in result["executed_eta_history"][idx]]
                            pred_acc = result["predicted_accuracy_history"][idx]
                            pred_delay = result["predicted_delay_history"][idx]
                            timeseries_rows.append(
                                {
                                    "t": idx,
                                    "algorithm": result["policy_name"],
                                    "policy": result["policy_name"],
                                    "policy_key": result.get("policy_key"),
                                    "codec_name": profile_spec["codec_name"],
                                    "accuracy_estimator_mode": accuracy_estimator_mode,
                                    "target_rate_hz": float(target_rate_hz),
                                    "mu": (float(result["mu_value"]) if result.get("mu_value") is not None else np.nan),
                                    "epsilon": (float(epsilon) if str(result.get("policy_key")) == "no_csi_single" else np.nan),
                                    "accuracy_true": plot_accuracy,
                                    "accuracy_fit": pred_acc,
                                    "delay_actual_sec": result["delay_list"][idx],
                                    "delay_pred_sec": pred_delay,
                                    "delay_ratio_actual": (
                                        float(result["delay_list"][idx]) * float(target_rate_hz)
                                        if float(target_rate_hz) > 0.0 else np.nan
                                    ),
                                    "delay_ratio_pred": (
                                        float(pred_delay) * float(target_rate_hz)
                                        if float(target_rate_hz) > 0.0 and not pd.isna(pred_delay) else np.nan
                                    ),
                                    "lambda_value": result["lambda_list"][idx],
                                    "accuracy": plot_accuracy,
                                    "delay_sec": result["delay_list"][idx],
                                    "lambda": result["lambda_list"][idx],
                                    "requested_eta": ",".join(_format_float4(x) for x in requested_eta),
                                    "executed_eta": ",".join(_format_float4(x) for x in executed_eta),
                                    "eta_0": requested_eta[0],
                                    "eta_1": requested_eta[1],
                                    "eta_2": requested_eta[2],
                                    "executed_eta_0": executed_eta[0],
                                    "executed_eta_1": executed_eta[1],
                                    "executed_eta_2": executed_eta[2],
                                }
                            )
                        _save_rows(os.path.join(t0_output_dir, "timeseries_{}_{}.csv".format(_sanitize_tag(profile_name), _sanitize_tag(policy.policy_name))), timeseries_rows)

                    summary_rows = _summarize_policy_results(policy_results)
                    _save_rows(os.path.join(t0_output_dir, "policy_summary_{}.csv".format(_sanitize_tag(profile_name))), summary_rows)
                    plot_context = _write_plot_context(
                        output_dir=t0_output_dir,
                        profile_name=profile_name,
                        profile_spec=profile_spec,
                        target_rate_hz=target_rate_hz,
                        t0_deadline=t0_deadline,
                        batch_size=batch_size,
                        accuracy_estimator_mode=accuracy_estimator_mode,
                        stein_sigma=stein_sigma,
                        stein_N=stein_N,
                    )
                    try:
                        render_trial_plots(t0_output_dir, plot_context=plot_context)
                    except Exception as exc:
                        logging.exception(
                            "[A][%s][%s][t0=%.2f] Plot rendering failed: %s",
                            profile_name,
                            accuracy_estimator_mode,
                            current_t0_ratio,
                            exc,
                        )
                    estimator_summary["t0_runs"].append(
                        {
                            "t0_ratio": float(current_t0_ratio),
                            "t0_deadline_sec": float(t0_deadline),
                            "output_dir": t0_output_dir,
                            "accuracy_summary": _aggregate_accuracy_summary(summary_rows),
                            "policy_summary": summary_rows,
                        }
                    )

                estimator_summary["accuracy_summary"] = _aggregate_accuracy_summary(
                    [row for t0_run in estimator_summary["t0_runs"] for row in t0_run["policy_summary"]]
                )
                profile_summary["accuracy_estimator_runs"].append(estimator_summary)

            profile_summary["accuracy_summary"] = _aggregate_accuracy_summary(
                [
                    row
                    for estimator_run in profile_summary["accuracy_estimator_runs"]
                    for t0_run in estimator_run["t0_runs"]
                    for row in t0_run["policy_summary"]
                ]
            )
            with open(os.path.join(profile_dir, "profile_summary.json"), "w", encoding="utf-8") as handle:
                json.dump(profile_summary, handle, ensure_ascii=False, indent=2)
            run_summary["profiles"].append(profile_summary)
    finally:
        _send_shutdown(transport)
        transport.close()

    run_summary["run_wall_sec"] = time.perf_counter() - run_start
    run_summary["accuracy_summary"] = _aggregate_accuracy_summary(
        [
            row
            for profile in run_summary["profiles"]
            for estimator_run in profile.get("accuracy_estimator_runs", [])
            for t0_run in estimator_run.get("t0_runs", [])
            for row in t0_run.get("policy_summary", [])
        ]
    )
    with open(os.path.join(output_dir, "run_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, ensure_ascii=False, indent=2)
    logging.info("[A] Completed experiment")
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))

def _run_node_a_dynamic(
    device="cuda",
    checkpoint_path=MODEL_CHECKPOINT,
    batch_size=DEFAULT_BATCH_SIZE,
    max_batches=DEFAULT_MAX_BATCHES,
    data_root=DEFAULT_DATA_ROOT,
    download_data=DEFAULT_DOWNLOAD_DATA,
    channel_predictor_mode=DEFAULT_CHANNEL_UPDATE_MODE,
    channel_model_type=DEFAULT_CHANNEL_MODEL_TYPE,
    channel_window_size=DEFAULT_CHANNEL_WINDOW_SIZE,
    warmup_steps=DEFAULT_WARMUP_STEPS,
    t0_ratio=DEFAULT_T0_RATIO,
    mu_values_arg=None,
    dynamic_timeslot_sizes_arg=None,
    max_dynamic_timeslot_count=DEFAULT_MAX_DYNAMIC_TIMESLOT_COUNT,
    max_total_batches=None,
    accuracy_estimator_modes_arg=DEFAULT_ACCURACY_ESTIMATOR_MODES,
    epsilon=DEFAULT_EPSILON,
    stein_sigma=DEFAULT_STEIN_SIGMA,
    stein_N=DEFAULT_STEIN_N,
    stein_fast_max_batches=DEFAULT_STEIN_FAST_MAX_BATCHES,
    transport_backend=TRANSPORT_BACKEND,
    tx_limit_mbps=DEFAULT_TX_LIMIT_MBPS,
    tx_bucket_capacity_bytes=DEFAULT_TX_BUCKET_CAPACITY_BYTES,
    compressor_profiles_arg="topk,quantization,llmint8_fp16_int4",
):
    output_dir = _make_run_output_dir("A", "experiment", device=device, tx_limit_mbps=tx_limit_mbps)
    logging.info("[A] Output directory: %s", output_dir)
    factory = ResNet56PartitionFactory(checkpoint_path=checkpoint_path, device=device)
    partition = factory.build_partition_for_node("A")
    batches = load_cifar10_batches(batch_size=batch_size, max_batches=max_batches, data_root=data_root, download=download_data)
    if not batches:
        raise RuntimeError("No CIFAR-10 batches were loaded; aborting experiment")

    requested_profiles = [
        _normalize_profile_name(item) for item in str(compressor_profiles_arg).split(",") if item.strip()
    ]
    profile_specs = _codec_profiles()
    for item in requested_profiles:
        if item not in profile_specs:
            raise ValueError("Unknown compressor profile '{}'. Choose from {}".format(item, list(profile_specs.keys())))

    warmup_total = min(max(0, int(warmup_steps)), len(batches))
    stein_fast_batches = batches[: max(1, int(stein_fast_max_batches))]
    mu_values = _parse_mu_values(mu_values_arg)
    dynamic_timeslot_sizes = _parse_dynamic_timeslot_sizes(dynamic_timeslot_sizes_arg)
    accuracy_estimator_modes = _parse_accuracy_estimator_modes(accuracy_estimator_modes_arg)
    t0_ratio_values = [float(t0_ratio)] if t0_ratio is not None else list(T0_RATIOS)

    transport = create_transport(
        "A",
        NODE_IPS,
        NODE_PORTS,
        backend=transport_backend,
        tx_limit_mbps=tx_limit_mbps,
        tx_bucket_capacity_bytes=tx_bucket_capacity_bytes,
    )
    transport.start()
    transport.wait_for_peers(["B", "C", "D"], timeout=TRANSPORT_READY_TIMEOUT_SEC)
    logging.info("[A] All workers are ready")

    run_start = time.perf_counter()
    run_summary = {
        "model_tag": MODEL_TAG,
        "checkpoint_path": checkpoint_path,
        "batch_size": batch_size,
        "experiment_batches": len(batches),
        "warmup_batches": warmup_total,
        "mu_values": list(mu_values),
        "dynamic_timeslot_sizes": list(dynamic_timeslot_sizes),
        "max_dynamic_timeslot_count": (
            None if max_dynamic_timeslot_count is None else int(max_dynamic_timeslot_count)
        ),
        "max_total_batches": (
            None if max_total_batches is None else int(max_total_batches)
        ),
        "epsilon": float(epsilon),
        "accuracy_estimator_modes": list(accuracy_estimator_modes),
        "stein_sigma": float(stein_sigma),
        "stein_N": int(stein_N),
        "stein_fast_max_batches": int(len(stein_fast_batches)),
        "t0_ratios": list(t0_ratio_values),
        "partition_plan": factory.partition_plan,
        "profiles": [],
    }

    try:
        for profile_name in requested_profiles:
            profile_spec = profile_specs[profile_name]
            profile_dir = os.path.join(output_dir, _sanitize_tag(profile_name))
            os.makedirs(profile_dir, exist_ok=True)
            profile_ctx = _prepare_profile_context(
                transport=transport,
                partition=partition,
                batches=batches,
                device=device,
                output_dir=profile_dir,
                profile_name=profile_name,
                warmup_total=warmup_total,
                channel_model_type=channel_model_type,
                channel_predictor_mode=channel_predictor_mode,
                channel_window_size=channel_window_size,
            )

            profile_summary = {
                "compressor_profile": profile_name,
                "compressor_display_name": profile_spec["display_name"],
                "eta_min": [float(item) for item in profile_spec["eta_min"]],
                "warmup_median_delay_sec": float(profile_ctx["warmup_median_delay"]),
                "a_ref": [float(item) for item in profile_ctx["a_ref"]],
                "tau_list": [float(item) for item in profile_ctx["tau_list"]],
                "accuracy_estimator_runs": [],
            }

            for accuracy_estimator_mode in accuracy_estimator_modes:
                estimator_dir = os.path.join(profile_dir, _sanitize_tag(accuracy_estimator_mode))
                os.makedirs(estimator_dir, exist_ok=True)
                estimator = _build_accuracy_estimator(
                    profile_spec=profile_spec,
                    estimator_mode=accuracy_estimator_mode,
                    checkpoint_path=checkpoint_path,
                    device=device,
                    batches=stein_fast_batches,
                    stein_sigma=stein_sigma,
                    stein_N=stein_N,
                )
                if estimator is None:
                    logging.warning(
                        "[A][%s][%s] Estimator unavailable; no_csi_single will be skipped",
                        profile_name,
                        accuracy_estimator_mode,
                    )

                estimator_summary = {
                    "accuracy_estimator_mode": accuracy_estimator_mode,
                    "estimator_path": profile_spec["estimator_path"] if accuracy_estimator_mode == "fitting_model" else None,
                    "t0_runs": [],
                }

                for current_t0_ratio in t0_ratio_values:
                    t0_deadline = float(profile_ctx["warmup_median_delay"]) * float(current_t0_ratio) if profile_ctx["warmup_median_delay"] > 0 else 0.0
                    target_rate_hz = (1.0 / float(t0_deadline)) if float(t0_deadline) > 0.0 else 0.0
                    t0_output_dir = _t0_output_dir(estimator_dir, current_t0_ratio)

                    for dynamic_timeslot_size in dynamic_timeslot_sizes:
                        dynamic_output_dir = _dynamic_timeslot_output_dir(t0_output_dir, dynamic_timeslot_size)
                        policies = []
                        if estimator is not None:
                            for mu_value in mu_values:
                                policies.append(NoCSISinglePolicy(profile_spec["eta_min"], estimator, mu=mu_value, epsilon=epsilon))
                        policies.extend(
                            [
                                MyopicSinglePolicy(profile_spec["eta_min"], estimator),
                                ConservativeSinglePolicy(profile_spec["eta_min"], estimator),
                                MovingAverageSinglePolicy(profile_spec["eta_min"], estimator, window_size=5),
                                NoCompressionBaselinePolicy(profile_spec["eta_min"]),
                                MaxCompressionBaselinePolicy(profile_spec["eta_min"]),
                            ]
                        )

                        policy_results = []
                        for policy in policies:
                            result = _run_policy(
                                transport=transport,
                                partition=partition,
                                batches=batches,
                                device=device,
                                profile_name=profile_name,
                                profile_spec=profile_spec,
                                t0_ratio=current_t0_ratio,
                                t0_deadline=t0_deadline,
                                seeded_channel_estimator=profile_ctx["seeded_channel_estimator"],
                                tau_list=profile_ctx["tau_list"],
                                a_ref=profile_ctx["a_ref"],
                                policy=policy,
                                dynamic_timeslot_size=dynamic_timeslot_size,
                                max_dynamic_timeslot_count=max_dynamic_timeslot_count,
                                max_total_batches=max_total_batches,
                            )
                            result["accuracy_estimator_mode"] = accuracy_estimator_mode
                            policy_results.append(result)
                            _save_rows(
                                os.path.join(dynamic_output_dir, "task_profiles_{}_{}.csv".format(_sanitize_tag(profile_name), _sanitize_tag(policy.policy_name))),
                                result["task_profiles"],
                            )
                            _save_rows(
                                os.path.join(dynamic_output_dir, "window_summary_{}_{}.csv".format(_sanitize_tag(profile_name), _sanitize_tag(policy.policy_name))),
                                result.get("window_summaries", []),
                            )
                            timeseries_rows = []
                            for idx in range(len(result["accuracy_list"])):
                                plot_accuracy = float(
                                    result["task_profiles"][idx].get("running_accuracy", result["accuracy_list"][idx])
                                )
                                requested_eta = [float(x) for x in result["requested_eta_history"][idx]]
                                executed_eta = [float(x) for x in result["executed_eta_history"][idx]]
                                pred_acc = result["predicted_accuracy_history"][idx]
                                pred_delay = result["predicted_delay_history"][idx]
                                timeseries_rows.append(
                                    {
                                        "t": idx,
                                        "algorithm": result["policy_name"],
                                        "policy": result["policy_name"],
                                        "policy_key": result.get("policy_key"),
                                        "codec_name": profile_spec["codec_name"],
                                        "accuracy_estimator_mode": accuracy_estimator_mode,
                                        "dynamic_timeslot_size": int(dynamic_timeslot_size),
                                        "target_rate_hz": float(target_rate_hz),
                                        "mu": (float(result["mu_value"]) if result.get("mu_value") is not None else np.nan),
                                        "epsilon": (float(epsilon) if str(result.get("policy_key")) == "no_csi_single" else np.nan),
                                        "accuracy_true": plot_accuracy,
                                        "accuracy_fit": pred_acc,
                                        "delay_actual_sec": result["delay_list"][idx],
                                        "delay_pred_sec": pred_delay,
                                        "delay_ratio_actual": (
                                            float(result["delay_list"][idx]) * float(target_rate_hz)
                                            if float(target_rate_hz) > 0.0 else np.nan
                                        ),
                                        "delay_ratio_pred": (
                                            float(pred_delay) * float(target_rate_hz)
                                            if float(target_rate_hz) > 0.0 and not pd.isna(pred_delay) else np.nan
                                        ),
                                        "lambda_value": result["lambda_list"][idx],
                                        "accuracy": plot_accuracy,
                                        "delay_sec": result["delay_list"][idx],
                                        "lambda": result["lambda_list"][idx],
                                        "requested_eta": ",".join(_format_float4(x) for x in requested_eta),
                                        "executed_eta": ",".join(_format_float4(x) for x in executed_eta),
                                        "eta_0": requested_eta[0],
                                        "eta_1": requested_eta[1],
                                        "eta_2": requested_eta[2],
                                        "executed_eta_0": executed_eta[0],
                                        "executed_eta_1": executed_eta[1],
                                        "executed_eta_2": executed_eta[2],
                                    }
                                )
                            _save_rows(
                                os.path.join(dynamic_output_dir, "timeseries_{}_{}.csv".format(_sanitize_tag(profile_name), _sanitize_tag(policy.policy_name))),
                                timeseries_rows,
                            )

                        summary_rows = _summarize_policy_results(policy_results)
                        lastslot_summary_rows = _summarize_last_timeslot(policy_results)
                        _save_rows(os.path.join(dynamic_output_dir, "policy_summary_{}.csv".format(_sanitize_tag(profile_name))), summary_rows)
                        _save_rows(os.path.join(dynamic_output_dir, "summary_lasttimeslot_{}.csv".format(_sanitize_tag(profile_name))), lastslot_summary_rows)
                        plot_context = _write_plot_context(
                            output_dir=dynamic_output_dir,
                            profile_name=profile_name,
                            profile_spec=profile_spec,
                            target_rate_hz=target_rate_hz,
                            t0_deadline=t0_deadline,
                            batch_size=batch_size,
                            dynamic_timeslot_size=dynamic_timeslot_size,
                            max_dynamic_timeslot_count=max_dynamic_timeslot_count,
                            accuracy_estimator_mode=accuracy_estimator_mode,
                            stein_sigma=stein_sigma,
                            stein_N=stein_N,
                        )
                        try:
                            render_trial_plots(dynamic_output_dir, plot_context=plot_context)
                        except Exception as exc:
                            logging.exception(
                                "[A][%s][%s][t0=%.2f][dyn=%d] Plot rendering failed: %s",
                                profile_name,
                                accuracy_estimator_mode,
                                current_t0_ratio,
                                int(dynamic_timeslot_size),
                                exc,
                            )
                        estimator_summary["t0_runs"].append(
                            {
                                "t0_ratio": float(current_t0_ratio),
                                "t0_deadline_sec": float(t0_deadline),
                                "dynamic_timeslot_size": int(dynamic_timeslot_size),
                                "output_dir": dynamic_output_dir,
                                "accuracy_summary": _aggregate_accuracy_summary(summary_rows),
                                "policy_summary": summary_rows,
                                "summary_lasttimeslot": lastslot_summary_rows,
                            }
                        )

                estimator_summary["accuracy_summary"] = _aggregate_accuracy_summary(
                    [row for t0_run in estimator_summary["t0_runs"] for row in t0_run["policy_summary"]]
                )
                profile_summary["accuracy_estimator_runs"].append(estimator_summary)

            profile_summary["accuracy_summary"] = _aggregate_accuracy_summary(
                [
                    row
                    for estimator_run in profile_summary["accuracy_estimator_runs"]
                    for t0_run in estimator_run["t0_runs"]
                    for row in t0_run["policy_summary"]
                ]
            )
            with open(os.path.join(profile_dir, "profile_summary.json"), "w", encoding="utf-8") as handle:
                json.dump(profile_summary, handle, ensure_ascii=False, indent=2)
            run_summary["profiles"].append(profile_summary)
    finally:
        _send_shutdown(transport)
        transport.close()

    run_summary["run_wall_sec"] = time.perf_counter() - run_start
    run_summary["accuracy_summary"] = _aggregate_accuracy_summary(
        [
            row
            for profile in run_summary["profiles"]
            for estimator_run in profile.get("accuracy_estimator_runs", [])
            for t0_run in estimator_run.get("t0_runs", [])
            for row in t0_run.get("policy_summary", [])
        ]
    )
    with open(os.path.join(output_dir, "run_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, ensure_ascii=False, indent=2)
    logging.info("[A] Completed experiment")
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))

run_node_a = _run_node_a_dynamic
