#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline codec mapping helpers.

Purpose
-------
The offline optimizer solves in an eta / feature-k space because the fitted
accuracy model is defined over that space.

For TopK:
    eta is already the real execution parameter.

For LLM.int8:
    eta is not the real execution parameter. Real execution requires a
    feasible tuple of per-link outlier ratios plus fixed precisions.
    Therefore we map the solver output eta to the nearest feasible
    feature_k_values entry and then recover the corresponding outlier tuple.
"""

from typing import Any, Dict, List, Sequence

import numpy as np


def _as_float_list(values: Sequence[Any]) -> List[float]:
    return [float(item) for item in values]


def validate_llmint8_mapping_entries(entries: Sequence[Dict[str, Any]], num_links: int) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        if "feature_k_values" not in entry or "outlier_values" not in entry:
            raise ValueError(
                "llmint8 mapping entry {} must contain both 'feature_k_values' and 'outlier_values'".format(idx)
            )
        feature_k_values = _as_float_list(entry["feature_k_values"])
        outlier_values = _as_float_list(entry["outlier_values"])
        if len(feature_k_values) != int(num_links):
            raise ValueError(
                "llmint8 mapping entry {} feature_k_values length {} != num_links {}".format(
                    idx, len(feature_k_values), num_links
                )
            )
        if len(outlier_values) != int(num_links):
            raise ValueError(
                "llmint8 mapping entry {} outlier_values length {} != num_links {}".format(
                    idx, len(outlier_values), num_links
                )
            )
        normalized.append(
            {
                "feature_k_values": feature_k_values,
                "outlier_values": outlier_values,
            }
        )
    if not normalized:
        raise ValueError("llmint8 mapping entries cannot be empty")
    return normalized


def nearest_llmint8_codec_config(
    eta: Sequence[float],
    mapping_entries: Sequence[Dict[str, Any]],
    active_mask: Sequence[bool] | None = None,
) -> Dict[str, Any]:
    eta_arr = np.asarray(eta, dtype=float)
    if active_mask is None:
        active_mask_arr = np.ones_like(eta_arr, dtype=bool)
    else:
        active_mask_arr = np.asarray(active_mask, dtype=bool)
        if active_mask_arr.shape != eta_arr.shape:
            raise ValueError(
                "llmint8 active_mask shape {} does not match eta shape {}".format(
                    active_mask_arr.shape, eta_arr.shape
                )
            )
    best_entry = None
    best_distance = None
    for entry in mapping_entries:
        feature_k = np.asarray(entry["feature_k_values"], dtype=float)
        if feature_k.shape != eta_arr.shape:
            raise ValueError(
                "llmint8 mapping feature_k_values shape {} does not match eta shape {}".format(
                    feature_k.shape, eta_arr.shape
                )
            )
        distance = float(np.linalg.norm(feature_k[active_mask_arr] - eta_arr[active_mask_arr]))
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_entry = entry
    if best_entry is None:
        raise RuntimeError("No llmint8 mapping entries available")
    return {
        "requested_eta": _as_float_list(eta_arr.tolist()),
        "matched_feature_k_values": _as_float_list(best_entry["feature_k_values"]),
        "matched_outlier_values": _as_float_list(best_entry["outlier_values"]),
        "match_distance_l2": float(best_distance),
    }


def resolve_codec_execution_plan(
    *,
    codec_name: str,
    eta: Sequence[float],
    outlier_precision: str = "fp16",
    regular_precision: str = "int4",
    llmint8_mapping_entries: Sequence[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    codec_name = str(codec_name).lower()
    eta_list = _as_float_list(eta)

    if codec_name == "topk":
        return {
            "codec_name": "topk",
            "requested_eta": eta_list,
            "execution_feature_k_values": eta_list,
            "compression_params_list": eta_list,
            "mapping_info": None,
        }

    if codec_name == "quantization":
        return {
            "codec_name": "quantization",
            "requested_eta": eta_list,
            "execution_feature_k_values": eta_list,
            "compression_params_list": eta_list,
            "mapping_info": None,
        }

    if codec_name == "identity":
        num_links = len(eta_list)
        return {
            "codec_name": "identity",
            "requested_eta": eta_list,
            "execution_feature_k_values": [1.0] * num_links,
            "compression_params_list": [None] * num_links,
            "mapping_info": None,
        }

    if codec_name == "llmint8":
        if llmint8_mapping_entries is None:
            raise ValueError("llmint8 execution requires llmint8_mapping_entries")
        eta_arr = np.asarray(eta_list, dtype=float)
        passthrough_mask = eta_arr >= 1.0 - 1e-12
        if bool(np.all(passthrough_mask)):
            num_links = len(eta_list)
            return {
                "codec_name": "identity",
                "requested_eta": eta_list,
                "execution_feature_k_values": [1.0] * num_links,
                "compression_params_list": [None] * num_links,
                "mapping_info": {
                    "requested_eta": eta_list,
                    "matched_feature_k_values": [1.0] * num_links,
                    "matched_outlier_values": [None] * num_links,
                    "match_distance_l2": 0.0,
                    "passthrough_mask": [True] * num_links,
                    "identity_fallback": True,
                },
            }

        match = nearest_llmint8_codec_config(
            eta_list,
            llmint8_mapping_entries,
            active_mask=(~passthrough_mask),
        )
        execution_feature_k_values = _as_float_list(match["matched_feature_k_values"])
        matched_outlier_values = _as_float_list(match["matched_outlier_values"])
        compression_params_list = []
        for idx, outlier in enumerate(matched_outlier_values):
            if bool(passthrough_mask[idx]):
                execution_feature_k_values[idx] = 1.0
                compression_params_list.append(None)
            else:
                compression_params_list.append(
                    [float(outlier), str(outlier_precision), str(regular_precision)]
                )
        match = dict(match)
        match["matched_feature_k_values"] = _as_float_list(execution_feature_k_values)
        match["matched_outlier_values"] = [
            (None if bool(passthrough_mask[idx]) else float(matched_outlier_values[idx]))
            for idx in range(len(matched_outlier_values))
        ]
        match["passthrough_mask"] = [bool(item) for item in passthrough_mask.tolist()]
        compression_params_list = [
            item
            for item in compression_params_list
        ]
        return {
            "codec_name": "llmint8",
            "requested_eta": eta_list,
            "execution_feature_k_values": _as_float_list(execution_feature_k_values),
            "compression_params_list": list(compression_params_list),
            "mapping_info": match,
        }

    raise ValueError("Unsupported codec_name '{}'".format(codec_name))
