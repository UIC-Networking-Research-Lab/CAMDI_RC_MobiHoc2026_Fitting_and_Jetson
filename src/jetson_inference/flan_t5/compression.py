"""Activation payloads, codec profiles and eta execution plans."""

import numpy as np

from collections import OrderedDict
from jetson_inference.common.compressors import get_compressor
from jetson_inference.common.llmint8_eta_mapping import (
    resolve_codec_execution_plan,
    validate_llmint8_mapping_entries,
)
from jetson_inference.flan_t5.config import (
    K_LEVELS,
    LLMINT8_ESTIMATOR_PATH,
    LLMINT8_MAPPING_PATH,
    LLMINT8_OUTLIER_PRECISION,
    LLMINT8_POLICY_NAME,
    LLMINT8_REGULAR_PRECISION,
    NUM_TRANSFER_POINTS,
    QUANTIZATION_ESTIMATOR_PATH,
    TOPK_ESTIMATOR_PATH,
    _load_json,
)


def _identity_activation_payload(tensor_cpu):
    return {"mode": "identity", "tensor": tensor_cpu}

def build_activation_payload(tensor, compression_param, compressor_name, feature_k_value):
    tensor_cpu = tensor.detach().cpu()
    original_bytes = int(tensor_cpu.numel() * tensor_cpu.element_size())
    if compressor_name in ("identity", None) or compression_param is None:
        payload = _identity_activation_payload(tensor_cpu)
        compressed_bytes = original_bytes
    else:
        compressor = get_compressor(compressor_name)
        compressed = compressor.compress(tensor_cpu, compression_param)
        compressed_bytes = None
        if hasattr(compressor, "get_compressed_size"):
            try:
                compressed_bytes = int(compressor.get_compressed_size(compressed))
            except Exception:
                compressed_bytes = None
        if compressed_bytes is None:
            compressed_bytes = int(compressed.get("compressed_bytes", original_bytes))
        payload = {
            "mode": "compressed",
            "compressor_name": compressor_name,
            "payload": compressed,
        }
    ratio = float(compressed_bytes) / float(original_bytes) if original_bytes > 0 else 1.0
    return payload, {
        "original_bytes": original_bytes,
        "compressed_bytes": compressed_bytes,
        "compression_ratio": ratio,
        "k_value": float(feature_k_value),
    }

def restore_activation_payload(payload, device):
    mode = payload.get("mode", "compressed")
    if mode == "identity":
        restored = payload["tensor"]
    else:
        compressor = get_compressor(payload["compressor_name"])
        restored = compressor.decompress(payload["payload"])
    return restored.to(device)

def _load_llmint8_mapping_entries(mapping_path, llm_policy, outlier_precision, regular_precision):
    payload = _load_json(mapping_path)
    policies = payload.get("policies", [])
    for policy in policies:
        if str(policy.get("llm_policy", "")).strip().lower() != str(llm_policy).strip().lower():
            continue
        if str(policy.get("outlier_precision", "")).strip().lower() != str(outlier_precision).strip().lower():
            continue
        if str(policy.get("regular_precision", "")).strip().lower() != str(regular_precision).strip().lower():
            continue
        entries = policy.get("llmint8_eta_to_codec_mapping", {}).get("entries", [])
        if entries:
            return validate_llmint8_mapping_entries(entries, num_links=NUM_TRANSFER_POINTS)
    for policy in policies:
        entries = policy.get("llmint8_eta_to_codec_mapping", {}).get("entries", [])
        if entries:
            return validate_llmint8_mapping_entries(entries, num_links=NUM_TRANSFER_POINTS)
    raise ValueError("No llmint8 mapping entries found in {}".format(mapping_path))

def _eta_min_from_mapping_entries(entries):
    feature_matrix = np.asarray([entry["feature_k_values"] for entry in entries], dtype=float)
    return np.min(feature_matrix, axis=0)

def _normalize_profile_name(name):
    normalized = str(name).strip().lower().replace("-", "_")
    aliases = {
        "topk": "topk",
        "quant": "quantization",
        "quantization": "quantization",
        "llmint8": "llmint8_{}".format(LLMINT8_POLICY_NAME),
        "llm_int8": "llmint8_{}".format(LLMINT8_POLICY_NAME),
        "llmint8_{}".format(LLMINT8_POLICY_NAME): "llmint8_{}".format(LLMINT8_POLICY_NAME),
    }
    return aliases.get(normalized, normalized)

def _codec_profiles(profile_overrides=None):
    profile_overrides = profile_overrides or {}
    llmint8_mapping_path = str(profile_overrides.get("llmint8_mapping_path", LLMINT8_MAPPING_PATH))
    llmint8_entries = _load_llmint8_mapping_entries(
        mapping_path=llmint8_mapping_path,
        llm_policy=LLMINT8_POLICY_NAME,
        outlier_precision=LLMINT8_OUTLIER_PRECISION,
        regular_precision=LLMINT8_REGULAR_PRECISION,
    )
    topk_estimator_path = str(profile_overrides.get("topk_estimator_path", TOPK_ESTIMATOR_PATH))
    quantization_estimator_path = str(profile_overrides.get("quantization_estimator_path", QUANTIZATION_ESTIMATOR_PATH))
    llmint8_estimator_path = str(profile_overrides.get("llmint8_estimator_path", LLMINT8_ESTIMATOR_PATH))
    return OrderedDict(
        [
            (
                "topk",
                {
                    "display_name": "TopK",
                    "codec_name": "topk",
                    "compressor_name": "topk",
                    "estimator_path": topk_estimator_path,
                    "eta_min": np.asarray([min(K_LEVELS)] * NUM_TRANSFER_POINTS, dtype=float),
                    "llmint8_mapping_entries": [],
                    "outlier_precision": None,
                    "regular_precision": None,
                },
            ),
            (
                "quantization",
                {
                    "display_name": "Quantization",
                    "codec_name": "quantization",
                    "compressor_name": "quantization",
                    "estimator_path": quantization_estimator_path,
                    "eta_min": np.asarray([0.125] * NUM_TRANSFER_POINTS, dtype=float),
                    "llmint8_mapping_entries": [],
                    "outlier_precision": None,
                    "regular_precision": None,
                },
            ),
            (
                "llmint8_{}".format(LLMINT8_POLICY_NAME),
                {
                    "display_name": "LLMInt8_{}".format(LLMINT8_POLICY_NAME),
                    "codec_name": "llmint8",
                    "compressor_name": "llmint8",
                    "estimator_path": llmint8_estimator_path,
                    "eta_min": _eta_min_from_mapping_entries(llmint8_entries),
                    "llmint8_mapping_entries": llmint8_entries,
                    "outlier_precision": LLMINT8_OUTLIER_PRECISION,
                    "regular_precision": LLMINT8_REGULAR_PRECISION,
                },
            ),
        ]
    )

def _build_execution_plan(profile_spec, requested_eta):
    requested_eta = [float(item) for item in requested_eta]
    if profile_spec["codec_name"] == "topk":
        return resolve_codec_execution_plan(codec_name="topk", eta=requested_eta)
    if profile_spec["codec_name"] == "quantization":
        return resolve_codec_execution_plan(codec_name="quantization", eta=requested_eta)
    return resolve_codec_execution_plan(
        codec_name="llmint8",
        eta=requested_eta,
        outlier_precision=profile_spec["outlier_precision"],
        regular_precision=profile_spec["regular_precision"],
        llmint8_mapping_entries=profile_spec["llmint8_mapping_entries"],
    )
