#!/usr/bin/env python3
"""Validate bundled assets and optionally cache external model/data assets."""

import argparse
import json
import os
import pickle
import sys


from jetson_inference.paths import PROJECT_ROOT
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
ESTIMATOR_DIR = os.path.join(MODEL_DIR, "accuracy_estimators")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

REQUIRED_FILES = [
    os.path.join(MODEL_DIR, "resnet56-4bfd9763.th"),
    os.path.join(DATA_DIR, "sst2_validation.jsonl"),
    os.path.join(ESTIMATOR_DIR, "jetson_resnet_3tp_poly3_flex.pkl"),
    os.path.join(ESTIMATOR_DIR, "jetson_resnet_3tp_quantization_poly3_flex.pkl"),
    os.path.join(ESTIMATOR_DIR, "jetson_resnet_3tp_llmint8_fp16_int4_poly3_flex.pkl"),
    os.path.join(ESTIMATOR_DIR, "raw_accuracy_resnet56_llmint8_eta_mapping.json"),
    os.path.join(ESTIMATOR_DIR, "flan_t5_sst2_3tp_topk_poly3_flex.pkl"),
    os.path.join(ESTIMATOR_DIR, "flan_t5_sst2_3tp_quantization_poly3_flex.pkl"),
    os.path.join(ESTIMATOR_DIR, "flan_t5_sst2_3tp_llmint8_fp16_int8_poly3_flex.pkl"),
    os.path.join(ESTIMATOR_DIR, "raw_accuracy_flan_t5_sst2_3tp_llmint8_eta_mapping.json"),
]


def validate_bundled_assets():
    missing = [path for path in REQUIRED_FILES if not os.path.isfile(path)]
    if missing:
        for path in missing:
            print("MISSING: {}".format(path))
        raise SystemExit("Bundled assets are incomplete.")

    for path in REQUIRED_FILES:
        if path.endswith(".pkl"):
            with open(path, "rb") as handle:
                payload = pickle.load(handle)
            if not isinstance(payload, dict):
                raise TypeError("Unexpected estimator payload: {}".format(path))
        elif path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as handle:
                json.load(handle)

    with open(os.path.join(DATA_DIR, "sst2_validation.jsonl"), "r", encoding="utf-8") as handle:
        first = json.loads(next(handle))
    if "sentence" not in first or "label" not in first:
        raise ValueError("SST-2 JSONL must contain sentence and label fields.")
    print("Bundled assets: OK")


def cache_cifar10():
    from torchvision.datasets import CIFAR10

    target = os.path.join(DATA_DIR, "cifar10")
    CIFAR10(root=target, train=False, download=True)
    print("CIFAR-10 cache: {}".format(target))


def cache_flan_t5(model_name):
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    AutoTokenizer.from_pretrained(model_name)
    T5ForConditionalGeneration.from_pretrained(model_name)
    print("Flan-T5 cache: {}".format(model_name))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download/cache CIFAR-10 and Flan-T5 assets.")
    parser.add_argument("--model_name", default=os.environ.get("FLAN_T5_MODEL", "google/flan-t5-base"))
    args = parser.parse_args()

    validate_bundled_assets()
    if args.download:
        cache_cifar10()
        cache_flan_t5(args.model_name)
    print("Asset preparation complete for Python {}".format(sys.version.split()[0]))


if __name__ == "__main__":
    main()
