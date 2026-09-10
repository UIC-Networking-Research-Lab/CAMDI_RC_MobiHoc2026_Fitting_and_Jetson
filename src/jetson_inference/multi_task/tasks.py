"""mixed-task construction, accuracy adapters and execution plans."""

import copy
import numpy as np

from jetson_inference.common.llmint8_eta_mapping import (
    resolve_codec_execution_plan,
    validate_llmint8_mapping_entries,
)
from jetson_inference.flan_t5.config import (
    DEFAULT_MAX_INPUT_LENGTH as FLAN_DEFAULT_MAX_INPUT_LENGTH,
    DEFAULT_MODEL_NAME as FLAN_DEFAULT_MODEL_NAME,
    DEFAULT_NEGATIVE_TOKEN as FLAN_DEFAULT_NEGATIVE_TOKEN,
    DEFAULT_POSITIVE_TOKEN as FLAN_DEFAULT_POSITIVE_TOKEN,
    DEFAULT_PROMPT_TEMPLATE as FLAN_DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_SPLIT as FLAN_DEFAULT_SPLIT,
)
from jetson_inference.flan_t5.estimation import FlanT5FastAccuracyEvaluator
from jetson_inference.flan_t5.model import (
    FlanT5PartitionFactory,
    build_prompt as build_flan_prompt,
    load_sst2_samples,
    resolve_single_token_verbalizers,
)
from jetson_inference.multi_task.config import (
    DEFAULT_STEIN_FAST_MAX_BATCHES,
    DEFAULT_STEIN_FAST_MAX_SAMPLES,
    DEFAULT_STEIN_N,
    DEFAULT_STEIN_SIGMA,
    NUM_TRANSFER_POINTS,
    _load_json,
    _resolve_path,
)
from jetson_inference.multi_task.state import RuntimeTaskDef
from jetson_inference.resnet.config import (
    DEFAULT_DATA_ROOT as RESNET_DEFAULT_DATA_ROOT,
    DEFAULT_DOWNLOAD_DATA as RESNET_DEFAULT_DOWNLOAD_DATA,
    MODEL_CHECKPOINT as RESNET_CHECKPOINT_PATH,
)
from jetson_inference.resnet.estimation import (
    Poly3AccuracyAdapter,
    ResNetFastAccuracyEvaluator,
    SteinAccuracyEstimatorAdapter as SteinAccuracyEstimatorAdapter,
)
from jetson_inference.resnet.model import load_cifar10_batches
from typing import Sequence


def _load_sst2_batches(factory, dataset_path, split, max_samples, prompt_template):
    samples = load_sst2_samples(dataset_path=dataset_path, split=split)
    if max_samples is not None and int(max_samples) > 0:
        samples = samples[: int(max_samples)]
    batches = []
    for idx, sample in enumerate(samples):
        prompt_text = build_flan_prompt(sample.sentence, prompt_template)
        encoded = factory.encode_prompt(prompt_text)
        batches.append(
            {
                "batch_idx": idx,
                "sample_id": sample.sample_id,
                "label": int(sample.label),
                "prompt_text": prompt_text,
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }
        )
    return batches

def _load_llmint8_mapping_entries(base_dir, source_path):
    if not source_path:
        return []
    payload = _load_json(_resolve_path(base_dir, source_path))
    entries = payload.get("policies", [{}])[0].get("llmint8_eta_to_codec_mapping", {}).get("entries", payload.get("entries", []))
    return validate_llmint8_mapping_entries(entries, num_links=NUM_TRANSFER_POINTS)

def _build_execution_plan_for_codec(codec_name, requested_eta, outlier_precision=None, regular_precision=None, llmint8_mapping_entries=None):
    if str(codec_name) == "topk":
        return resolve_codec_execution_plan(codec_name="topk", eta=requested_eta)
    if str(codec_name) == "quantization":
        return resolve_codec_execution_plan(codec_name="quantization", eta=requested_eta)
    return resolve_codec_execution_plan(
        codec_name="llmint8",
        eta=requested_eta,
        outlier_precision=outlier_precision or "fp16",
        regular_precision=regular_precision or "int8",
        llmint8_mapping_entries=llmint8_mapping_entries or [],
    )

def _build_accuracy_estimator(
    task_family,
    estimator_path,
    eta_min,
    accuracy_estimator_mode,
    codec_name,
    device,
    task_cfg,
    batches,
    outlier_precision=None,
    regular_precision=None,
    llmint8_entries=None,
    stein_sigma=DEFAULT_STEIN_SIGMA,
    stein_N=DEFAULT_STEIN_N,
    stein_fast_max_batches=DEFAULT_STEIN_FAST_MAX_BATCHES,
    stein_fast_max_samples=DEFAULT_STEIN_FAST_MAX_SAMPLES,
):
    if str(accuracy_estimator_mode) == "fitting_model":
        return Poly3AccuracyAdapter(estimator_path)
    if str(task_family) == "resnet":
        fast_batches = list(batches[: max(1, int(stein_fast_max_batches))])
        evaluator = ResNetFastAccuracyEvaluator(
            checkpoint_path=str(task_cfg.get("checkpoint_path", RESNET_CHECKPOINT_PATH)),
            device=device,
            batches=fast_batches,
        )

        def accuracy_callable(eta_vec):
            execution_plan = _build_execution_plan_for_codec(
                codec_name=codec_name,
                requested_eta=eta_vec,
                outlier_precision=outlier_precision,
                regular_precision=regular_precision,
                llmint8_mapping_entries=llmint8_entries,
            )
            return float(evaluator.evaluate(execution_plan))

    elif str(task_family) == "flan_t5":
        fast_batches = list(batches[: max(1, int(stein_fast_max_samples))])
        evaluator = FlanT5FastAccuracyEvaluator(
            model_name=str(task_cfg.get("model_name", FLAN_DEFAULT_MODEL_NAME)),
            device=device,
            max_input_length=int(task_cfg.get("max_input_length", FLAN_DEFAULT_MAX_INPUT_LENGTH)),
            positive_token=str(task_cfg.get("positive_token", FLAN_DEFAULT_POSITIVE_TOKEN)),
            negative_token=str(task_cfg.get("negative_token", FLAN_DEFAULT_NEGATIVE_TOKEN)),
            batches=fast_batches,
        )

        def accuracy_callable(eta_vec):
            execution_plan = _build_execution_plan_for_codec(
                codec_name=codec_name,
                requested_eta=eta_vec,
                outlier_precision=outlier_precision,
                regular_precision=regular_precision,
                llmint8_mapping_entries=llmint8_entries,
            )
            return float(evaluator.evaluate(execution_plan))

    else:
        raise ValueError("Unsupported task family '{}'".format(task_family))
    return SteinAccuracyEstimatorAdapter(
        accuracy_callable=accuracy_callable,
        num_links=len(eta_min),
        eta_min=np.asarray(eta_min, dtype=float),
        sigma=float(stein_sigma),
        N=int(stein_N),
    )

def _build_task_defs(
    manifest_dir,
    manifest,
    codec_name,
    device,
    accuracy_estimator_mode="fitting_model",
    stein_sigma=DEFAULT_STEIN_SIGMA,
    stein_N=DEFAULT_STEIN_N,
    stein_fast_max_batches=DEFAULT_STEIN_FAST_MAX_BATCHES,
    stein_fast_max_samples=DEFAULT_STEIN_FAST_MAX_SAMPLES,
):
    tasks = []
    flan_factory_cache = {}
    for task_cfg in manifest["tasks"]:
        task_family = str(task_cfg["family"]).lower()
        profile_cfg = copy.deepcopy(task_cfg["codec_variants"][codec_name])
        estimator_path = _resolve_path(manifest_dir, profile_cfg["accuracy_model"]["path"])
        eta_min = np.asarray(task_cfg.get("eta_min", [0.125, 0.125, 0.125]), dtype=float)
        logical_task_id = int(task_cfg["task_id"])
        weight = float(task_cfg.get("weight", 1.0))
        batch_size = int(task_cfg.get("batch_size", 1 if task_family == "flan_t5" else 100))
        target_rate_hz = float(task_cfg["target_rate_hz"])
        outlier_precision = str(profile_cfg.get("outlier_precision", task_cfg.get("outlier_precision", "fp16"))) if codec_name == "llmint8" else None
        regular_precision = str(profile_cfg.get("regular_precision", task_cfg.get("regular_precision", "int8"))) if codec_name == "llmint8" else None
        llmint8_entries = _load_llmint8_mapping_entries(manifest_dir, profile_cfg.get("llmint8_mapping_source")) if codec_name == "llmint8" else []

        if task_family == "resnet":
            batches = load_cifar10_batches(
                batch_size=batch_size,
                max_batches=int(task_cfg.get("max_items", 100)),
                data_root=str(task_cfg.get("data_root", RESNET_DEFAULT_DATA_ROOT)),
                download=bool(task_cfg.get("download_data", RESNET_DEFAULT_DOWNLOAD_DATA)),
            )
            extra = {"checkpoint_path": str(task_cfg.get("checkpoint_path", RESNET_CHECKPOINT_PATH))}
        elif task_family == "flan_t5":
            factory_key = (
                str(task_cfg.get("model_name", FLAN_DEFAULT_MODEL_NAME)),
                int(task_cfg.get("max_input_length", FLAN_DEFAULT_MAX_INPUT_LENGTH)),
            )
            if factory_key not in flan_factory_cache:
                flan_factory_cache[factory_key] = FlanT5PartitionFactory(
                    model_name=factory_key[0],
                    device=device,
                    max_input_length=factory_key[1],
                )
            flan_factory = flan_factory_cache[factory_key]
            batches = _load_sst2_batches(
                factory=flan_factory,
                dataset_path=_resolve_path(manifest_dir, task_cfg.get("dataset_path")),
                split=str(task_cfg.get("split", FLAN_DEFAULT_SPLIT)),
                max_samples=int(task_cfg.get("max_items", 100)),
                prompt_template=str(task_cfg.get("prompt_template", FLAN_DEFAULT_PROMPT_TEMPLATE)),
            )
            verbalizers = resolve_single_token_verbalizers(
                flan_factory.tokenizer,
                positive_text=str(task_cfg.get("positive_token", FLAN_DEFAULT_POSITIVE_TOKEN)),
                negative_text=str(task_cfg.get("negative_token", FLAN_DEFAULT_NEGATIVE_TOKEN)),
            )
            extra = {
                "model_name": str(task_cfg.get("model_name", FLAN_DEFAULT_MODEL_NAME)),
                "max_input_length": int(task_cfg.get("max_input_length", FLAN_DEFAULT_MAX_INPUT_LENGTH)),
                "decoder_start_token_id": int(flan_factory.decoder_start_token_id),
                "verbalizers": verbalizers,
            }
        else:
            raise ValueError("Unsupported task family '{}'".format(task_family))

        estimator = _build_accuracy_estimator(
            task_family=task_family,
            estimator_path=estimator_path,
            eta_min=eta_min,
            accuracy_estimator_mode=accuracy_estimator_mode,
            codec_name=codec_name,
            device=device,
            task_cfg=task_cfg,
            batches=batches,
            outlier_precision=outlier_precision,
            regular_precision=regular_precision,
            llmint8_entries=llmint8_entries,
            stein_sigma=stein_sigma,
            stein_N=stein_N,
            stein_fast_max_batches=stein_fast_max_batches,
            stein_fast_max_samples=stein_fast_max_samples,
        )

        tasks.append(
            RuntimeTaskDef(
                logical_task_id=logical_task_id,
                family=task_family,
                name=str(task_cfg.get("name", "task_{}".format(logical_task_id))),
                model=str(task_cfg.get("model", task_family)),
                dataset=str(task_cfg.get("dataset", "")),
                weight=weight,
                eta_min=eta_min,
                target_rate_hz=target_rate_hz,
                batch_size=batch_size,
                estimator=estimator,
                codec_name=str(codec_name),
                outlier_precision=outlier_precision,
                regular_precision=regular_precision,
                llmint8_mapping_entries=llmint8_entries,
                warmup_items=list(copy.deepcopy(batches)),
                experiment_items=list(copy.deepcopy(batches)),
                extra=extra,
            )
        )
    return tasks

def _build_worker_task_defs(manifest):
    tasks = []
    for task_cfg in manifest["tasks"]:
        task_family = str(task_cfg["family"]).lower()
        logical_task_id = int(task_cfg["task_id"])
        eta_min = np.asarray(task_cfg.get("eta_min", [0.125, 0.125, 0.125]), dtype=float)
        batch_size = int(task_cfg.get("batch_size", 1 if task_family == "flan_t5" else 100))
        extra = {}
        if task_family == "resnet":
            extra["checkpoint_path"] = str(task_cfg.get("checkpoint_path", RESNET_CHECKPOINT_PATH))
        elif task_family == "flan_t5":
            extra = {
                "model_name": str(task_cfg.get("model_name", FLAN_DEFAULT_MODEL_NAME)),
                "max_input_length": int(task_cfg.get("max_input_length", FLAN_DEFAULT_MAX_INPUT_LENGTH)),
                "decoder_start_token_id": 0,
                "positive_token": str(task_cfg.get("positive_token", FLAN_DEFAULT_POSITIVE_TOKEN)),
                "negative_token": str(task_cfg.get("negative_token", FLAN_DEFAULT_NEGATIVE_TOKEN)),
            }
        else:
            raise ValueError("Unsupported task family '{}'".format(task_family))
        tasks.append(
            RuntimeTaskDef(
                logical_task_id=logical_task_id,
                family=task_family,
                name=str(task_cfg.get("name", "task_{}".format(logical_task_id))),
                model=str(task_cfg.get("model", task_family)),
                dataset=str(task_cfg.get("dataset", "")),
                weight=float(task_cfg.get("weight", 1.0)),
                eta_min=eta_min,
                target_rate_hz=float(task_cfg.get("target_rate_hz", 1.0)),
                batch_size=batch_size,
                estimator=None,
                codec_name="topk",
                outlier_precision=None,
                regular_precision=None,
                llmint8_mapping_entries=[],
                warmup_items=[],
                experiment_items=[],
                extra=extra,
            )
        )
    return tasks

def _build_execution_plan(task_def: RuntimeTaskDef, requested_eta: Sequence[float]):
    return _build_execution_plan_for_codec(
        codec_name=task_def.codec_name,
        requested_eta=requested_eta,
        outlier_precision=task_def.outlier_precision,
        regular_precision=task_def.regular_precision,
        llmint8_mapping_entries=task_def.llmint8_mapping_entries,
    )
