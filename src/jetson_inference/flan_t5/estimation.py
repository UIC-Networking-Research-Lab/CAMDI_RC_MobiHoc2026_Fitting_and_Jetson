"""Accuracy and channel estimation."""

import logging
import numpy as np
import os
import pickle
import torch

from jetson_inference.common.channel_estimator import ChannelEstimator
from jetson_inference.flan_t5.compression import (
    _build_execution_plan,
    build_activation_payload,
    restore_activation_payload,
)
from jetson_inference.flan_t5.config import (
    DEFAULT_CHANNEL_MODEL_TYPE,
    DEFAULT_CHANNEL_UPDATE_MODE,
    DEFAULT_CHANNEL_WINDOW_SIZE,
    DEFAULT_STEIN_N,
    DEFAULT_STEIN_SIGMA,
    NUM_TRANSFER_POINTS,
)
from jetson_inference.flan_t5.model import (
    FlanT5PartitionFactory,
    resolve_single_token_verbalizers,
)


class AccuracyModelAdapter:
    mode_name = "fitting_model"

    def __init__(self, model_path):
        with open(model_path, "rb") as handle:
            state = pickle.load(handle)
        self.model = state["model"]
        self.scaler = state["scaler"]
        self.poly = state.get("poly")
        self.model_type = str(state["model_type"])
        self.n_features = int(state["n_features"])
        if self.model_type not in {"linear_monotonic", "poly2", "poly3"}:
            raise ValueError("Unsupported estimator model_type '{}'".format(self.model_type))

    def predict_raw(self, eta):
        x = np.asarray(eta, dtype=float).reshape(1, -1)
        z = self.scaler.transform(x)
        if self.poly is not None:
            z = self.poly.transform(z)
        return float(self.model.predict(z)[0])

    def predict(self, eta):
        return float(np.clip(self.predict_raw(eta), 0.0, 1.0))

    def gradient(self, eta):
        x = np.asarray(eta, dtype=float).reshape(-1)
        mean = np.asarray(getattr(self.scaler, "mean_", np.zeros(self.n_features)), dtype=float)
        scale = np.asarray(getattr(self.scaler, "scale_", np.ones(self.n_features)), dtype=float)
        scale = np.where(scale == 0.0, 1.0, scale)
        z = (x - mean) / scale
        if self.poly is None:
            coef = np.asarray(self.model.coef_, dtype=float).reshape(-1)
            grad = coef / scale
        else:
            coef = np.asarray(self.model.coef_, dtype=float).reshape(-1)
            powers = np.asarray(self.poly.powers_, dtype=int)
            grad_z = np.zeros(self.n_features, dtype=float)
            for feature_idx in range(self.n_features):
                partial = 0.0
                for term_idx, power_vec in enumerate(powers):
                    exponent = int(power_vec[feature_idx])
                    if exponent == 0:
                        continue
                    term = coef[term_idx] * exponent
                    for dim_idx, dim_power in enumerate(power_vec):
                        p = int(dim_power)
                        if dim_idx == feature_idx:
                            if p - 1 > 0:
                                term *= z[dim_idx] ** (p - 1)
                        else:
                            if p > 0:
                                term *= z[dim_idx] ** p
                    partial += term
                grad_z[feature_idx] = partial
            grad = grad_z / scale
        raw = self.predict_raw(x)
        if raw <= 0.0 or raw >= 1.0:
            return np.zeros_like(grad)
        return grad.astype(float)

class SteinAccuracyEstimatorAdapter:
    mode_name = "stein_estimator"

    def __init__(self, accuracy_callable, num_links, eta_min, sigma=DEFAULT_STEIN_SIGMA, N=DEFAULT_STEIN_N):
        self.accuracy_callable = accuracy_callable
        self.num_links = int(num_links)
        self.eta_min = np.asarray(eta_min, dtype=float).reshape(-1)
        if self.eta_min.shape[0] != self.num_links:
            raise ValueError("Expected {} eta values, got {}".format(self.num_links, self.eta_min.shape[0]))
        self.sigma = float(sigma)
        self.N = int(N)

    def _clamp_eta(self, eta):
        eta_arr = np.asarray(eta, dtype=float).reshape(-1)
        if eta_arr.shape[0] != self.num_links:
            raise ValueError("Expected {} eta values, got {}".format(self.num_links, eta_arr.shape[0]))
        return np.clip(eta_arr, self.eta_min, 1.0).astype(float)

    def predict(self, eta):
        return float(self.accuracy_callable(self._clamp_eta(eta)))

    def gradient(self, eta):
        eta_t = torch.tensor(self._clamp_eta(eta), dtype=torch.float32)

        def f(x):
            eta_np = self._clamp_eta(x.detach().cpu().numpy())
            return torch.tensor(float(self.accuracy_callable(eta_np)), dtype=torch.float32, device=x.device)

        g = torch.zeros_like(eta_t)
        for _ in range(self.N):
            z = torch.randn_like(eta_t)
            g += z * (f(eta_t + self.sigma * z) - f(eta_t - self.sigma * z))
        return (g / (2.0 * self.N * self.sigma)).detach().cpu().numpy().astype(float)

class FlanT5FastAccuracyEvaluator:
    def __init__(self, model_name, device, max_input_length, positive_token, negative_token, batches):
        self.device = device
        self.factory = FlanT5PartitionFactory(model_name=model_name, device=device, max_input_length=max_input_length)
        self.partition_a = self.factory.build_partition_for_node("A")
        self.partition_b = self.factory.build_partition_for_node("B")
        self.partition_c = self.factory.build_partition_for_node("C")
        self.partition_d = self.factory.build_partition_for_node("D")
        self.verbalizers = resolve_single_token_verbalizers(
            self.factory.tokenizer,
            positive_text=positive_token,
            negative_text=negative_token,
        )
        self.batches = list(batches)

    def evaluate(self, execution_plan):
        if not self.batches:
            return 0.0
        compressor_name = str(execution_plan["codec_name"])
        compression_params_list = list(execution_plan["compression_params_list"])
        feature_k_values = list(execution_plan["execution_feature_k_values"])
        total_correct = 0
        total_seen = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.batches):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                hidden, position_bias = self.partition_a(input_ids, attention_mask=attention_mask, position_bias=None)
                payload, _ = build_activation_payload(
                    hidden,
                    compression_param=compression_params_list[0],
                    compressor_name=compressor_name,
                    feature_k_value=feature_k_values[0],
                )
                hidden = restore_activation_payload(payload, self.device)
                hidden, position_bias = self.partition_b(hidden, attention_mask=attention_mask, position_bias=position_bias)
                payload, _ = build_activation_payload(
                    hidden,
                    compression_param=compression_params_list[1],
                    compressor_name=compressor_name,
                    feature_k_value=feature_k_values[1],
                )
                hidden = restore_activation_payload(payload, self.device)
                hidden, _ = self.partition_c(hidden, attention_mask=attention_mask, position_bias=position_bias)
                payload, _ = build_activation_payload(
                    hidden,
                    compression_param=compression_params_list[2],
                    compressor_name=compressor_name,
                    feature_k_value=feature_k_values[2],
                )
                encoder_hidden = restore_activation_payload(payload, self.device)
                task_id = "stein_eval_{}".format(batch_idx)
                decoder_input_ids = torch.tensor([[int(self.factory.decoder_start_token_id)]], dtype=torch.long, device=self.device)
                logits, _, _ = self.partition_d(
                    decoder_input_ids,
                    encoder_hidden,
                    task_id,
                    encoder_attention_mask=attention_mask,
                    position_bias=None,
                    encoder_decoder_position_bias=None,
                )
                if logits.dim() == 3:
                    token_logits = logits[:, -1, :]
                elif logits.dim() == 2:
                    token_logits = logits
                else:
                    raise RuntimeError("Unexpected logits shape {}".format(tuple(logits.shape)))
                positive_id = int(self.verbalizers["positive_id"])
                negative_id = int(self.verbalizers["negative_id"])
                pred_label = 1 if float(token_logits[0, positive_id].item()) >= float(token_logits[0, negative_id].item()) else 0
                self.partition_d.clear_cache(task_id)
                total_correct += int(pred_label == int(batch["label"]))
                total_seen += 1
        return (float(total_correct) / float(total_seen)) if total_seen > 0 else 0.0

def _build_accuracy_estimator(
    *,
    profile_spec,
    estimator_mode,
    model_name,
    device,
    max_input_length,
    positive_token,
    negative_token,
    batches,
    stein_sigma,
    stein_N,
):
    if estimator_mode == "fitting_model":
        estimator_path = profile_spec["estimator_path"]
        if not os.path.exists(estimator_path):
            return None
        return AccuracyModelAdapter(estimator_path)
    if estimator_mode == "stein_estimator":
        evaluator = FlanT5FastAccuracyEvaluator(
            model_name=model_name,
            device=device,
            max_input_length=max_input_length,
            positive_token=positive_token,
            negative_token=negative_token,
            batches=batches,
        )
        return SteinAccuracyEstimatorAdapter(
            accuracy_callable=lambda eta: evaluator.evaluate(_build_execution_plan(profile_spec, eta)),
            num_links=NUM_TRANSFER_POINTS,
            eta_min=profile_spec["eta_min"],
            sigma=stein_sigma,
            N=stein_N,
        )
    raise ValueError("Unsupported accuracy_estimator_mode '{}'".format(estimator_mode))

def build_channel_estimator(
    device="cpu",
    model_type=DEFAULT_CHANNEL_MODEL_TYPE,
    update_mode=DEFAULT_CHANNEL_UPDATE_MODE,
    window_size=DEFAULT_CHANNEL_WINDOW_SIZE,
):
    estimator = ChannelEstimator(
        n_links=NUM_TRANSFER_POINTS,
        model_type=model_type,
        window_size=window_size,
        update_mode=update_mode,
        default_bandwidth=10e6,
        device=device,
    )
    logging.info("Initialized channel estimator: %s", estimator.summary())
    return estimator

def _build_seeded_channel_estimator(warmup_tp_stats, device, model_type, update_mode, window_size):
    estimator = build_channel_estimator(
        device=device,
        model_type=model_type,
        update_mode=update_mode,
        window_size=window_size,
    )
    for tp_stats in warmup_tp_stats:
        estimator.observe_task(tp_stats, refit=False)
    estimator.fit()
    return estimator
