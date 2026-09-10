"""online optimization, baseline policies and dual updates."""

import logging
import numpy as np

from jetson_inference.flan_t5.config import (
    DEFAULT_EPSILON,
    DEFAULT_MAX_LAMBDA,
    NUM_TRANSFER_POINTS,
)
from scipy.optimize import minimize


def compute_pipeline_stage_delays(eta, a_t, tau_t, c_t):
    stage_delays = []
    for idx in range(NUM_TRANSFER_POINTS):
        comm = float(a_t[idx]) * float(eta[idx]) / float(c_t[idx]) if float(c_t[idx]) > 0 else float("inf")
        stage_delays.append(max(float(tau_t[idx]), comm))
    stage_delays.append(float(tau_t[NUM_TRANSFER_POINTS]))
    return stage_delays

def compute_pipeline_delay(eta, a_t, tau_t, c_t):
    return max(compute_pipeline_stage_delays(eta, a_t, tau_t, c_t))

def _sanitize_lambda_value(value, epsilon, upper=DEFAULT_MAX_LAMBDA):
    try:
        numeric = float(value)
    except Exception:
        logging.warning("[control] lambda sanitize: invalid value=%r, fallback to epsilon=%.6g", value, float(epsilon))
        numeric = float(epsilon)
    if not np.isfinite(numeric):
        logging.warning("[control] lambda sanitize: non-finite value=%r, clipping to upper=%.6g", value, float(upper))
        numeric = float(upper)
    clipped = float(min(max(float(epsilon), numeric), float(upper)))
    if clipped != numeric:
        logging.warning(
            "[control] lambda sanitize: clipping value=%.6g into [%.6g, %.6g] -> %.6g",
            float(numeric),
            float(epsilon),
            float(upper),
            float(clipped),
        )
    return clipped

def _finite_mean(values, fallback):
    raw = np.asarray(list(values), dtype=float)
    arr = raw[np.isfinite(raw)]
    if arr.size != raw.size:
        logging.warning(
            "[control] finite mean: filtered %d non-finite values before averaging",
            int(raw.size - arr.size),
        )
    if arr.size == 0:
        logging.warning("[control] finite mean: no finite values left, fallback=%.6g", float(fallback))
        return float(fallback)
    return float(np.mean(arr))

class BaseOnlinePolicy(object):
    def __init__(self, policy_key, display_name, eta_min):
        self.policy_key = str(policy_key)
        self.policy_name = str(display_name)
        self.eta_min = np.asarray(eta_min, dtype=float)
        self.lambda_t = 0.0
        self.mu = None

    def select_eta(self, a_t, tau_t, c_hat, deadline):
        raise NotImplementedError

    def update_dual(self, actual_delay, deadline):
        return None

    def initialize_from_channel_estimator(self, channel_estimator):
        return None

    def observe_channel(self, tp_stats):
        return None

class EstimatedCSISingleBaselinePolicy(BaseOnlinePolicy):
    def __init__(self, policy_key, display_name, eta_min, acc_model=None, mode="last", window_size=5):
        super().__init__(policy_key, display_name, eta_min)
        self.acc_model = acc_model
        self.mode = str(mode)
        self.window_size = max(1, int(window_size))
        self.link_history = [[] for _ in range(NUM_TRANSFER_POINTS)]

    def initialize_from_channel_estimator(self, channel_estimator):
        links = getattr(channel_estimator, "links", None)
        if links is None:
            return
        histories = []
        for link in list(links)[:NUM_TRANSFER_POINTS]:
            histories.append(
                [float(x) for x in getattr(link, "history", []) if np.isfinite(x) and float(x) > 0.0]
            )
        if len(histories) == NUM_TRANSFER_POINTS:
            self.link_history = histories

    def observe_channel(self, tp_stats):
        for idx, (bytes_sent, elapsed_sec) in enumerate(tp_stats[:NUM_TRANSFER_POINTS]):
            if elapsed_sec > 0.0 and bytes_sent > 0.0:
                throughput = float(bytes_sent) / float(elapsed_sec)
                if np.isfinite(throughput) and throughput > 0.0:
                    self.link_history[idx].append(float(throughput))

    def _estimate_c_hat(self, fallback):
        fallback_arr = np.asarray(fallback, dtype=float).reshape(NUM_TRANSFER_POINTS)
        estimates = []
        for idx in range(NUM_TRANSFER_POINTS):
            history = [float(x) for x in self.link_history[idx] if np.isfinite(x) and float(x) > 0.0]
            if not history:
                estimates.append(max(float(fallback_arr[idx]), 1e-9))
                continue
            if self.mode == "last":
                estimates.append(float(history[-1]))
            elif self.mode == "min":
                estimates.append(float(np.min(history)))
            else:
                estimates.append(float(np.mean(history[-self.window_size :])))
        return np.maximum(np.asarray(estimates, dtype=float), 1e-9)

    def select_eta(self, a_t, tau_t, c_hat, deadline):
        c_hat_est = self._estimate_c_hat(c_hat)
        deadline = float(deadline)
        target_rate_hz = (1.0 / deadline) if deadline > 0.0 else np.nan
        eta = np.divide(
            c_hat_est,
            target_rate_hz * np.asarray(a_t, dtype=float),
            out=np.ones_like(self.eta_min, dtype=float),
            where=(target_rate_hz * np.asarray(a_t, dtype=float)) > 0.0,
        )
        eta = np.minimum(1.0, eta)
        eta = np.maximum(eta, self.eta_min)
        pred_delay = compute_pipeline_delay(eta, a_t, tau_t, c_hat_est)
        pred_acc = self.acc_model.predict(eta) if self.acc_model is not None else None
        return {
            "requested_eta": eta,
            "predicted_delay": pred_delay,
            "predicted_accuracy": pred_acc,
            "solver_z": pred_delay,
        }

class MyopicSinglePolicy(EstimatedCSISingleBaselinePolicy):
    def __init__(self, eta_min, acc_model=None):
        super().__init__("myopic_baseline", "No-CSI Baseline: myopic", eta_min, acc_model=acc_model, mode="last")

class ConservativeSinglePolicy(EstimatedCSISingleBaselinePolicy):
    def __init__(self, eta_min, acc_model=None):
        super().__init__("conservative_baseline", "No-CSI Baseline: conservative", eta_min, acc_model=acc_model, mode="min")

class MovingAverageSinglePolicy(EstimatedCSISingleBaselinePolicy):
    def __init__(self, eta_min, acc_model=None, window_size=5):
        super().__init__(
            "moving_average_baseline",
            "No-CSI Baseline: moving average",
            eta_min,
            acc_model=acc_model,
            mode="moving_average",
            window_size=window_size,
        )

class NoCompressionBaselinePolicy(BaseOnlinePolicy):
    def __init__(self, eta_min):
        super().__init__("no_compression_baseline", "No Compression Baseline", eta_min)

    def select_eta(self, a_t, tau_t, c_hat, deadline):
        eta = np.ones_like(self.eta_min, dtype=float)
        pred_delay = compute_pipeline_delay(eta, a_t, tau_t, c_hat)
        return {
            "requested_eta": eta,
            "predicted_delay": pred_delay,
            "predicted_accuracy": None,
            "solver_z": pred_delay,
        }

class MaxCompressionBaselinePolicy(BaseOnlinePolicy):
    def __init__(self, eta_min):
        super().__init__("max_compression_baseline", "Max Compression Baseline", eta_min)

    def select_eta(self, a_t, tau_t, c_hat, deadline):
        eta = self.eta_min.copy()
        pred_delay = compute_pipeline_delay(eta, a_t, tau_t, c_hat)
        return {
            "requested_eta": eta,
            "predicted_delay": pred_delay,
            "predicted_accuracy": None,
            "solver_z": pred_delay,
        }

class NoCSISinglePolicy(BaseOnlinePolicy):
    def __init__(self, eta_min, acc_model, mu, epsilon=DEFAULT_EPSILON):
        display_name = "Non-CSI (mu={})".format("{:.6g}".format(float(mu)))
        super().__init__("no_csi_single", display_name, eta_min)
        self.acc_model = acc_model
        self.mu = float(mu)
        self.epsilon = float(epsilon)
        self.lambda_t = float(epsilon)
        self._last_solution = None

    def _objective(self, x):
        eta = x[:-1]
        z = float(x[-1])
        acc = self.acc_model.predict(eta)
        grad_acc = self.acc_model.gradient(eta)
        obj = -acc + self.mu * self.lambda_t * z
        grad = np.zeros_like(x)
        grad[:-1] = -grad_acc
        grad[-1] = self.mu * self.lambda_t
        return obj, grad

    def select_eta(self, a_t, tau_t, c_hat, deadline):
        self.lambda_t = _sanitize_lambda_value(self.lambda_t, self.epsilon)
        c_hat_safe = np.asarray(c_hat, dtype=float).copy()
        invalid_mask = (~np.isfinite(c_hat_safe)) | (c_hat_safe <= 0.0)
        if np.any(invalid_mask):
            logging.warning(
                "[policy=%s] invalid c_hat detected: raw=%s; replacing invalid entries with 1e-9",
                self.policy_name,
                [float(x) if np.isfinite(x) else str(x) for x in np.asarray(c_hat, dtype=float)],
            )
            c_hat_safe[invalid_mask] = 1e-9
        lower_z = max(float(np.max(tau_t)), 0.0)
        bounds = [(float(self.eta_min[i]), 1.0) for i in range(NUM_TRANSFER_POINTS)] + [(lower_z, None)]
        if self._last_solution is not None and len(self._last_solution) == len(bounds):
            x0 = np.asarray(self._last_solution, dtype=float).copy()
            for idx, (lower, upper) in enumerate(bounds):
                x0[idx] = max(lower, x0[idx])
                if upper is not None:
                    x0[idx] = min(upper, x0[idx])
        else:
            x0 = np.array([bounds[idx][0] for idx in range(len(bounds))], dtype=float)
        constraints = []
        for idx in range(NUM_TRANSFER_POINTS):
            beta = float(a_t[idx]) / float(c_hat_safe[idx]) if float(c_hat_safe[idx]) > 0 else 1e12

            def make_constraint(local_idx=idx, local_beta=beta):
                return {
                    "type": "ineq",
                    "fun": lambda x: float(x[-1]) - local_beta * float(x[local_idx]),
                    "jac": lambda x: np.array(
                        [
                            (-local_beta if j == local_idx else (1.0 if j == len(x) - 1 else 0.0))
                            for j in range(len(x))
                        ],
                        dtype=float,
                    ),
                }

            constraints.append(make_constraint())
        result = minimize(
            fun=self._objective,
            x0=x0,
            method="SLSQP",
            jac=True,
            bounds=bounds,
            constraints=constraints,
        )
        if not result.success:
            logging.warning(
                "[policy=%s] SLSQP failed: success=%s status=%s message=%s mu=%.6g lambda=%.6g deadline=%.6g "
                "lower_z=%.6g x0=%s c_hat=%s a=%s tau=%s",
                self.policy_name,
                result.success,
                getattr(result, "status", None),
                getattr(result, "message", ""),
                float(self.mu),
                float(self.lambda_t),
                float(deadline),
                float(lower_z),
                [float(item) for item in np.asarray(x0, dtype=float)],
                [float(item) for item in np.asarray(c_hat_safe, dtype=float)],
                [float(item) for item in np.asarray(a_t, dtype=float)],
                [float(item) for item in np.asarray(tau_t, dtype=float)],
            )
        use_result = bool(result.success) and getattr(result, "x", None) is not None and np.all(
            np.isfinite(np.asarray(result.x, dtype=float))
        )
        if not use_result:
            logging.warning(
                "[policy=%s] Falling back to feasible point due to invalid solver output: raw_x=%s",
                self.policy_name,
                None if getattr(result, "x", None) is None else [float(item) for item in np.asarray(result.x, dtype=float)],
            )
        selected_x = np.asarray(result.x, dtype=float) if use_result else x0
        eta = np.asarray(selected_x[:-1], dtype=float)
        eta = np.clip(eta, self.eta_min, 1.0)
        z = float(max(lower_z, selected_x[-1]))
        self._last_solution = np.concatenate([eta, np.array([z], dtype=float)])
        return {
            "requested_eta": eta,
            "predicted_delay": compute_pipeline_delay(eta, a_t, tau_t, c_hat_safe),
            "predicted_accuracy": self.acc_model.predict(eta),
            "solver_z": z,
        }

    def update_dual(self, actual_delay, deadline):
        actual_delay = float(actual_delay)
        deadline = float(deadline)
        if not np.isfinite(actual_delay) or not np.isfinite(deadline) or deadline <= 0.0:
            logging.warning(
                "[policy=%s] skipping dual update due to invalid actual_delay=%.6g deadline=%.6g",
                self.policy_name,
                float(actual_delay),
                float(deadline),
            )
            self.lambda_t = _sanitize_lambda_value(self.lambda_t, self.epsilon)
            return
        updated = float(self.lambda_t) + actual_delay - deadline
        self.lambda_t = _sanitize_lambda_value(updated, self.epsilon)
