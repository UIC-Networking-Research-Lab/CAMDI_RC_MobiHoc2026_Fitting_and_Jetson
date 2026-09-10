"""online optimization, baseline policies and dual updates."""

import copy
import logging
import numpy as np

from jetson_inference.multi_task.config import (
    DEFAULT_EPSILON,
    DEFAULT_J,
    DEFAULT_MU_FALLBACK,
    MIN_LINK_SHARE,
    NUM_NODES,
    NUM_TRANSFER_POINTS,
    _order_mu_values_for_multitask,
)
from jetson_inference.multi_task.state import (
    LatestSnapshot,
    RuntimeTaskDef,
)
from scipy.optimize import minimize
from typing import (
    Dict,
    Sequence,
)


class BaseCommOnlyPolicy:
    def __init__(self, policy_key: str, policy_name: str, tasks: Sequence[RuntimeTaskDef]):
        self.policy_key = str(policy_key)
        self.policy_name = str(policy_name)
        self.tasks = list(tasks)
        self.mu = None

    def update_dual(self, actual_delay_by_task: Dict[int, float], tasks_by_id: Dict[int, RuntimeTaskDef]):
        return None

    def current_allocations(self, snapshots: Dict[int, LatestSnapshot], total_bw_bps: np.ndarray) -> Dict[int, Dict[str, np.ndarray]]:
        raise NotImplementedError

def _policy_uses_fixed_channel(policy=None, algorithm_type=None):
    policy_key = str(algorithm_type or getattr(policy, "policy_key", "")).strip().lower()
    return policy_key in {
        "no_compression_multi_baseline",
        "max_compression_multi_baseline",
    }

def _policy_uses_mu(policy=None, algorithm_type=None):
    policy_key = str(algorithm_type or getattr(policy, "policy_key", "")).strip().lower()
    return policy_key in {
        "no_csi_multi",
        "decoupled_equal_split_multi_baseline",
        "queue_proportional_multi_baseline",
    }

def _policy_uses_estimator(policy=None, algorithm_type=None):
    policy_key = str(algorithm_type or getattr(policy, "policy_key", "")).strip().lower()
    return policy_key in {
        "no_csi_multi",
        "decoupled_equal_split_multi_baseline",
        "queue_proportional_multi_baseline",
        "certainty_equivalence_multi_baseline",
    }

def _normalize_allocations_to_unit_sum(tasks, allocations, link_floor=0.0):
    task_ids = [int(task.logical_task_id) for task in tasks]
    if not task_ids:
        return allocations
    for link_idx in range(NUM_TRANSFER_POINTS):
        raw = np.asarray(
            [max(float(np.asarray(allocations[tid]["s_comm"], dtype=float)[link_idx]), 0.0) for tid in task_ids],
            dtype=float,
        )
        raw_sum = float(np.sum(raw))
        if raw_sum <= 1e-12:
            shares = np.full(len(task_ids), 1.0 / float(len(task_ids)), dtype=float)
        else:
            shares = raw / raw_sum
            if float(link_floor) > 0.0:
                floor_total = float(link_floor) * len(task_ids)
                if floor_total < 1.0:
                    shares = np.maximum(shares, float(link_floor))
                    shares = shares / max(float(np.sum(shares)), 1e-12)
        for pos, tid in enumerate(task_ids):
            allocations[tid]["s_comm"][link_idx] = float(shares[pos])
    for node_idx in range(NUM_NODES):
        raw = np.asarray(
            [max(float(np.asarray(allocations[tid]["s_comp"], dtype=float)[node_idx]), 0.0) for tid in task_ids],
            dtype=float,
        )
        raw_sum = float(np.sum(raw))
        if raw_sum <= 1e-12:
            shares = np.full(len(task_ids), 1.0 / float(len(task_ids)), dtype=float)
        else:
            shares = raw / raw_sum
        for pos, tid in enumerate(task_ids):
            allocations[tid]["s_comp"][node_idx] = float(shares[pos])
    return allocations

def _solve_single_task_eta_with_fixed_resources_online(task, snapshot, c_eff, max_comp_delay, mu, lambda_t):
    c_eff = np.asarray(c_eff, dtype=float)
    bounds = [(float(task.eta_min[idx]), 1.0) for idx in range(task.num_links)] + [(float(max_comp_delay), None)]
    x0 = np.asarray([bound[0] for bound in bounds], dtype=float)

    def objective(x):
        eta_flat = np.asarray(x[:-1], dtype=float)
        z_val = float(x[-1])
        acc = float(task.predict_accuracy(eta_flat))
        grad_acc = np.asarray(task.gradient_accuracy(eta_flat), dtype=float)
        grad = np.zeros_like(x)
        grad[:-1] = -float(task.weight) * grad_acc
        grad[-1] = float(mu) * float(lambda_t)
        return -float(task.weight) * acc + float(mu) * float(lambda_t) * z_val, grad

    constraints = []
    for link_idx in range(task.num_links):
        scale = max(float(c_eff[link_idx]), 1e-12)
        overhead_beta = float(snapshot.fixed_overhead_bytes[link_idx]) / scale
        activation_beta = float(snapshot.a_ref[link_idx]) / scale
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x, link_idx=link_idx, overhead_beta=overhead_beta, activation_beta=activation_beta: (
                    float(x[-1]) - overhead_beta - activation_beta * float(x[link_idx])
                ),
                "jac": lambda x, link_idx=link_idx, activation_beta=activation_beta: np.asarray(
                    [-activation_beta if j == link_idx else (1.0 if j == len(x) - 1 else 0.0) for j in range(len(x))],
                    dtype=float,
                ),
            }
        )
    result = minimize(fun=objective, x0=x0, method="SLSQP", jac=True, bounds=bounds, constraints=constraints)
    eta = np.asarray(result.x[:-1] if result.success else x0[:-1], dtype=float)
    return np.clip(eta, np.asarray(task.eta_min, dtype=float), 1.0)

class NoCompressionMultiTaskBaseline(BaseCommOnlyPolicy):
    def __init__(self, tasks):
        super().__init__("no_compression_multi_baseline", "Baseline: no compression", tasks)

    def current_allocations(self, snapshots, total_bw_bps):
        num_tasks = max(len(self.tasks), 1)
        allocations = {}
        for task in self.tasks:
            allocations[task.logical_task_id] = {
                "eta": np.ones(task.num_links, dtype=float),
                "s_comm": np.full(task.num_links, 1.0 / float(num_tasks), dtype=float),
                "s_comp": np.full(NUM_NODES, 1.0 / float(num_tasks), dtype=float),
            }
        return allocations

class MaxCompressionMultiTaskBaseline(BaseCommOnlyPolicy):
    def __init__(self, tasks):
        super().__init__("max_compression_multi_baseline", "Baseline: max compression", tasks)

    def current_allocations(self, snapshots, total_bw_bps):
        num_tasks = max(len(self.tasks), 1)
        allocations = {}
        for task in self.tasks:
            allocations[task.logical_task_id] = {
                "eta": np.asarray(task.eta_min, dtype=float).copy(),
                "s_comm": np.full(task.num_links, 1.0 / float(num_tasks), dtype=float),
                "s_comp": np.full(NUM_NODES, 1.0 / float(num_tasks), dtype=float),
            }
        return allocations

class EqualShareMultiTaskBaseline(BaseCommOnlyPolicy):
    def __init__(self, tasks):
        super().__init__("equal_share_multi_baseline", "Baseline: equal share", tasks)

    def current_allocations(self, snapshots, total_bw_bps):
        num_tasks = max(len(self.tasks), 1)
        share_vec = np.full(NUM_TRANSFER_POINTS, 1.0 / float(num_tasks), dtype=float)
        allocations = {}
        for task in self.tasks:
            snap = snapshots.get(task.logical_task_id)
            if snap is None:
                eta = np.asarray(task.eta_min, dtype=float).copy()
            else:
                eta = []
                for link_idx in range(task.num_links):
                    raw_budget = (float(share_vec[link_idx]) * float(total_bw_bps[link_idx])) / max(
                        float(task.target_rate_hz),
                        1e-12,
                    )
                    raw = (raw_budget - float(snap.fixed_overhead_bytes[link_idx])) / max(float(snap.a_ref[link_idx]), 1e-12)
                    eta.append(max(float(task.eta_min[link_idx]), min(1.0, raw)))
                eta = np.asarray(eta, dtype=float)
            allocations[task.logical_task_id] = {
                "eta": eta,
                "s_comm": share_vec.copy(),
                "s_comp": np.full(NUM_NODES, 1.0 / float(num_tasks), dtype=float),
            }
        return allocations

class HistoricalAverageCertaintyEquivalenceMultiTaskBaseline(BaseCommOnlyPolicy):
    def __init__(self, tasks):
        super().__init__("certainty_equivalence_multi_baseline", "No-CSI Baseline: moving average", tasks)

    def current_allocations(self, snapshots, total_bw_bps):
        if not snapshots:
            return EqualShareMultiTaskBaseline(self.tasks).current_allocations(snapshots, total_bw_bps)
        for node_idx in range(NUM_NODES):
            comp_sum = sum(
                float(snapshots[int(task.logical_task_id)].tau_list[node_idx]) * float(task.target_rate_hz)
                for task in self.tasks
            )
            if comp_sum > 1.0:
                return MaxCompressionMultiTaskBaseline(self.tasks).current_allocations(snapshots, total_bw_bps)
        for link_idx in range(NUM_TRANSFER_POINTS):
            comm_sum = sum(
                float(snapshots[int(task.logical_task_id)].a_ref[link_idx]) * float(task.eta_min[link_idx]) * float(task.target_rate_hz)
                / max(float(total_bw_bps[link_idx]), 1e-12)
                for task in self.tasks
            )
            if comm_sum > 1.0:
                return MaxCompressionMultiTaskBaseline(self.tasks).current_allocations(snapshots, total_bw_bps)
        context = []
        for task in self.tasks:
            tid = int(task.logical_task_id)
            for link_idx in range(task.num_links):
                beta = float(total_bw_bps[link_idx]) / max(
                    float(task.target_rate_hz) * float(snapshots[tid].a_ref[link_idx]),
                    1e-12,
                )
                s_min = (
                    float(snapshots[tid].a_ref[link_idx]) * float(task.eta_min[link_idx]) * float(task.target_rate_hz)
                ) / max(float(total_bw_bps[link_idx]), 1e-12)
                s_max = min(1.0, 1.0 / max(beta, 1e-12))
                context.append(
                    {
                        "task": task,
                        "task_id": tid,
                        "link_idx": link_idx,
                        "beta": beta,
                        "s_min": s_min,
                        "s_max": s_max,
                    }
                )
        bounds = [(float(ctx["s_min"]), float(ctx["s_max"])) for ctx in context]
        x0 = np.asarray([float(ctx["s_min"]) for ctx in context], dtype=float)

        def objective(s_flat):
            total_obj = 0.0
            grad_flat = np.zeros_like(s_flat)
            for task in self.tasks:
                task_indices = [idx for idx, ctx in enumerate(context) if int(ctx["task_id"]) == int(task.logical_task_id)]
                eta_vec = []
                d_eta_ds = []
                for idx in task_indices:
                    beta = float(context[idx]["beta"])
                    s_val = float(s_flat[idx])
                    eta_vec.append(min(1.0, beta * s_val))
                    d_eta_ds.append(beta)
                eta_arr = np.asarray(eta_vec, dtype=float)
                acc = float(task.predict_accuracy(eta_arr))
                grad_acc = np.asarray(task.gradient_accuracy(eta_arr), dtype=float)
                total_obj += float(task.weight) * acc
                for local_i, flat_i in enumerate(task_indices):
                    grad_flat[flat_i] = float(task.weight) * float(grad_acc[local_i]) * float(d_eta_ds[local_i])
            return -float(total_obj), -grad_flat

        constraints = []
        for link_idx in range(NUM_TRANSFER_POINTS):
            indices = [idx for idx, ctx in enumerate(context) if int(ctx["link_idx"]) == int(link_idx)]
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda s_flat, idxs=indices: 1.0 - sum(float(s_flat[j]) for j in idxs),
                }
            )
        result = minimize(fun=objective, x0=x0, method="SLSQP", jac=True, bounds=bounds, constraints=constraints)
        s_flat = np.asarray(result.x if result.success else x0, dtype=float)
        allocations = {}
        for task in self.tasks:
            tid = int(task.logical_task_id)
            s_comp = np.asarray(
                [float(snapshots[tid].tau_list[node_idx]) * float(task.target_rate_hz) for node_idx in range(NUM_NODES)],
                dtype=float,
            )
            task_indices = [idx for idx, ctx in enumerate(context) if int(ctx["task_id"]) == tid]
            s_comm = []
            eta = []
            for idx in task_indices:
                s_val = float(s_flat[idx])
                beta = float(context[idx]["beta"])
                s_comm.append(s_val)
                eta.append(min(1.0, beta * s_val))
            allocations[tid] = {
                "eta": np.asarray(eta, dtype=float),
                "s_comm": np.asarray(s_comm, dtype=float),
                "s_comp": s_comp,
            }
        return _normalize_allocations_to_unit_sum(self.tasks, allocations)

class NoCSICommOnlyMultiTaskPolicy(BaseCommOnlyPolicy):
    def __init__(self, tasks: Sequence[RuntimeTaskDef], mu: float, epsilon: float, J: int):
        super().__init__("no_csi_multi", "No-CSI (Alg2) (mu={})".format(mu), tasks)
        self.mu = float(mu)
        self.epsilon = float(epsilon)
        self.J = int(J)
        self.lambda_k = {task.logical_task_id: float(self.epsilon) for task in self.tasks}

    def _phase_a_objective(self, z_flat, task_ids):
        grad = np.asarray([self.mu * self.lambda_k[int(tid)] for tid in task_ids], dtype=float)
        return float(np.dot(grad, z_flat)), grad

    def _phase_a_constraints(self, snapshots, current_eta, total_bw_bps, task_ids):
        constraints = []
        for node_idx in range(NUM_NODES):
            active_task_ids = [int(tid) for tid in task_ids]
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda z, node_idx=node_idx, active_task_ids=active_task_ids: 1.0
                    - sum(
                        float(snapshots[int(tid)].tau_list[node_idx]) / max(float(z[pos]), 1e-12)
                        for pos, tid in enumerate(active_task_ids)
                    ),
                }
            )
        for link_idx in range(NUM_TRANSFER_POINTS):
            active_task_ids = [int(tid) for tid in task_ids]
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda z, link_idx=link_idx, active_task_ids=active_task_ids: 1.0
                    - sum(
                        float(
                            snapshots[int(tid)].estimated_payload_bytes(current_eta[int(tid)])[link_idx]
                        )
                        / (max(float(total_bw_bps[link_idx]), 1e-12) * max(float(z[pos]), 1e-12))
                        for pos, tid in enumerate(active_task_ids)
                    ),
                }
            )
        return constraints

    def _phase_a_optimize_resources(self, snapshots, total_bw_bps, current_eta):
        task_ids = [int(task.logical_task_id) for task in self.tasks]
        bounds = [(max(float(snapshots[int(tid)].compute_floor), 1e-9), None) for tid in task_ids]
        x0 = np.asarray([float(bound[0]) * 1.1 for bound in bounds], dtype=float)
        result = minimize(
            fun=self._phase_a_objective,
            x0=x0,
            args=(task_ids,),
            method="SLSQP",
            jac=True,
            bounds=bounds,
            constraints=self._phase_a_constraints(snapshots, current_eta, total_bw_bps, task_ids),
        )
        z_flat = np.asarray(result.x if result.success else x0, dtype=float)
        updated = {}
        num_tasks = max(len(self.tasks), 1)
        for pos, task in enumerate(self.tasks):
            tid = int(task.logical_task_id)
            z_val = max(float(z_flat[pos]), 1e-9)
            snap = snapshots[tid]
            s_comp = np.asarray(
                [
                    float(snap.tau_list[node_idx]) / z_val
                    for node_idx in range(NUM_NODES)
                ],
                dtype=float,
            )
            s_comm = np.asarray(
                [
                    float(snap.estimated_payload_bytes(current_eta[tid])[link_idx])
                    / (max(float(total_bw_bps[link_idx]), 1e-12) * z_val)
                    for link_idx in range(task.num_links)
                ],
                dtype=float,
            )
            updated[tid] = {
                "s_comm": s_comm,
                "s_comp": s_comp,
            }
        return self._stabilize_allocations(updated)

    def _stabilize_allocations(self, allocations):
        task_ids = [int(task.logical_task_id) for task in self.tasks]
        if not task_ids:
            return allocations
        num_tasks = float(len(task_ids))

        for link_idx in range(NUM_TRANSFER_POINTS):
            raw = np.asarray(
                [max(float(allocations[tid]["s_comm"][link_idx]), 0.0) for tid in task_ids],
                dtype=float,
            )
            raw_sum = float(np.sum(raw))
            if np.any(raw <= 1e-12):
                logging.warning(
                    "[No-CSI] raw s_comm on link %s is degenerate before stabilization: %s",
                    int(link_idx),
                    {int(tid): float(raw[pos]) for pos, tid in enumerate(task_ids)},
                )
            if raw_sum <= 1e-12:
                shares = np.full(len(task_ids), 1.0 / num_tasks, dtype=float)
            else:
                shares = raw / raw_sum
                floor_total = MIN_LINK_SHARE * len(task_ids)
                if floor_total < 1.0:
                    shares = np.maximum(shares, MIN_LINK_SHARE)
                    shares = shares / max(float(np.sum(shares)), 1e-12)
            for pos, tid in enumerate(task_ids):
                allocations[tid]["s_comm"][link_idx] = float(shares[pos])

        for node_idx in range(NUM_NODES):
            raw = np.asarray(
                [max(float(allocations[tid]["s_comp"][node_idx]), 0.0) for tid in task_ids],
                dtype=float,
            )
            raw_sum = float(np.sum(raw))
            if raw_sum <= 1e-12:
                shares = np.full(len(task_ids), 1.0 / num_tasks, dtype=float)
            else:
                shares = raw / raw_sum
            for pos, tid in enumerate(task_ids):
                allocations[tid]["s_comp"][node_idx] = float(shares[pos])

        return allocations

    def _phase_b_optimize_configuration(self, snapshots, total_bw_bps, current_s):
        updated_eta = {}
        for task in self.tasks:
            tid = int(task.logical_task_id)
            snap = snapshots[tid]
            s_comm = np.asarray(current_s[tid]["s_comm"], dtype=float)
            s_comp = np.asarray(current_s[tid]["s_comp"], dtype=float)
            max_comp_delay = max(
                float(snap.tau_list[node_idx]) / max(float(s_comp[node_idx]), 1e-12)
                for node_idx in range(NUM_NODES)
            )
            bounds = [(float(task.eta_min[idx]), 1.0) for idx in range(task.num_links)] + [(max_comp_delay, None)]
            x0 = np.asarray([bound[0] for bound in bounds], dtype=float)

            def objective(x):
                eta_flat = np.asarray(x[:-1], dtype=float)
                z_val = float(x[-1])
                acc = float(task.predict_accuracy(eta_flat))
                grad_acc = np.asarray(task.gradient_accuracy(eta_flat), dtype=float)
                grad = np.zeros_like(x)
                grad[:-1] = -float(task.weight) * grad_acc
                grad[-1] = self.mu * self.lambda_k[tid]
                return -float(task.weight) * acc + self.mu * self.lambda_k[tid] * z_val, grad

            constraints = []
            for node_idx in range(NUM_NODES):
                comp_beta = float(snap.tau_list[node_idx]) / max(float(s_comp[node_idx]), 1e-12)
                constraints.append(
                    {
                        "type": "ineq",
                        "fun": lambda x, comp_beta=comp_beta: float(x[-1]) - comp_beta,
                        "jac": lambda x, comp_beta=comp_beta: np.asarray(
                            [1.0 if j == len(x) - 1 else 0.0 for j in range(len(x))],
                            dtype=float,
                        ),
                    }
                )
            for link_idx in range(task.num_links):
                scale = max(float(total_bw_bps[link_idx]) * max(float(s_comm[link_idx]), 1e-12), 1e-12)
                overhead_beta = float(snap.fixed_overhead_bytes[link_idx]) / scale
                activation_beta = float(snap.a_ref[link_idx]) / scale
                constraints.append(
                    {
                        "type": "ineq",
                        "fun": lambda x, link_idx=link_idx, overhead_beta=overhead_beta, activation_beta=activation_beta: (
                            float(x[-1]) - overhead_beta - activation_beta * float(x[link_idx])
                        ),
                        "jac": lambda x, link_idx=link_idx, activation_beta=activation_beta: np.asarray(
                            [-activation_beta if j == link_idx else (1.0 if j == len(x) - 1 else 0.0) for j in range(len(x))],
                            dtype=float,
                        ),
                    }
                )
            result = minimize(fun=objective, x0=x0, method="SLSQP", jac=True, bounds=bounds, constraints=constraints)
            updated_eta[tid] = np.asarray(result.x[:-1] if result.success else x0[:-1], dtype=float)
        return updated_eta

    def current_allocations(self, snapshots, total_bw_bps):
        if not snapshots:
            return EqualShareMultiTaskBaseline(self.tasks).current_allocations(snapshots, total_bw_bps)
        current_eta = {int(task.logical_task_id): np.asarray(task.eta_min, dtype=float).copy() for task in self.tasks}
        current_s = None
        for _ in range(self.J):
            current_s = self._phase_a_optimize_resources(snapshots, total_bw_bps, current_eta)
            current_eta = self._phase_b_optimize_configuration(snapshots, total_bw_bps, current_s)
        assert current_s is not None
        allocations = {}
        for task in self.tasks:
            tid = int(task.logical_task_id)
            allocations[tid] = {
                "eta": np.asarray(current_eta[tid], dtype=float),
                "s_comm": np.asarray(current_s[tid]["s_comm"], dtype=float),
                "s_comp": np.asarray(current_s[tid]["s_comp"], dtype=float),
            }
        return allocations

    def update_dual(self, actual_delay_by_task, tasks_by_id):
        for tid, actual_delay in actual_delay_by_task.items():
            task = tasks_by_id[int(tid)]
            self.lambda_k[int(tid)] = max(
                self.epsilon,
                float(self.lambda_k[int(tid)]) + float(actual_delay) - (1.0 / float(task.target_rate_hz)),
            )

class DecoupledEqualSplitStochasticDescentMultiTaskBaseline(BaseCommOnlyPolicy):
    def __init__(self, tasks, mu, epsilon):
        super().__init__("decoupled_equal_split_multi_baseline", "Baseline: decoupled equal split (mu={})".format(mu), tasks)
        self.mu = float(mu)
        self.epsilon = float(epsilon)
        self.lambda_k = {int(task.logical_task_id): float(self.epsilon) for task in self.tasks}

    def current_allocations(self, snapshots, total_bw_bps):
        if not snapshots:
            return EqualShareMultiTaskBaseline(self.tasks).current_allocations(snapshots, total_bw_bps)
        num_tasks = max(len(self.tasks), 1)
        allocations = {}
        for task in self.tasks:
            tid = int(task.logical_task_id)
            snapshot = snapshots[tid]
            s_comp = np.full(NUM_NODES, 1.0 / float(num_tasks), dtype=float)
            s_comm = np.full(task.num_links, 1.0 / float(num_tasks), dtype=float)
            c_eff = np.asarray(total_bw_bps, dtype=float) * np.asarray(s_comm, dtype=float)
            max_comp_delay = max(float(snapshot.tau_list[node_idx]) / max(float(s_comp[node_idx]), 1e-12) for node_idx in range(NUM_NODES))
            eta = _solve_single_task_eta_with_fixed_resources_online(
                task=task,
                snapshot=snapshot,
                c_eff=c_eff,
                max_comp_delay=max_comp_delay,
                mu=self.mu,
                lambda_t=float(self.lambda_k[tid]),
            )
            allocations[tid] = {"eta": eta, "s_comm": s_comm, "s_comp": s_comp}
        return _normalize_allocations_to_unit_sum(self.tasks, allocations)

    def update_dual(self, actual_delay_by_task, tasks_by_id):
        for tid, actual_delay in actual_delay_by_task.items():
            task = tasks_by_id[int(tid)]
            self.lambda_k[int(tid)] = max(
                self.epsilon,
                float(self.lambda_k[int(tid)]) + float(actual_delay) - (1.0 / float(task.target_rate_hz)),
            )

class QueueProportionalHeuristicMultiTaskBaseline(BaseCommOnlyPolicy):
    def __init__(self, tasks, mu, epsilon):
        super().__init__("queue_proportional_multi_baseline", "Baseline: queue proportional (mu={})".format(mu), tasks)
        self.mu = float(mu)
        self.epsilon = float(epsilon)
        self.lambda_k = {int(task.logical_task_id): float(self.epsilon) for task in self.tasks}

    def current_allocations(self, snapshots, total_bw_bps):
        if not snapshots:
            return EqualShareMultiTaskBaseline(self.tasks).current_allocations(snapshots, total_bw_bps)
        allocations = {}
        s_comp_map = {int(task.logical_task_id): np.zeros(NUM_NODES, dtype=float) for task in self.tasks}
        s_comm_map = {int(task.logical_task_id): np.zeros(task.num_links, dtype=float) for task in self.tasks}
        for node_idx in range(NUM_NODES):
            denom = sum(float(self.lambda_k[int(task.logical_task_id)]) for task in self.tasks)
            for task in self.tasks:
                tid = int(task.logical_task_id)
                s_comp_map[tid][node_idx] = (float(self.lambda_k[tid]) / denom) if denom > 0.0 else (1.0 / max(len(self.tasks), 1))
        for link_idx in range(NUM_TRANSFER_POINTS):
            denom = sum(float(self.lambda_k[int(task.logical_task_id)]) for task in self.tasks)
            for task in self.tasks:
                tid = int(task.logical_task_id)
                s_comm_map[tid][link_idx] = (float(self.lambda_k[tid]) / denom) if denom > 0.0 else (1.0 / max(len(self.tasks), 1))
        for task in self.tasks:
            tid = int(task.logical_task_id)
            snapshot = snapshots[tid]
            s_comp = s_comp_map[tid]
            s_comm = s_comm_map[tid]
            c_eff = np.asarray(total_bw_bps, dtype=float) * np.asarray(s_comm, dtype=float)
            max_comp_delay = max(float(snapshot.tau_list[node_idx]) / max(float(s_comp[node_idx]), 1e-12) for node_idx in range(NUM_NODES))
            eta = _solve_single_task_eta_with_fixed_resources_online(
                task=task,
                snapshot=snapshot,
                c_eff=c_eff,
                max_comp_delay=max_comp_delay,
                mu=self.mu,
                lambda_t=float(self.lambda_k[tid]),
            )
            allocations[tid] = {"eta": eta, "s_comm": s_comm, "s_comp": s_comp}
        return _normalize_allocations_to_unit_sum(self.tasks, allocations)

    def update_dual(self, actual_delay_by_task, tasks_by_id):
        for tid, actual_delay in actual_delay_by_task.items():
            task = tasks_by_id[int(tid)]
            self.lambda_k[int(tid)] = max(
                self.epsilon,
                float(self.lambda_k[int(tid)]) + float(actual_delay) - (1.0 / float(task.target_rate_hz)),
            )

def create_policy(policy_cfg, tasks):
    algo_type = str(policy_cfg["type"]).lower()
    if algo_type == "no_csi_multi":
        return NoCSICommOnlyMultiTaskPolicy(
            tasks,
            mu=float(policy_cfg.get("mu", DEFAULT_MU_FALLBACK)),
            epsilon=float(policy_cfg.get("epsilon", DEFAULT_EPSILON)),
            J=int(policy_cfg.get("J", DEFAULT_J)),
        )
    if algo_type == "equal_share_multi_baseline":
        return EqualShareMultiTaskBaseline(tasks)
    if algo_type == "no_compression_multi_baseline":
        return NoCompressionMultiTaskBaseline(tasks)
    if algo_type == "max_compression_multi_baseline":
        return MaxCompressionMultiTaskBaseline(tasks)
    if algo_type == "certainty_equivalence_multi_baseline":
        return HistoricalAverageCertaintyEquivalenceMultiTaskBaseline(tasks)
    if algo_type == "decoupled_equal_split_multi_baseline":
        return DecoupledEqualSplitStochasticDescentMultiTaskBaseline(
            tasks,
            mu=float(policy_cfg.get("mu", DEFAULT_MU_FALLBACK)),
            epsilon=float(policy_cfg.get("epsilon", DEFAULT_EPSILON)),
        )
    if algo_type == "queue_proportional_multi_baseline":
        return QueueProportionalHeuristicMultiTaskBaseline(
            tasks,
            mu=float(policy_cfg.get("mu", DEFAULT_MU_FALLBACK)),
            epsilon=float(policy_cfg.get("epsilon", DEFAULT_EPSILON)),
        )
    raise ValueError("Unsupported algorithm type '{}'".format(algo_type))

def _default_algorithms(mu_values):
    algorithms = []
    for mu in _order_mu_values_for_multitask(mu_values):
        algorithms.append(
            {
                "name": "No-CSI (Alg2) (mu={})".format(mu),
                "type": "no_csi_multi",
                "mu": float(mu),
                "epsilon": DEFAULT_EPSILON,
                "J": DEFAULT_J,
            }
        )
    algorithms.extend(
        [
            {"name": "No-CSI Baseline: moving average", "type": "certainty_equivalence_multi_baseline"},
            {"name": "Baseline: no compression", "type": "no_compression_multi_baseline"},
            {"name": "Baseline: max compression", "type": "max_compression_multi_baseline"},
            {"name": "Baseline: equal share", "type": "equal_share_multi_baseline"},
        ]
    )
    for mu in _order_mu_values_for_multitask(mu_values):
        algorithms.append(
            {
                "name": "Baseline: decoupled equal split (mu={})".format(mu),
                "type": "decoupled_equal_split_multi_baseline",
                "mu": float(mu),
                "epsilon": DEFAULT_EPSILON,
            }
        )
        algorithms.append(
            {
                "name": "Baseline: queue proportional (mu={})".format(mu),
                "type": "queue_proportional_multi_baseline",
                "mu": float(mu),
                "epsilon": DEFAULT_EPSILON,
            }
        )
    return algorithms

def _expand_algorithms(manifest, mu_values):
    if "algorithms" not in manifest or not manifest["algorithms"]:
        return _default_algorithms(mu_values)
    ordered_mu_values = _order_mu_values_for_multitask(mu_values)
    expanded = []
    for algo_cfg in manifest["algorithms"]:
        if str(algo_cfg.get("type", "")).lower() != "no_csi_multi":
            expanded.append(copy.deepcopy(algo_cfg))
            continue
        if "mu" in algo_cfg:
            expanded.append(copy.deepcopy(algo_cfg))
            continue
        for mu in ordered_mu_values:
            item = copy.deepcopy(algo_cfg)
            item["mu"] = float(mu)
            name = str(item.get("name", "No-CSI (Alg2)"))
            item["name"] = "{} (mu={})".format(name, mu) if "mu=" not in name else name
            expanded.append(item)
    return expanded
