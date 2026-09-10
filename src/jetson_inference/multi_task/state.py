"""task, snapshot and shared-state structures."""

import numpy as np

from dataclasses import (
    dataclass,
    field,
)
from jetson_inference.multi_task.config import NUM_NODES
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)


@dataclass
class RuntimeTaskDef:
    logical_task_id: int
    family: str
    name: str
    model: str
    dataset: str
    weight: float
    eta_min: np.ndarray
    target_rate_hz: float
    batch_size: int
    estimator: Any
    codec_name: str
    outlier_precision: Optional[str]
    regular_precision: Optional[str]
    llmint8_mapping_entries: List[Dict[str, Any]]
    warmup_items: List[Dict[str, Any]]
    experiment_items: List[Dict[str, Any]]
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_links(self) -> int:
        return int(len(self.eta_min))

    def predict_accuracy(self, eta: Sequence[float]) -> float:
        return float(self.estimator.predict(eta))

    def gradient_accuracy(self, eta: Sequence[float]) -> np.ndarray:
        return np.asarray(self.estimator.gradient(eta), dtype=float)

@dataclass
class LatestSnapshot:
    a_ref: np.ndarray
    fixed_overhead_bytes: np.ndarray
    tau_list: np.ndarray
    total_delay_sec: float
    end_to_end_delay_sec: float
    tp_bandwidth_bps: np.ndarray
    sample_correct: float
    running_accuracy: float
    service_delay_sec: float
    communication_delay_sec: float
    miscellaneous_delay_sec: float

    @property
    def compute_floor(self) -> float:
        return float(np.max(self.tau_list))

    def estimated_payload_bytes(self, eta: Sequence[float]) -> np.ndarray:
        eta_arr = np.asarray(eta, dtype=float)
        return np.asarray(self.fixed_overhead_bytes, dtype=float) + np.asarray(self.a_ref, dtype=float) * eta_arr

def _compute_task_predictions(tasks_by_id, allocations, snapshots, total_bw_bps, enable_accuracy_prediction=True):
    predicted_accuracy_by_task = {}
    predicted_delay_by_task = {}
    if not enable_accuracy_prediction:
        for tid in tasks_by_id.keys():
            predicted_accuracy_by_task[int(tid)] = np.nan
            predicted_delay_by_task[int(tid)] = np.nan
        return predicted_accuracy_by_task, predicted_delay_by_task
    total_bw_bps = np.asarray(total_bw_bps, dtype=float)
    for tid, task in tasks_by_id.items():
        snap = snapshots.get(int(tid))
        alloc = allocations.get(int(tid))
        if snap is None or alloc is None:
            predicted_accuracy_by_task[int(tid)] = np.nan
            predicted_delay_by_task[int(tid)] = np.nan
            continue
        eta = np.asarray(alloc["eta"], dtype=float)
        s_comm = np.asarray(alloc["s_comm"], dtype=float)
        s_comp = np.asarray(alloc["s_comp"], dtype=float)
        predicted_accuracy_by_task[int(tid)] = float(task.predict_accuracy(eta))
        comp_delays = [
            float(snap.tau_list[node_idx]) / max(float(s_comp[node_idx]), 1e-12)
            for node_idx in range(NUM_NODES)
        ]
        comm_delays = [
            float(snap.estimated_payload_bytes(eta)[link_idx])
            / max(float(total_bw_bps[link_idx]) * max(float(s_comm[link_idx]), 1e-12), 1e-12)
            for link_idx in range(task.num_links)
        ]
        predicted_delay_by_task[int(tid)] = float(max(comp_delays + comm_delays))
    return predicted_accuracy_by_task, predicted_delay_by_task

class SharedBandwidthState:
    def __init__(self, task_ids: Sequence[int], num_links: int, initial_total_bps: float):
        self.task_ids = [int(item) for item in task_ids]
        self.num_links = int(num_links)
        self.initial_total_bps = float(initial_total_bps)
        per_task_seed = self.initial_total_bps / max(len(self.task_ids), 1)
        self.latest_by_task = {
            int(task_id): np.full(self.num_links, per_task_seed, dtype=float)
            for task_id in self.task_ids
        }

    def observe(self, logical_task_id: int, tp_bandwidth_bps: Sequence[float]):
        self.latest_by_task[int(logical_task_id)] = np.asarray(tp_bandwidth_bps, dtype=float)

    def total_bandwidth_bps(self) -> np.ndarray:
        total = np.zeros(self.num_links, dtype=float)
        for values in self.latest_by_task.values():
            total += np.asarray(values, dtype=float)
        return np.where(total > 1e-6, total, self.initial_total_bps)

    def total_tp_stats(self) -> List[Tuple[float, float]]:
        total_bps = self.total_bandwidth_bps()
        return [(float(bps) / 8.0, 1.0) for bps in total_bps]

def _clone_allocations(allocations):
    return {
        int(tid): {
            "eta": np.asarray(alloc["eta"], dtype=float).copy(),
            "s_comm": np.asarray(alloc["s_comm"], dtype=float).copy(),
            "s_comp": np.asarray(alloc["s_comp"], dtype=float).copy(),
        }
        for tid, alloc in allocations.items()
    }
