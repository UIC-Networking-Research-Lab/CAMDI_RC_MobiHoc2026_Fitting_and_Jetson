"""model partitions, partition factory and partition planning."""

import jetson_inference.common.resnet_arch as resnet20
import math
import torch
import torch.nn as nn

from collections import OrderedDict
from jetson_inference.resnet.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATA_ROOT,
    DEFAULT_DOWNLOAD_DATA,
    DEFAULT_MAX_BATCHES,
    MODEL_CHECKPOINT,
    NODE_ORDER,
    NUM_PARTITIONS,
)


class ResNetHead(nn.Module):
    def __init__(self, avgpool, flatten, linear):
        super().__init__()
        self.avgpool = avgpool
        self.flatten = flatten
        self.linear = linear

    def forward(self, x):
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

class ResNetUnitPartition(nn.Module):
    def __init__(self, units, unit_names, partition_idx, is_last_partition):
        super().__init__()
        self.units = nn.ModuleList(list(units))
        self.unit_names = list(unit_names)
        self.partition_idx = int(partition_idx)
        self.is_last_partition = bool(is_last_partition)
        self.partition_info = {
            "partition_idx": int(partition_idx),
            "unit_names": list(unit_names),
            "is_last_partition": bool(is_last_partition),
        }

    def forward(self, x):
        for module in self.units:
            x = module(x)
        return x

def conv2d_macs(module, input_shape):
    batch, in_channels, in_h, in_w = input_shape
    out_channels = module.out_channels
    kernel_h, kernel_w = module.kernel_size
    stride_h, stride_w = module.stride
    pad_h, pad_w = module.padding
    dil_h, dil_w = module.dilation
    groups = module.groups
    out_h = math.floor((in_h + 2 * pad_h - dil_h * (kernel_h - 1) - 1) / stride_h + 1)
    out_w = math.floor((in_w + 2 * pad_w - dil_w * (kernel_w - 1) - 1) / stride_w + 1)
    macs = batch * out_h * out_w * out_channels * (in_channels // groups) * kernel_h * kernel_w
    return float(macs), (batch, out_channels, out_h, out_w)

def linear_macs(module, input_shape):
    batch = input_shape[0]
    macs = batch * module.in_features * module.out_features
    return float(macs), (batch, module.out_features)

def estimate_unit_cost(unit, input_shape):
    if isinstance(unit, nn.Sequential):
        total = 0.0
        shape = input_shape
        for child in unit.children():
            child_cost, shape = estimate_unit_cost(child, shape)
            total += child_cost
        return total, shape

    if isinstance(unit, resnet20.BasicBlock):
        total = 0.0
        shape = input_shape
        cost, shape = conv2d_macs(unit.conv1, shape)
        total += cost
        cost, shape = conv2d_macs(unit.conv2, shape)
        total += cost
        for child in unit.shortcut.children():
            if isinstance(child, nn.Conv2d):
                cost, _ = conv2d_macs(child, input_shape)
                total += cost
        return total, shape

    if isinstance(unit, nn.Conv2d):
        return conv2d_macs(unit, input_shape)

    if isinstance(unit, ResNetHead):
        batch, channels, _, _ = input_shape
        return linear_macs(unit.linear, (batch, channels))

    return 0.0, input_shape

def build_equal_cost_ranges(costs, num_partitions):
    if num_partitions <= 0:
        raise ValueError("num_partitions must be positive")
    if num_partitions > len(costs):
        raise ValueError("num_partitions cannot exceed number of units")
    total_cost = float(sum(costs))
    if total_cost <= 0:
        step = len(costs) // num_partitions
        remainder = len(costs) % num_partitions
        ranges = []
        start = 0
        for partition_idx in range(num_partitions):
            width = step + (1 if partition_idx < remainder else 0)
            end = start + width
            ranges.append((start, end))
            start = end
        return ranges

    prefix = [0.0]
    for cost in costs:
        prefix.append(prefix[-1] + float(cost))

    ranges = []
    start = 0
    for partition_idx in range(num_partitions - 1):
        target_cost = total_cost * float(partition_idx + 1) / float(num_partitions)
        min_end = start + 1
        max_end = len(costs) - (num_partitions - partition_idx - 1)
        best_end = min_end
        best_err = None
        for end in range(min_end, max_end + 1):
            err = abs(prefix[end] - target_cost)
            if best_err is None or err < best_err:
                best_err = err
                best_end = end
        ranges.append((start, best_end))
        start = best_end

    ranges.append((start, len(costs)))
    return ranges

class ResNet56PartitionFactory(object):
    def __init__(self, checkpoint_path=MODEL_CHECKPOINT, device="cuda", num_partitions=NUM_PARTITIONS):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.num_partitions = int(num_partitions)
        self.full_model = self._load_full_model()
        self.units, self.unit_names = self._build_units()
        self.unit_costs = self._estimate_unit_costs()
        self.partition_ranges = build_equal_cost_ranges(self.unit_costs, self.num_partitions)
        self.partition_plan = self._build_partition_plan()

    def _load_full_model(self):
        model = resnet20.resnet56()
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        normalized = OrderedDict()
        for key, value in state_dict.items():
            normalized[key[7:] if key.startswith("module.") else key] = value
        model.load_state_dict(normalized)
        model.to(self.device)
        model.eval()
        return model

    def _build_units(self):
        units = []
        names = []
        units.append(nn.Sequential(self.full_model.conv1, self.full_model.bn1, self.full_model.relu))
        names.append("stem")
        for idx, block in enumerate(self.full_model.layer1):
            units.append(block)
            names.append("layer1.{}".format(idx))
        for idx, block in enumerate(self.full_model.layer2):
            units.append(block)
            names.append("layer2.{}".format(idx))
        for idx, block in enumerate(self.full_model.layer3):
            units.append(block)
            names.append("layer3.{}".format(idx))
        units.append(ResNetHead(self.full_model.avgpool, self.full_model.flatten, self.full_model.linear))
        names.append("head")
        return units, names

    def _estimate_unit_costs(self):
        costs = []
        shape = (1, 3, 32, 32)
        for unit in self.units:
            cost, shape = estimate_unit_cost(unit, shape)
            costs.append(max(cost, 1.0))
        return costs

    def _build_partition_plan(self):
        plan = []
        for partition_idx, (start, end) in enumerate(self.partition_ranges):
            plan.append(
                {
                    "partition_idx": partition_idx,
                    "node_id": NODE_ORDER[partition_idx],
                    "unit_start": start,
                    "unit_end": end,
                    "unit_names": list(self.unit_names[start:end]),
                    "total_cost": float(sum(self.unit_costs[start:end])),
                    "is_last_partition": partition_idx == self.num_partitions - 1,
                }
            )
        return plan

    def build_partition(self, partition_idx):
        start, end = self.partition_ranges[partition_idx]
        node = ResNetUnitPartition(
            units=self.units[start:end],
            unit_names=self.unit_names[start:end],
            partition_idx=partition_idx,
            is_last_partition=(partition_idx == self.num_partitions - 1),
        ).to(self.device)
        node.eval()
        node.partition_info["node_id"] = NODE_ORDER[partition_idx]
        node.partition_info["checkpoint_path"] = self.checkpoint_path
        return node

    def build_partition_for_node(self, node_id):
        return self.build_partition(NODE_ORDER.index(node_id))

def load_cifar10_batches(
    batch_size=DEFAULT_BATCH_SIZE,
    max_batches=DEFAULT_MAX_BATCHES,
    data_root=DEFAULT_DATA_ROOT,
    download=DEFAULT_DOWNLOAD_DATA,
):
    try:
        import torchvision
        import torchvision.transforms as transforms
    except ImportError as exc:
        raise RuntimeError("torchvision is required for CIFAR-10 loading") from exc

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    dataset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=download,
        transform=transform,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    batches = []
    for batch_idx, (images, labels) in enumerate(loader):
        batches.append({"batch_idx": batch_idx, "images": images, "labels": labels})
        if max_batches is not None and len(batches) >= int(max_batches):
            break
    return batches

def print_partition_plan(factory):
    print("=" * 80)
    print("ResNet56 partition plan")
    print("=" * 80)
    for item in factory.partition_plan:
        print("[{node_id}] partition={partition_idx} units={unit_start}:{unit_end} cost={total_cost:.0f} names={unit_names}".format(**item))
