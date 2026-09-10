"""ResNet partitioning, identity payloads and measurements for trace collection."""

import copy
import math
import os
import time
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from jetson_inference.common import resnet_arch as resnet20
from jetson_inference.common.compressors import get_compressor
from jetson_inference.common.http_transport import MessageType, create_transport


NODE_ORDER = ['A', 'B', 'C', 'D']

NODE_IPS = {
    'A': '192.168.1.20',
    'B': '192.168.1.21',
    'C': '192.168.1.22',
    'D': '192.168.1.23',
}

NODE_PORTS = {
    'A': 52000,
    'B': 52001,
    'C': 52002,
    'D': 52003,
}

MODEL_CHECKPOINT = 'resnet56-4bfd9763.th'

MODEL_TAG = 'resnet56_cifar10'

NUM_PARTITIONS = 4

NUM_TRANSFER_POINTS = 3

DEFAULT_BATCH_SIZE = 100

DEFAULT_MAX_BATCHES = None

DEFAULT_DATA_ROOT = 'data'

DEFAULT_DOWNLOAD_DATA = False

ASYNC_MAX_INFLIGHT_TASKS = 4

ASYNC_QUEUE_POLL_SEC = 0.01

TRANSPORT_BACKEND = 'http'

TRANSPORT_READY_TIMEOUT_SEC = 180.0

DEFAULT_TX_LIMIT_MBPS = None

DEFAULT_TX_BUCKET_CAPACITY_BYTES = 64 * 1024

def _identity_activation_payload(tensor_cpu):
    return {
        'mode': 'identity',
        'tensor': tensor_cpu,
    }

def _flatten_dict(data, prefix=''):
    flat = {}
    for key, value in data.items():
        new_key = '{}_{}'.format(prefix, key) if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, new_key))
        elif isinstance(value, (list, tuple)):
            flat[new_key] = ','.join(str(item) for item in value)
        else:
            flat[new_key] = value
    return flat

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
        self.units = nn.ModuleList(units)
        self.unit_names = list(unit_names)
        self.partition_idx = partition_idx
        self.is_last_partition = bool(is_last_partition)
        self.partition_info = {
            'partition_idx': partition_idx,
            'unit_names': list(unit_names),
            'is_last_partition': bool(is_last_partition),
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
        raise ValueError('num_partitions must be positive')
    if num_partitions > len(costs):
        raise ValueError('num_partitions cannot exceed number of units')
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

    if len(ranges) != num_partitions:
        raise RuntimeError('Failed to build {} partition ranges'.format(num_partitions))
    if any(end <= begin for begin, end in ranges):
        raise RuntimeError('Encountered empty partition while building ranges: {}'.format(ranges))
    return ranges

class ResNet56PartitionFactory(object):
    def __init__(self, checkpoint_path=MODEL_CHECKPOINT, device='cpu', num_partitions=NUM_PARTITIONS):
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
        state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
        normalized = OrderedDict()
        for key, value in state_dict.items():
            normalized[key[7:] if key.startswith('module.') else key] = value
        model.load_state_dict(normalized)
        model.to(self.device)
        model.eval()
        return model

    def _build_units(self):
        units = []
        names = []
        units.append(nn.Sequential(self.full_model.conv1, self.full_model.bn1, self.full_model.relu))
        names.append('stem')
        for idx, block in enumerate(self.full_model.layer1):
            units.append(block)
            names.append('layer1.{}'.format(idx))
        for idx, block in enumerate(self.full_model.layer2):
            units.append(block)
            names.append('layer2.{}'.format(idx))
        for idx, block in enumerate(self.full_model.layer3):
            units.append(block)
            names.append('layer3.{}'.format(idx))
        units.append(ResNetHead(self.full_model.avgpool, self.full_model.flatten, self.full_model.linear))
        names.append('head')
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
            plan.append({
                'partition_idx': partition_idx,
                'node_id': NODE_ORDER[partition_idx],
                'unit_start': start,
                'unit_end': end,
                'unit_names': list(self.unit_names[start:end]),
                'total_cost': float(sum(self.unit_costs[start:end])),
                'is_last_partition': partition_idx == self.num_partitions - 1,
            })
        return plan

    @property
    def info(self):
        return {
            'model_tag': MODEL_TAG,
            'checkpoint_path': self.checkpoint_path,
            'device': self.device,
            'unit_names': list(self.unit_names),
            'unit_costs': list(self.unit_costs),
            'partition_ranges': list(self.partition_ranges),
            'partition_plan': list(self.partition_plan),
        }

    def build_partition(self, partition_idx):
        start, end = self.partition_ranges[partition_idx]
        node = ResNetUnitPartition(
            units=self.units[start:end],
            unit_names=self.unit_names[start:end],
            partition_idx=partition_idx,
            is_last_partition=(partition_idx == self.num_partitions - 1),
        ).to(self.device)
        node.eval()
        node.partition_info['node_id'] = NODE_ORDER[partition_idx]
        node.partition_info['checkpoint_path'] = self.checkpoint_path
        return node

    def build_partition_for_node(self, node_id):
        return self.build_partition(NODE_ORDER.index(node_id))

    def build_all_partitions(self):
        return [self.build_partition(idx) for idx in range(self.num_partitions)]

def build_activation_payload(tensor, compression_param, compressor_name, feature_k_value):
    tensor_cpu = tensor.detach().cpu()
    original_bytes = int(tensor_cpu.numel() * tensor_cpu.element_size())

    if compressor_name in ('identity', None):
        payload = _identity_activation_payload(tensor_cpu)
        compressed_bytes = original_bytes
    else:
        compressor = get_compressor(compressor_name)
        compressed = compressor.compress(tensor_cpu, compression_param)
        compressed_bytes = None
        if hasattr(compressor, 'get_compressed_size'):
            try:
                compressed_bytes = int(compressor.get_compressed_size(compressed))
            except Exception:
                compressed_bytes = None
        if compressed_bytes is None:
            compressed_bytes = int(compressed.get('compressed_bytes', original_bytes))
        payload = {
            'mode': 'compressed',
            'compressor_name': compressor_name,
            'payload': compressed,
        }

    ratio = float(compressed_bytes) / float(original_bytes) if original_bytes > 0 else 1.0
    return payload, {
        'original_bytes': original_bytes,
        'compressed_bytes': compressed_bytes,
        'compression_ratio': ratio,
        'k_value': float(feature_k_value),
    }

def restore_activation_payload(payload, device):
    mode = payload.get('mode', 'compressed')
    if mode == 'identity':
        restored = payload['tensor']
    else:
        compressor = get_compressor(payload['compressor_name'])
        restored = compressor.decompress(payload['payload'])
    return restored.to(device)

def load_cifar10_batches(batch_size=DEFAULT_BATCH_SIZE,
                         max_batches=DEFAULT_MAX_BATCHES,
                         data_root=DEFAULT_DATA_ROOT,
                         download=DEFAULT_DOWNLOAD_DATA):
    try:
        import torchvision
        import torchvision.transforms as transforms
    except ImportError as exc:
        raise RuntimeError('torchvision is required for CIFAR-10 loading') from exc

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    try:
        dataset = torchvision.datasets.CIFAR10(
            root=data_root,
            train=False,
            download=download,
            transform=transform,
        )
    except PermissionError as exc:
        raise RuntimeError(
            'Unable to access CIFAR-10 under {}. The directory exists but this Python '
            'process cannot enumerate/read it.'.format(data_root)
        ) from exc
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    batches = []
    for batch_idx, (images, labels) in enumerate(loader):
        batches.append({
            'batch_idx': batch_idx,
            'images': images,
            'labels': labels,
        })
        if max_batches is not None and len(batches) >= max_batches:
            break
    return batches

def _collect_result_payload(transport):
    try:
        return transport.receive(
            msg_type=MessageType.FINAL_RESULT,
            source='D',
            timeout=0.0,
        )
    except TimeoutError:
        return None

def _drain_worker_stats(transport, pending, buffered_stats):
    while True:
        try:
            envelope = transport.receive(
                msg_type=MessageType.WORKER_STATS,
                source=None,
                timeout=0.0,
            )
        except TimeoutError:
            break

        payload = envelope['payload']
        task_id = payload['task_id']
        if task_id in pending:
            pending[task_id]['worker_stats'][payload['node_id']] = payload
        else:
            buffered_stats.setdefault(task_id, {})[payload['node_id']] = payload

def _profile_from_partition(partition, input_tensor):
    start = time.perf_counter()
    with torch.no_grad():
        output = partition(input_tensor)
    elapsed = time.perf_counter() - start
    return output, elapsed

def _worker_profile(node_id, compute_sec, restore_sec, prepare_sec, send_bytes, send_sec):
    return {
        'node': node_id,
        'compute_sec': float(compute_sec),
        'restore_sec': float(restore_sec),
        'prepare_send_sec': float(prepare_sec),
        'service_total_sec': float(compute_sec + restore_sec + prepare_sec),
        'send_packet_bytes': int(send_bytes),
        'send_packet_sec': float(send_sec),
    }

def _accuracy_from_predictions(predictions, labels):
    pred_tensor = torch.as_tensor(predictions, dtype=torch.long)
    labels = labels.detach().cpu().to(torch.long)
    correct = int((pred_tensor == labels).sum().item())
    total = int(labels.numel())
    return correct, total, (float(correct) / float(total) if total > 0 else 0.0)

def _save_rows(output_path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
