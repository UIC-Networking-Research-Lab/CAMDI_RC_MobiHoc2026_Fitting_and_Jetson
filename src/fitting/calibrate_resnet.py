"""Collect measured resnet accuracy for compression sweeps and eta mappings."""

from fitting.resnet import (
    _discover_supported_model_types,
    _resolve_model_types,
    _sanitize_tag,
    _ordered_policy_groups,
    _summary_output_path,
    _combined_summary_output_path,
    _cleanup_removed_llm_artifacts,
    _load_existing_accuracy_df,
    _llmint8_mapping_output_path,
    _save_llmint8_eta_mapping,
    _model_output_path,
    train_estimators_from_csv,
)

import argparse

import glob

import itertools

import json

import logging

import math

import os

import time

from collections import OrderedDict

import pandas as pd

from tqdm import tqdm

from fitting import estimator as accuracy_estimator_flex

from fitting.estimator import AccuracyEstimator

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

    class _FallbackModule(object):
        pass

    class _FallbackNN(object):
        Module = _FallbackModule
        ModuleList = list
        Sequential = tuple
        Conv2d = object

    nn = _FallbackNN()

try:
    from jetson_inference.common import resnet_arch as resnet20
except ImportError:
    resnet20 = None

try:
    from jetson_inference.common.compressors import get_compressor
except ImportError:
    get_compressor = None

NODE_ORDER = ['A', 'B', 'C', 'D']

NUM_PARTITIONS = 4

COMPRESSOR_NAME = 'quantization'

DEFAULT_BATCH_SIZE = 100

DEFAULT_DATA_ROOT = 'data'

DEFAULT_DOWNLOAD_DATA = False

ESTIMATOR_MODEL_DIR = 'acc_estimatior_fitting_models'

ESTIMATOR_MODEL_TYPE = 'linear_monotonic'

ESTIMATOR_PATH = os.path.join(ESTIMATOR_MODEL_DIR, 'jetson_resnet_3tp_poly3_flex.pkl')

K_LEVELS = [0.125, 0.25, 0.50, 0.75, 0.9, 1.0]

MODEL_CHECKPOINT = 'resnet56-4bfd9763.th'

RAW_ACCURACY_CSV = os.path.join(ESTIMATOR_MODEL_DIR, 'raw_accuracy_resnet56_quantization.csv')

def _require_runtime_dependencies(action_name):
    if torch is None or resnet20 is None or get_compressor is None:
        raise RuntimeError(
            '{} requires torch/resnet20/compressors runtime dependencies, '
            'but they are not available in the current Python environment.'.format(action_name)
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
    def __init__(self, checkpoint_path=MODEL_CHECKPOINT, device='cuda', num_partitions=NUM_PARTITIONS):
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
        node.partition_info = {
            'partition_idx': partition_idx,
            'unit_names': list(self.unit_names[start:end]),
            'is_last_partition': bool(partition_idx == self.num_partitions - 1),
            'node_id': NODE_ORDER[partition_idx],
            'checkpoint_path': self.checkpoint_path,
        }
        return node

    def build_partition_for_node(self, node_id):
        return self.build_partition(NODE_ORDER.index(node_id))

    def build_all_partitions(self):
        return [self.build_partition(idx) for idx in range(self.num_partitions)]

def load_cifar10_batches(batch_size=DEFAULT_BATCH_SIZE,
                         max_batches=None,
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
        batches.append({
            'batch_idx': batch_idx,
            'images': images,
            'labels': labels,
        })
        if max_batches is not None and len(batches) >= max_batches:
            break
    return batches

def _identity_activation_payload(tensor_cpu):
    return {'mode': 'identity', 'tensor': tensor_cpu}

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

def simulate_partitioned_batch(partitions, images, k_values, device='cuda', compressor_name=COMPRESSOR_NAME):
    x = images.to(device)
    transfer_stats = []
    tp_original_bytes = []
    stage_compute_sec = []

    with torch.no_grad():
        for idx, partition in enumerate(partitions):
            t_comp = time.perf_counter()
            x = partition(x)
            stage_compute_sec.append(time.perf_counter() - t_comp)

            if idx < len(partitions) - 1:
                payload, stats = build_activation_payload(
                    x,
                    compression_param=k_values[idx],
                    compressor_name=compressor_name,
                    feature_k_value=k_values[idx],
                )
                tp_original_bytes.append(stats['original_bytes'])
                transfer_stats.append(stats)
                x = restore_activation_payload(payload, device)

    return {
        'logits': x,
        'transfer_stats': transfer_stats,
        'tp_original_bytes': tp_original_bytes,
        'stage_compute_sec': stage_compute_sec,
        'total_latency': float(sum(stage_compute_sec)),
    }

def _default_estimator_model_type():
    preferred_order = ['linear_monotonic', 'poly3', 'poly2', 'gbm', 'rf', 'mlp_small', 'mlp']
    for model_type in preferred_order:
        if model_type in SUPPORTED_MODEL_TYPES:
            return model_type
    if not SUPPORTED_MODEL_TYPES:
        raise ValueError('No supported estimator model types discovered from accuracy_estimator_flex')
    return SUPPORTED_MODEL_TYPES[0]

SUPPORTED_MODEL_TYPES = _discover_supported_model_types()

DEFAULT_QUANTIZATION_K_LEVELS = [0.0625, 0.125, 0.25, 0.5, 1.0]

SUPPORTED_COMPRESSOR_NAMES = ['topk', 'quantization', 'llmint8', 'all']

DEFAULT_DEVICE = 'cuda'

DEFAULT_TRAINING_COMPRESSOR_NAME = 'quantization'

DEFAULT_LLMINT8_POLICY_NAMES = 'fp16_int2,fp16_int4,INT8INT2,INT8INT4'

DEFAULT_LLMINT8_OUTLIER_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]

DEFAULT_TRAIN_MODE = 'all'

DEFAULT_RESNET_RAW_CSV_GLOB = RAW_ACCURACY_CSV

DEFAULT_LLMINT8_MAPPING_PATH = os.path.join(
    ESTIMATOR_MODEL_DIR,
    'raw_accuracy_resnet56_llmint8_eta_mapping.json',
)

LLMINT8_POLICY_SPECS = OrderedDict([
    ('fp16_int2', {
        'outlier_precision': 'fp16',
        'regular_precision': 'int2',
    }),
    ('fp16_int4', {
        'outlier_precision': 'fp16',
        'regular_precision': 'int4',
    }),
    ('INT8INT4', {
        'outlier_precision': 'int8',
        'regular_precision': 'int4',
    }),
    ('INT8INT2', {
        'outlier_precision': 'int8',
        'regular_precision': 'int2',
    }),
])

LLMINT8_POLICY_ALIASES = {
    str(policy_name).strip().lower(): policy_name
    for policy_name in LLMINT8_POLICY_SPECS.keys()
}

LOCAL_SWEEP_BATCH_SIZE = 100

DEFAULT_ESTIMATOR_MODEL_TYPES = 'poly3,linear_monotonic'

def generate_k_candidates(k_levels=K_LEVELS):
    return [list(item) for item in itertools.product(k_levels, repeat=3)]

def _resolve_compressor_names(compressor_name_arg):
    names = [item.strip().lower() for item in str(compressor_name_arg).split(',') if item.strip()]
    if not names:
        raise ValueError('At least one compressor name must be provided.')
    if 'all' in names:
        return ['topk', 'quantization', 'llmint8']

    invalid = [item for item in names if item not in SUPPORTED_COMPRESSOR_NAMES]
    if invalid:
        raise ValueError(
            'Unsupported compressor names: {}. Choose from {}'.format(
                invalid,
                SUPPORTED_COMPRESSOR_NAMES,
            )
        )
    return list(OrderedDict.fromkeys(names))

def _resolve_llmint8_policies(policies_arg):
    if policies_arg is None:
        return list(LLMINT8_POLICY_SPECS.keys())

    if isinstance(policies_arg, (list, tuple)):
        raw_names = [str(item).strip() for item in policies_arg if str(item).strip()]
    else:
        raw_names = [item.strip() for item in str(policies_arg).split(',') if item.strip()]
    if not raw_names or any(item.lower() == 'all' for item in raw_names):
        return list(LLMINT8_POLICY_SPECS.keys())

    names = []
    invalid = []
    for item in raw_names:
        canonical_name = LLMINT8_POLICY_ALIASES.get(item.lower())
        if canonical_name is None:
            invalid.append(item)
        else:
            names.append(canonical_name)
    if invalid:
        raise ValueError(
            'Unsupported llmint8 policies: {}. Choose from {}'.format(
                invalid,
                list(LLMINT8_POLICY_SPECS.keys()),
            )
        )
    return list(OrderedDict.fromkeys(names))

def _quantization_precision_from_k(k_value):
    k_value = float(k_value)
    if k_value >= 1.0:
        return 'passthrough'
    if k_value >= 0.5:
        return 'fp16'
    if k_value >= 0.25:
        return 'int8'
    if k_value >= 0.125:
        return 'int4'
    return 'int2'

def _parse_float_levels(levels_arg, default_levels):
    if levels_arg is None:
        return list(default_levels)

    if isinstance(levels_arg, (list, tuple)):
        return [float(item) for item in levels_arg]

    values = [item.strip() for item in str(levels_arg).split(',') if item.strip()]
    if not values:
        raise ValueError('At least one float level must be provided.')
    return [float(item) for item in values]

def _round4(value):
    return round(float(value), 4)

def _cap_k(value):
    return min(1.0, float(value))

def _format_float4(value):
    return '{:.4f}'.format(float(value))

def _format_k_values(k_values):
    return ','.join(_format_float4(item) for item in k_values)

def _describe_candidate(candidate):
    compressor = candidate['compressor']
    if compressor == 'topk':
        return 'topk k={}'.format(_format_k_values(candidate['compression_params']))
    if compressor == 'quantization':
        return 'quantization k={} prec={}'.format(
            _format_k_values(candidate['compression_params']),
            ','.join(str(item) for item in candidate.get('quant_precision_values', [])),
        )

    if compressor == 'llmint8':
        return 'llmint8 policy={} outlier={}'.format(
            candidate.get('llm_policy', ''),
            _format_k_values(candidate.get('outlier_values', [])),
        )

    return str(candidate)

def _collect_transfer_stats(compressor, tensor_cpu, compression_params):
    original_bytes = int(tensor_cpu.numel() * tensor_cpu.element_size())
    if compression_params is None:
        compressed = {
            'tensor': tensor_cpu,
            'original_bytes': original_bytes,
            'compressed_bytes': original_bytes,
            'compression_ratio': 1.0,
        }
        return compressed, {
            'original_bytes': original_bytes,
            'compressed_bytes': original_bytes,
            'compression_ratio': 1.0,
        }
    compressed = compressor.compress(tensor_cpu, compression_params)

    compressed_bytes = None
    if hasattr(compressor, 'get_compressed_size'):
        try:
            compressed_bytes = int(compressor.get_compressed_size(compressed))
        except Exception:
            compressed_bytes = None
    if compressed_bytes is None:
        compressed_bytes = int(compressed.get('compressed_bytes', original_bytes))

    payload_ratio = float(compressed_bytes) / float(original_bytes) if original_bytes > 0 else 1.0
    return compressed, {
        'original_bytes': original_bytes,
        'compressed_bytes': compressed_bytes,
        'compression_ratio': payload_ratio,
    }

def simulate_partitioned_batch_llmint8(partitions, images, compression_params_list, device='cuda'):
    compressor = get_compressor('llmint8')
    x = images.to(device)
    transfer_stats = []
    tp_original_bytes = []
    stage_compute_sec = []

    with torch.no_grad():
        for idx, partition in enumerate(partitions):
            t_comp_start = time.perf_counter()
            x = partition(x)
            stage_compute_sec.append(time.perf_counter() - t_comp_start)

            if idx < len(partitions) - 1:
                payload, stats = _collect_transfer_stats(
                    compressor=compressor,
                    tensor_cpu=x.detach().cpu(),
                    compression_params=compression_params_list[idx],
                )
                tp_original_bytes.append(stats['original_bytes'])
                transfer_stats.append(stats)
                x = compressor.decompress(payload).to(device)

    return {
        'logits': x,
        'transfer_stats': transfer_stats,
        'tp_original_bytes': tp_original_bytes,
        'stage_compute_sec': stage_compute_sec,
        'total_latency': float(sum(stage_compute_sec)),
    }

def generate_llmint8_candidates(outlier_levels, policy_names):
    candidates = []
    for policy_name in policy_names:
        policy_spec = LLMINT8_POLICY_SPECS[policy_name]
        for outlier_values in itertools.product(outlier_levels, repeat=3):
            outlier_values = [float(item) for item in outlier_values]
            compression_params = [
                [
                    outlier_value,
                    policy_spec['outlier_precision'],
                    policy_spec['regular_precision'],
                ]
                for outlier_value in outlier_values
            ]
            candidates.append({
                'compressor': 'llmint8',
                'compressor_policy': 'llmint8_{}'.format(policy_name),
                'llm_policy': policy_name,
                'outlier_precision': policy_spec['outlier_precision'],
                'regular_precision': policy_spec['regular_precision'],
                'compression_params': compression_params,
                'outlier_values': outlier_values,
            })
    return candidates

def generate_quantization_candidates(quant_k_levels):
    candidates = []
    for k_values in generate_k_candidates(quant_k_levels):
        quant_precision_values = [_quantization_precision_from_k(item) for item in k_values]
        candidates.append({
            'compressor': 'quantization',
            'compressor_policy': 'quantization',
            'llm_policy': '',
            'outlier_precision': '',
            'regular_precision': '',
            'compression_params': list(k_values),
            'feature_k_values': [float(item) for item in k_values],
            'quant_precision_values': list(quant_precision_values),
            'outlier_values': [None, None, None],
        })
    return candidates

def _build_sweep_candidates(compressor_names, k_levels, llm_policies, llm_outlier_levels, quant_k_levels):
    candidates = []
    if 'llmint8' in compressor_names:
        candidates.extend(generate_llmint8_candidates(llm_outlier_levels, llm_policies))
    if 'topk' in compressor_names:
        for k_values in generate_k_candidates(k_levels):
            candidates.append({
                'compressor': 'topk',
                'compressor_policy': 'topk',
                'llm_policy': '',
                'outlier_precision': '',
                'regular_precision': '',
                'compression_params': list(k_values),
                'feature_k_values': [float(item) for item in k_values],
                'quant_precision_values': ['', '', ''],
                'outlier_values': [None, None, None],
            })
    if 'quantization' in compressor_names:
        candidates.extend(generate_quantization_candidates(quant_k_levels))
    return candidates

def _expected_compressor_policies(compressor_names, llm_policies):
    policies = []
    if 'llmint8' in compressor_names:
        for policy_name in llm_policies:
            policies.append('llmint8_{}'.format(policy_name))
    if 'quantization' in compressor_names:
        policies.append('quantization')
    if 'topk' in compressor_names:
        policies.append('topk')
    return policies

def _policy_raw_csv_path(base_csv_path, compressor_policy):
    directory = os.path.dirname(base_csv_path)
    stem, ext = os.path.splitext(os.path.basename(base_csv_path))
    suffix = _sanitize_tag(compressor_policy)
    filename = '{}_{}{}'.format(stem, suffix, ext or '.csv')
    return os.path.join(directory, filename)

def _existing_policy_coverage(csv_path):
    df = _load_existing_accuracy_df(csv_path)
    if df is None or df.empty:
        return set(), df

    if 'compressor_policy' in df.columns:
        values = df['compressor_policy'].dropna().astype(str)
        return set(values.tolist()), df

    if 'compressor' in df.columns:
        values = df['compressor'].dropna().astype(str)
        return set(values.tolist()), df

    return set(), df

def _missing_required_policies(csv_path, required_policies):
    existing_policies, _ = _existing_policy_coverage(csv_path)
    missing = [policy for policy in required_policies if policy not in existing_policies]
    return missing

def _prepare_accuracy_df_for_save(df):
    df_to_save = df.copy()
    for col in ['k0', 'k1', 'k2', 'outlier0', 'outlier1', 'outlier2']:
        if col in df_to_save.columns:
            df_to_save[col] = df_to_save[col].map(
                lambda value: '' if pd.isna(value) else _format_float4(value)
            )
    if 'outlier_values' in df_to_save.columns:
        df_to_save['outlier_values'] = df_to_save['outlier_values'].fillna('')
    if 'quant_precision_values' in df_to_save.columns:
        df_to_save['quant_precision_values'] = df_to_save['quant_precision_values'].fillna('')
    if 'llm_policy' in df_to_save.columns:
        df_to_save['llm_policy'] = df_to_save['llm_policy'].fillna('')
    for col in ['outlier_precision', 'regular_precision', 'quant0_precision', 'quant1_precision', 'quant2_precision']:
        if col in df_to_save.columns:
            df_to_save[col] = df_to_save[col].fillna('')
    return df_to_save

def _collect_candidate_rows(partitions, batches, candidates, device):
    rows = []

    for candidate in tqdm(candidates, desc='config-sweep'):
        total_correct = 0
        total_count = 0
        total_latency = 0.0
        total_ce_loss = 0.0
        tp_original_bytes = [0.0, 0.0, 0.0]
        tp_compressed_bytes = [0.0, 0.0, 0.0]

        for batch in batches:
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)
            if candidate['compressor'] == 'topk':
                simulation = simulate_partitioned_batch(
                    partitions=partitions,
                    images=images,
                    k_values=candidate['compression_params'],
                    device=device,
                    compressor_name='topk',
                )
            elif candidate['compressor'] == 'quantization':
                simulation = simulate_partitioned_batch(
                    partitions=partitions,
                    images=images,
                    k_values=candidate['compression_params'],
                    device=device,
                    compressor_name='quantization',
                )
            elif candidate['compressor'] == 'llmint8':
                simulation = simulate_partitioned_batch_llmint8(
                    partitions=partitions,
                    images=images,
                    compression_params_list=candidate['compression_params'],
                    device=device,
                )
            else:
                raise ValueError('Unsupported candidate compressor: {}'.format(candidate['compressor']))
            logits = simulation['logits']
            predictions = torch.argmax(logits, dim=1)
            total_correct += int((predictions == labels).sum().item())
            total_count += int(labels.numel())
            total_latency += float(simulation['total_latency'])
            total_ce_loss += float(F.cross_entropy(logits, labels, reduction='mean').item())

            for idx, stats in enumerate(simulation['transfer_stats']):
                tp_original_bytes[idx] += float(stats['original_bytes'])
                tp_compressed_bytes[idx] += float(stats['compressed_bytes'])

        avg_accuracy = float(total_correct) / float(total_count) if total_count > 0 else 0.0
        avg_latency = total_latency / max(1, len(batches))
        avg_ce_loss = total_ce_loss / max(1, len(batches))
        avg_compression_ratio = (
            sum(tp_original_bytes) / max(sum(tp_compressed_bytes), 1.0)
            if sum(tp_original_bytes) > 0 else 1.0
        )
        avg_payload_ratio = (
            sum(tp_compressed_bytes) / max(sum(tp_original_bytes), 1.0)
            if sum(tp_original_bytes) > 0 else 1.0
        )
        avg_payload_ratio = _cap_k(avg_payload_ratio)

        if candidate['compressor'] in ('topk', 'quantization'):
            realized_k_values = [_round4(_cap_k(item)) for item in candidate['feature_k_values']]
        else:
            realized_k_values = [
                _round4(_cap_k(tp_compressed_bytes[idx] / max(tp_original_bytes[idx], 1.0)))
                if tp_original_bytes[idx] > 0 else 1.0
                for idx in range(3)
            ]

        outlier_values = candidate.get('outlier_values', [None, None, None])
        quant_precision_values = candidate.get('quant_precision_values', ['', '', ''])
        outlier_values_text = (
            _format_k_values(outlier_values)
            if all(value is not None for value in outlier_values) else ''
        )

        rows.append({
            'compressor': candidate['compressor'],
            'compressor_policy': candidate['compressor_policy'],
            'llm_policy': candidate.get('llm_policy', ''),
            'outlier_precision': candidate.get('outlier_precision', ''),
            'regular_precision': candidate.get('regular_precision', ''),
            'outlier_values': outlier_values_text,
            'outlier0': outlier_values[0],
            'outlier1': outlier_values[1],
            'outlier2': outlier_values[2],
            'quant_precision_values': ','.join(str(item) for item in quant_precision_values if str(item)),
            'quant0_precision': quant_precision_values[0] if len(quant_precision_values) > 0 else '',
            'quant1_precision': quant_precision_values[1] if len(quant_precision_values) > 1 else '',
            'quant2_precision': quant_precision_values[2] if len(quant_precision_values) > 2 else '',
            'k_values': _format_k_values(realized_k_values),
            'k0': realized_k_values[0],
            'k1': realized_k_values[1],
            'k2': realized_k_values[2],
            'num_batches': len(batches),
            'num_samples': total_count,
            'correct_count': total_correct,
            'avg_accuracy': avg_accuracy,
            'avg_latency': avg_latency,
            'avg_ce_loss': avg_ce_loss,
            'avg_payload_ratio': avg_payload_ratio,
            'tp0_original_bytes': tp_original_bytes[0],
            'tp1_original_bytes': tp_original_bytes[1],
            'tp2_original_bytes': tp_original_bytes[2],
            'tp0_compressed_bytes': tp_compressed_bytes[0],
            'tp1_compressed_bytes': tp_compressed_bytes[1],
            'tp2_compressed_bytes': tp_compressed_bytes[2],
            'tp0_payload_ratio': (
                _round4(_cap_k(tp_compressed_bytes[0] / max(tp_original_bytes[0], 1.0)))
                if tp_original_bytes[0] > 0 else 1.0
            ),
            'tp1_payload_ratio': (
                _round4(_cap_k(tp_compressed_bytes[1] / max(tp_original_bytes[1], 1.0)))
                if tp_original_bytes[1] > 0 else 1.0
            ),
            'tp2_payload_ratio': (
                _round4(_cap_k(tp_compressed_bytes[2] / max(tp_original_bytes[2], 1.0)))
                if tp_original_bytes[2] > 0 else 1.0
            ),
            'avg_compression_ratio': avg_compression_ratio,
        })
        logging.info(
            'Completed candidate: %s | realized_k=%s | acc=%.6f | latency=%.6f | payload_ratio=%.4f',
            _describe_candidate(candidate),
            _format_k_values(realized_k_values),
            avg_accuracy,
            avg_latency,
            avg_payload_ratio,
        )

    return rows

def validate_partition_equivalence(checkpoint_path=MODEL_CHECKPOINT,
                                   batch_size=8,
                                   num_batches=2,
                                   data_root=DEFAULT_DATA_ROOT,
                                   download_data=DEFAULT_DOWNLOAD_DATA,
                                   device=DEFAULT_DEVICE):
    _require_runtime_dependencies('validate_partition_equivalence')
    factory = ResNet56PartitionFactory(checkpoint_path=checkpoint_path, device=device)
    partitions = factory.build_all_partitions()
    batches = load_cifar10_batches(
        batch_size=batch_size,
        max_batches=num_batches,
        data_root=data_root,
        download=download_data,
    )
    reports = []
    for batch in batches:
        images = batch['images'].to(device)
        with torch.no_grad():
            reference_logits = factory.full_model(images)
            partitioned_logits = images
            for partition in partitions:
                partitioned_logits = partition(partitioned_logits)

        reports.append({
            'batch_idx': batch['batch_idx'],
            'max_abs_diff': float((reference_logits - partitioned_logits).abs().max().item()),
            'top1_pred_equal': bool(torch.equal(
                torch.argmax(reference_logits, dim=1),
                torch.argmax(partitioned_logits, dim=1),
            )),
        })

    passed = all(item['top1_pred_equal'] for item in reports)
    return {
        'reports': reports,
        'passed': passed,
        'partition_plan': factory.partition_plan,
    }

def print_validation_report(report):
    print('=' * 80)
    print('ResNet56 partition validation')
    print('=' * 80)
    for item in report['reports']:
        print(
            'batch={batch_idx:02d} max_abs_diff={max_abs_diff:.8f} top1_pred_equal={top1_pred_equal}'.format(**item)
        )
    print('passed={}'.format(report['passed']))

def collect_raw_accuracy_data(output_csv=RAW_ACCURACY_CSV,
                              checkpoint_path=MODEL_CHECKPOINT,
                              batch_size=DEFAULT_BATCH_SIZE,
                              max_batches=None,
                              data_root=DEFAULT_DATA_ROOT,
                              download_data=DEFAULT_DOWNLOAD_DATA,
                              device=DEFAULT_DEVICE,
                              compressor_name=COMPRESSOR_NAME,
                              k_levels=None,
                              quant_k_levels=None,
                              llm_policies=None,
                              llm_outlier_levels=None,
                              target_policies=None):
    _require_runtime_dependencies('collect_raw_accuracy_data')
    factory = ResNet56PartitionFactory(checkpoint_path=checkpoint_path, device=device)
    partitions = factory.build_all_partitions()
    batches = load_cifar10_batches(
        batch_size=batch_size,
        max_batches=max_batches,
        data_root=data_root,
        download=download_data,
    )
    resolved_compressors = _resolve_compressor_names(compressor_name)
    resolved_policies = _resolve_llmint8_policies(llm_policies)
    resolved_outlier_levels = _parse_float_levels(
        llm_outlier_levels,
        DEFAULT_LLMINT8_OUTLIER_LEVELS,
    )
    resolved_quant_k_levels = _parse_float_levels(
        quant_k_levels,
        DEFAULT_QUANTIZATION_K_LEVELS,
    )
    candidates = _build_sweep_candidates(
        compressor_names=resolved_compressors,
        k_levels=k_levels or K_LEVELS,
        llm_policies=resolved_policies,
        llm_outlier_levels=resolved_outlier_levels,
        quant_k_levels=resolved_quant_k_levels,
    )
    if target_policies is not None:
        target_policy_set = set(target_policies)
        candidates = [item for item in candidates if item['compressor_policy'] in target_policy_set]

    logging.info(
        'Offline sweep: %d candidates x %d batches | compressors=%s',
        len(candidates),
        len(batches),
        resolved_compressors,
    )
    if not candidates:
        existing_df = _load_existing_accuracy_df(output_csv)
        return existing_df if existing_df is not None else pd.DataFrame()

    rows = _collect_candidate_rows(
        partitions=partitions,
        batches=batches,
        candidates=candidates,
        device=device,
    )

    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    policy_df = pd.DataFrame(rows).sort_values(['avg_accuracy', 'avg_latency'], ascending=[False, True])
    if policy_df.empty:
        return policy_df

    compressor_policy = str(policy_df['compressor_policy'].iloc[0])
    policy_csv_path = _policy_raw_csv_path(output_csv, compressor_policy)
    _prepare_accuracy_df_for_save(policy_df).to_csv(policy_csv_path, index=False)
    logging.info('Saved raw accuracy CSV for %s to %s', compressor_policy, policy_csv_path)

    existing_df = _load_existing_accuracy_df(output_csv)
    if existing_df is None or existing_df.empty:
        combined_df = policy_df.copy()
    else:
        existing_df = existing_df.copy()
        if 'compressor_policy' not in existing_df.columns:
            existing_df['compressor_policy'] = existing_df.get('compressor', 'topk')
        combined_df = pd.concat(
            [existing_df[existing_df['compressor_policy'].astype(str) != compressor_policy], policy_df],
            ignore_index=True,
        )

    if not combined_df.empty:
        combined_df = combined_df.sort_values(
            ['compressor_policy', 'avg_accuracy', 'avg_latency'],
            ascending=[True, False, True],
        )
    _prepare_accuracy_df_for_save(combined_df).to_csv(output_csv, index=False)
    logging.info('Updated combined raw accuracy CSV at %s after %s', output_csv, compressor_policy)
    mapping_output_path = _save_llmint8_eta_mapping(combined_df, output_csv)
    if mapping_output_path is not None:
        logging.info('Saved llmint8 eta mapping to %s', mapping_output_path)
    return combined_df

def main():
    parser = argparse.ArgumentParser(description='Offline local training for the ResNet 3-cutpoint accuracy estimator.')
    parser.add_argument('--mode', choices=['check', 'collect', 'train', 'all'], default=DEFAULT_TRAIN_MODE)
    parser.add_argument('--checkpoint_path', default=MODEL_CHECKPOINT)
    parser.add_argument('--batch_size', type=int, default=LOCAL_SWEEP_BATCH_SIZE)
    parser.add_argument('--max_batches', type=int, default=None)
    parser.add_argument('--check_batches', type=int, default=2)
    parser.add_argument('--skip_check', action='store_true', help='Skip the full-model vs partitioned-model validation')
    parser.add_argument('--data_root', default=DEFAULT_DATA_ROOT)
    parser.add_argument('--download_data', action='store_true', help='Allow torchvision to download CIFAR-10')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=DEFAULT_DEVICE)
    parser.add_argument('--compressor_name', default=DEFAULT_TRAINING_COMPRESSOR_NAME,
                        help='topk, llmint8, all, or a comma-separated subset')
    parser.add_argument('--quant_k_levels', default=','.join(str(item) for item in DEFAULT_QUANTIZATION_K_LEVELS),
                        help='Comma-separated quantization k boundary levels')
    parser.add_argument('--llm_policies', default=DEFAULT_LLMINT8_POLICY_NAMES,
                        help='Comma-separated llmint8 policy names, or all')
    parser.add_argument('--llm_outlier_levels', default='0,0.25,0.5,0.75,1',
                        help='Comma-separated llmint8 outlier ratios')
    parser.add_argument('--csv_path', default=DEFAULT_RESNET_RAW_CSV_GLOB)
    parser.add_argument('--output_dir', default=ESTIMATOR_MODEL_DIR)
    parser.add_argument('--model_types', default=DEFAULT_ESTIMATOR_MODEL_TYPES)
    parser.add_argument('--force_collect', action='store_true', help='Run the local sweep even if the raw CSV already exists')
    parser.add_argument('--log_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING'])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )

    if args.mode in ('check', 'collect', 'all'):
        _require_runtime_dependencies('mode={}'.format(args.mode))
    if args.mode in ('check', 'collect', 'all') and args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Default device is cuda, but torch.cuda.is_available() is False. Use --device cpu explicitly.')

    if args.mode in ('check', 'collect', 'all') and not args.skip_check:
        report = validate_partition_equivalence(
            checkpoint_path=args.checkpoint_path,
            batch_size=min(args.batch_size, 8),
            num_batches=args.check_batches,
            data_root=args.data_root,
            download_data=args.download_data,
            device=args.device,
        )
        print_validation_report(report)
        if not report['passed']:
            raise RuntimeError('Partition validation failed. Refusing to continue with estimator data collection.')

    if args.mode == 'check':
        return

    resolved_compressors = _resolve_compressor_names(args.compressor_name)
    resolved_policies = _resolve_llmint8_policies(args.llm_policies)
    required_policies = _expected_compressor_policies(resolved_compressors, resolved_policies)
    logging.info(
        'Training data requirements: compressors=%s required_policies=%s',
        resolved_compressors,
        required_policies,
    )

    if args.mode in ('collect', 'all'):
        if os.path.isdir(args.csv_path):
            raise ValueError(
                '--csv_path currently must be a file path for collect/all mode; got directory: {}'.format(args.csv_path)
            )
        missing_policies = _missing_required_policies(args.csv_path, required_policies)
        if missing_policies and os.path.exists(args.csv_path):
            logging.info(
                'Raw accuracy data at %s is incomplete. Missing policies=%s. Running local sweep.',
                args.csv_path,
                missing_policies,
            )

        for compressor_policy in required_policies:
            need_collect = args.force_collect or (compressor_policy in missing_policies)
            if need_collect:
                logging.info('Collecting raw accuracy data for policy=%s', compressor_policy)
                collect_raw_accuracy_data(
                    output_csv=args.csv_path,
                    checkpoint_path=args.checkpoint_path,
                    batch_size=args.batch_size,
                    max_batches=args.max_batches,
                    data_root=args.data_root,
                    download_data=args.download_data,
                    device=args.device,
                    compressor_name=args.compressor_name,
                    quant_k_levels=args.quant_k_levels,
                    llm_policies=resolved_policies,
                    llm_outlier_levels=args.llm_outlier_levels,
                    target_policies=[compressor_policy],
                )
            else:
                logging.info('Skipping collection for policy=%s because CSV data already exists', compressor_policy)

            if args.mode == 'all':
                logging.info('Training estimator(s) for policy=%s', compressor_policy)
                _, policy_summary_df = train_estimators_from_csv(
                    csv_path=args.csv_path,
                    output_dir=args.output_dir,
                    model_types=args.model_types,
                    required_policies=[compressor_policy],
                    only_required_policies=True,
                    active_policies=required_policies,
                )
                print(policy_summary_df.to_string(index=False))

    if args.mode == 'train':
        for compressor_policy in required_policies:
            logging.info('Training estimator(s) for policy=%s', compressor_policy)
            _, policy_summary_df = train_estimators_from_csv(
                csv_path=args.csv_path,
                output_dir=args.output_dir,
                model_types=args.model_types,
                required_policies=[compressor_policy],
                only_required_policies=True,
                active_policies=required_policies,
            )
            print(policy_summary_df.to_string(index=False))
