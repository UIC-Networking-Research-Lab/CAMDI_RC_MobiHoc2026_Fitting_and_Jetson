"""Collect measured flan_t5 accuracy for compression sweeps and eta mappings."""

from fitting.flan_t5 import (
    _sanitize_tag,
    _parse_model_types,
    _ordered_policy_groups,
    _summary_output_path,
    _combined_summary_output_path,
    _model_output_path,
    _llmint8_mapping_output_path,
    _save_llmint8_eta_mapping,
    train_estimators_from_csv,
)

import argparse

import csv

import inspect

import itertools

import json

import logging

import os

import statistics

import sys

import time

from collections import OrderedDict

from dataclasses import dataclass

import pandas as pd

import torch

import torch.nn as nn

from tqdm import tqdm

from fitting.estimator import AccuracyEstimator

from jetson_inference.common.compressors import get_compressor

DEFAULT_MODEL_NAME = 'google/flan-t5-base'

DEFAULT_DEVICE = 'cpu'

DEFAULT_SPLIT = 'validation'

DEFAULT_PROMPT_TEMPLATE = (
    'Classify the sentiment of the following sentence as positive or negative.\n'
    'Sentence: {sentence}\n'
    'Sentiment:'
)

DEFAULT_POSITIVE_TOKEN = 'positive'

DEFAULT_NEGATIVE_TOKEN = 'negative'

DEFAULT_MAX_INPUT_LENGTH = 128

DEFAULT_MAX_SAMPLES = 0

DEFAULT_VERBALIZER_CANDIDATES = [
    ('positive', 'negative'),
    ('yes', 'no'),
    ('true', 'false'),
]

NUM_TRANSFER_POINTS = 3

K_LEVELS = [0.125, 0.25, 0.50, 0.75, 1.0]

SUPPORTED_COMPRESSOR_NAMES = ['topk', 'quantization', 'llmint8', 'all']

SUPPORTED_MODEL_TYPES = ['linear_monotonic', 'poly3']

DEFAULT_COMPRESSOR_NAME = 'quantization'

DEFAULT_MODEL_TYPES = 'poly3,linear_monotonic'

DEFAULT_LLMINT8_POLICY_NAMES = 'fp16_int8'

DEFAULT_LLMINT8_OUTLIER_LEVELS = [0.01, 0.25, 0.5, 1.0]

DEFAULT_QUANTIZATION_K_LEVELS = [0.0625, 0.125, 0.25, 0.5, 1.0]

ESTIMATOR_MODEL_DIR = 'acc_estimatior_fitting_models'

RAW_ACCURACY_CSV = os.path.join(ESTIMATOR_MODEL_DIR, 'raw_accuracy_flan_t5_sst2_3tp_quantization.csv')

LLMINT8_POLICY_SPECS = OrderedDict([
    ('fp16_int8', {
        'outlier_precision': 'fp16',
        'regular_precision': 'int8',
    }),
])

LLMINT8_POLICY_ALIASES = {
    str(policy_name).strip().lower(): policy_name
    for policy_name in LLMINT8_POLICY_SPECS.keys()
}

LLMINT8_POLICY_ALIASES['fp16int8'] = 'fp16_int8'

_TRANSFORMERS_RUNTIME = None

@dataclass
class SST2Sample:
    sample_id: str
    sentence: str
    label: int

class TransformersRuntime:
    def __init__(self):
        try:
            from transformers import AutoTokenizer, T5ForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                'transformers is required in interpreter {}.'.format(sys.executable)
            ) from exc

        try:
            from transformers.models.t5.modeling_t5 import (
                create_bidirectional_mask,
                create_causal_mask,
            )
        except Exception:
            create_bidirectional_mask = None
            create_causal_mask = None

        self.AutoTokenizer = AutoTokenizer
        self.T5ForConditionalGeneration = T5ForConditionalGeneration
        self.create_bidirectional_mask = create_bidirectional_mask
        self.create_causal_mask = create_causal_mask

def get_transformers_runtime():
    global _TRANSFORMERS_RUNTIME
    if _TRANSFORMERS_RUNTIME is None:
        _TRANSFORMERS_RUNTIME = TransformersRuntime()
    return _TRANSFORMERS_RUNTIME

def supports_parameter(fn, name):
    try:
        return name in inspect.signature(fn).parameters
    except Exception:
        return False

def build_extended_attention_mask_compat(stack_module, attention_mask, hidden_states):
    fn = getattr(stack_module, 'get_extended_attention_mask', None)
    if fn is None:
        return attention_mask

    kwargs = {}
    if supports_parameter(fn, 'dtype'):
        kwargs['dtype'] = hidden_states.dtype
    elif supports_parameter(fn, 'device'):
        kwargs['device'] = hidden_states.device

    return fn(attention_mask, hidden_states.shape[:2], **kwargs)

def normalize_attention_mask(input_ids, attention_mask, device):
    if attention_mask is None:
        return torch.ones_like(input_ids, dtype=torch.long, device=device)
    return attention_mask.to(device)

def tensor_payload_bytes(tensor):
    return int(tensor.numel() * tensor.element_size())

def _round4(value):
    return round(float(value), 4)

def _cap_k(value):
    return min(1.0, float(value))

def _format_float4(value):
    return '{:.4f}'.format(float(value))

def _format_k_values(k_values):
    return ','.join(_format_float4(item) for item in k_values)

def _resolve_compressor_names(compressor_name_arg):
    names = [item.strip().lower() for item in str(compressor_name_arg).split(',') if item.strip()]
    if not names:
        raise ValueError('At least one compressor name must be provided.')
    if 'all' in names:
        return ['topk', 'quantization', 'llmint8']
    invalid = [item for item in names if item not in SUPPORTED_COMPRESSOR_NAMES]
    if invalid:
        raise ValueError('Unsupported compressor names: {}. Choose from {}'.format(invalid, SUPPORTED_COMPRESSOR_NAMES))
    return list(OrderedDict.fromkeys(names))

def _resolve_llmint8_policies(policies_arg):
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
        raise ValueError('Unsupported llmint8 policies: {}. Choose from {}'.format(
            invalid,
            list(LLMINT8_POLICY_SPECS.keys()),
        ))
    return list(OrderedDict.fromkeys(names))

def _parse_float_levels(levels_arg, default_levels):
    if levels_arg is None:
        return list(default_levels)
    values = [item.strip() for item in str(levels_arg).split(',') if item.strip()]
    if not values:
        raise ValueError('At least one float level must be provided.')
    return [float(item) for item in values]

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

def _parse_sample_records(data):
    if isinstance(data, dict):
        if 'samples' in data and isinstance(data['samples'], list):
            return data['samples']
        if 'data' in data and isinstance(data['data'], list):
            return data['data']
    if isinstance(data, list):
        return data
    raise ValueError('Unsupported JSON dataset format.')

def _extract_sentence_label(record, index):
    sentence = None
    for key in ('sentence', 'text', 'input_text'):
        value = record.get(key)
        if value is not None:
            sentence = str(value).strip()
            break
    if not sentence:
        raise ValueError('Record {} is missing sentence/text field.'.format(index))
    if 'label' not in record:
        raise ValueError('Record {} is missing label field.'.format(index))
    label = int(record['label'])
    if label not in (0, 1):
        raise ValueError('Record {} has non-binary label: {}'.format(index, label))
    return SST2Sample(sample_id=str(record.get('id', index)), sentence=sentence, label=label)

def load_local_sst2_samples(dataset_path):
    ext = os.path.splitext(dataset_path)[1].lower()
    if ext == '.json':
        with open(dataset_path, 'r', encoding='utf-8') as handle:
            records = _parse_sample_records(json.load(handle))
    elif ext == '.jsonl':
        records = []
        with open(dataset_path, 'r', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    elif ext in ('.csv', '.tsv'):
        delimiter = ',' if ext == '.csv' else '\t'
        with open(dataset_path, 'r', encoding='utf-8', newline='') as handle:
            records = list(csv.DictReader(handle, delimiter=delimiter))
    else:
        raise ValueError('Unsupported dataset extension: {}'.format(ext))
    return [_extract_sentence_label(record, idx) for idx, record in enumerate(records)]

def load_hf_sst2_samples(split=DEFAULT_SPLIT):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            'datasets is required when --dataset_path is not provided. Interpreter={}'.format(sys.executable)
        ) from exc
    dataset = load_dataset('glue', 'sst2', split=split)
    return [
        SST2Sample(
            sample_id=str(record.get('idx', index)),
            sentence=str(record['sentence']).strip(),
            label=int(record['label']),
        )
        for index, record in enumerate(dataset)
    ]

def load_sst2_samples(dataset_path=None, split=DEFAULT_SPLIT):
    if dataset_path:
        return load_local_sst2_samples(dataset_path)
    return load_hf_sst2_samples(split=split)

def build_prompt(sentence, prompt_template):
    return prompt_template.format(sentence=sentence.strip())

def resolve_single_token_verbalizers(tokenizer, positive_text=None, negative_text=None):
    candidates = []
    if positive_text is not None or negative_text is not None:
        if not positive_text or not negative_text:
            raise ValueError('Both positive_token and negative_token must be provided together.')
        candidates.append((str(positive_text), str(negative_text)))
    else:
        candidates.append((DEFAULT_POSITIVE_TOKEN, DEFAULT_NEGATIVE_TOKEN))
    candidates.extend(DEFAULT_VERBALIZER_CANDIDATES)

    seen = set()
    for positive_candidate, negative_candidate in candidates:
        key = (str(positive_candidate), str(negative_candidate))
        if key in seen:
            continue
        seen.add(key)
        positive_ids = tokenizer.encode(str(positive_candidate), add_special_tokens=False)
        negative_ids = tokenizer.encode(str(negative_candidate), add_special_tokens=False)
        if len(positive_ids) != 1 or len(negative_ids) != 1:
            continue
        if positive_ids[0] == negative_ids[0]:
            continue
        return {
            'positive_text': str(positive_candidate),
            'negative_text': str(negative_candidate),
            'positive_id': int(positive_ids[0]),
            'negative_id': int(negative_ids[0]),
        }

    raise RuntimeError('No valid single-token verbalizer pair found for the current tokenizer.')

def decide_label(logits, verbalizers):
    positive_logit = logits[verbalizers['positive_id']]
    negative_logit = logits[verbalizers['negative_id']]
    return 1 if float(positive_logit.item()) >= float(negative_logit.item()) else 0

def build_encoder_mask(runtime, stack_module, config, hidden_states, attention_mask):
    if runtime.create_bidirectional_mask is not None:
        try:
            return runtime.create_bidirectional_mask(
                config=config,
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
            )
        except TypeError:
            pass
    if hasattr(stack_module, 'get_extended_attention_mask'):
        return build_extended_attention_mask_compat(
            stack_module,
            attention_mask,
            hidden_states,
        )
    return attention_mask

def build_cross_attention_mask(runtime, stack_module, config, hidden_states, encoder_hidden_states, encoder_attention_mask):
    if runtime.create_bidirectional_mask is not None:
        try:
            return runtime.create_bidirectional_mask(
                config=config,
                inputs_embeds=hidden_states,
                attention_mask=encoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
            )
        except TypeError:
            pass
    if hasattr(stack_module, 'invert_attention_mask'):
        return stack_module.invert_attention_mask(encoder_attention_mask)
    return encoder_attention_mask

def build_decoder_mask(runtime, stack_module, config, hidden_states, decoder_attention_mask):
    if hasattr(stack_module, 'get_extended_attention_mask'):
        return build_extended_attention_mask_compat(
            stack_module,
            decoder_attention_mask,
            hidden_states,
        )

    if runtime.create_causal_mask is not None:
        try:
            cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
            return runtime.create_causal_mask(
                config=config,
                inputs_embeds=hidden_states,
                attention_mask=decoder_attention_mask,
                cache_position=cache_position,
                past_key_values=None,
            )
        except TypeError as exc:
            raise RuntimeError(
                'create_causal_mask signature mismatch for decoder mask construction. '
                'transformers runtime is incompatible with this code path.'
            ) from exc
    return decoder_attention_mask

def apply_transport_compression(tensor, compressor, compression_param, device):
    original_bytes = tensor_payload_bytes(tensor)
    if compressor is None or compression_param is None:
        return tensor, {
            'original_bytes': original_bytes,
            'compressed_bytes': original_bytes,
            'payload_ratio': 1.0,
        }
    compressed = compressor.compress(tensor.detach().cpu(), compression_param)
    if hasattr(compressor, 'get_compressed_size'):
        compressed_bytes = int(compressor.get_compressed_size(compressed))
    else:
        compressed_bytes = int(compressed.get('compressed_bytes', original_bytes))
    restored = compressor.decompress(compressed).to(device)
    payload_ratio = float(compressed_bytes) / float(original_bytes) if original_bytes > 0 else 1.0
    return restored, {
        'original_bytes': original_bytes,
        'compressed_bytes': compressed_bytes,
        'payload_ratio': payload_ratio,
    }

ENCODER_RANGES = [(0, 4), (4, 8), (8, 12)]

DECODER_RANGES = [(0, 12)]

TRANSFER_POINT_LABELS = {
    'tp0': 'Enc0 -> Enc1',
    'tp1': 'Enc1 -> Enc2',
    'tp2': 'Enc2 -> Decoder',
}

class FlanT5EncoderPartition(nn.Module):
    def __init__(self, full_model, layer_indices, is_first_partition, is_last_partition, runtime):
        super().__init__()
        self.layer_indices = list(layer_indices)
        self.is_first_partition = bool(is_first_partition)
        self.is_last_partition = bool(is_last_partition)
        self.runtime = runtime
        self.model_config = full_model.config
        self.stack_module = full_model.encoder
        self.blocks = [full_model.encoder.block[idx] for idx in self.layer_indices]
        self.embed_tokens = full_model.encoder.embed_tokens if self.is_first_partition else None
        self.dropout = full_model.encoder.dropout if self.is_first_partition else None
        self.final_layer_norm = full_model.encoder.final_layer_norm if self.is_last_partition else None
        self.final_dropout = full_model.encoder.dropout if self.is_last_partition else None
        sample_block = self.blocks[0]
        self.supports_return_dict = supports_parameter(sample_block.forward, 'return_dict')
        self.supports_layer_head_mask = supports_parameter(sample_block.forward, 'layer_head_mask')
        self.supports_cross_attn_layer_head_mask = supports_parameter(sample_block.forward, 'cross_attn_layer_head_mask')
        self.supports_cache_position = supports_parameter(sample_block.forward, 'cache_position')

    def forward(self, hidden_states, attention_mask=None, position_bias=None):
        if self.embed_tokens is not None and hidden_states.dtype == torch.long:
            hidden_states = self.embed_tokens(hidden_states)
            hidden_states = self.dropout(hidden_states)

        if attention_mask is None:
            attention_mask = torch.ones(
                hidden_states.shape[0],
                hidden_states.shape[1],
                dtype=torch.long,
                device=hidden_states.device,
            )

        block_attention_mask = build_encoder_mask(
            runtime=self.runtime,
            stack_module=self.stack_module,
            config=self.model_config,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)

        for block in self.blocks:
            block_kwargs = {
                'hidden_states': hidden_states,
                'attention_mask': block_attention_mask,
                'position_bias': position_bias,
                'encoder_hidden_states': None,
                'encoder_attention_mask': None,
                'use_cache': False,
                'output_attentions': False,
            }
            if self.supports_return_dict:
                block_kwargs['return_dict'] = False
            if self.supports_layer_head_mask:
                block_kwargs['layer_head_mask'] = None
            if self.supports_cross_attn_layer_head_mask:
                block_kwargs['cross_attn_layer_head_mask'] = None
            if self.supports_cache_position:
                block_kwargs['cache_position'] = cache_position

            outputs = block(**block_kwargs)
            hidden_states = outputs[0]
            position_bias = outputs[1] if len(outputs) > 1 else None

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)
            hidden_states = self.final_dropout(hidden_states)

        return hidden_states, position_bias

class FlanT5DecoderPartition(nn.Module):
    def __init__(self, full_model, layer_indices, is_first_partition, is_last_partition, runtime):
        super().__init__()
        self.layer_indices = list(layer_indices)
        self.is_first_partition = bool(is_first_partition)
        self.is_last_partition = bool(is_last_partition)
        self.runtime = runtime
        self.model_config = full_model.config
        self.stack_module = full_model.decoder
        self.model_dim = getattr(full_model, 'model_dim', full_model.config.d_model)
        self.blocks = [full_model.decoder.block[idx] for idx in self.layer_indices]
        self.embed_tokens = full_model.decoder.embed_tokens if self.is_first_partition else None
        self.dropout = full_model.decoder.dropout if self.is_first_partition else None
        self.final_layer_norm = full_model.decoder.final_layer_norm if self.is_last_partition else None
        self.final_dropout = full_model.decoder.dropout if self.is_last_partition else None
        self.lm_head = full_model.lm_head if self.is_last_partition else None
        self.cache_by_task = {}
        sample_block = self.blocks[0]
        self.supports_return_dict = supports_parameter(sample_block.forward, 'return_dict')
        self.supports_layer_head_mask = supports_parameter(sample_block.forward, 'layer_head_mask')
        self.supports_cross_attn_layer_head_mask = supports_parameter(sample_block.forward, 'cross_attn_layer_head_mask')
        self.supports_encoder_decoder_position_bias = supports_parameter(sample_block.forward, 'encoder_decoder_position_bias')
        self.supports_cache_position = supports_parameter(sample_block.forward, 'cache_position')

    def _get_or_create_cache(self, task_id):
        if task_id not in self.cache_by_task:
            self.cache_by_task[task_id] = [None for _ in self.layer_indices]
        return self.cache_by_task[task_id]

    def clear_cache(self, task_id):
        if task_id in self.cache_by_task:
            del self.cache_by_task[task_id]

    def clear_all_cache(self):
        self.cache_by_task.clear()

    def forward(self, hidden_states, encoder_hidden_states, task_id,
                encoder_attention_mask=None, position_bias=None,
                encoder_decoder_position_bias=None):
        self._get_or_create_cache(task_id)

        if self.embed_tokens is not None and hidden_states.dtype == torch.long:
            hidden_states = self.embed_tokens(hidden_states)
            hidden_states = self.dropout(hidden_states)

        decoder_attention_mask = torch.ones(
            hidden_states.shape[0],
            hidden_states.shape[1],
            dtype=torch.long,
            device=hidden_states.device,
        )
        cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        self_attention_mask = build_decoder_mask(
            runtime=self.runtime,
            stack_module=self.stack_module,
            config=self.model_config,
            hidden_states=hidden_states,
            decoder_attention_mask=decoder_attention_mask,
        )

        cross_attention_mask = None
        if encoder_attention_mask is not None:
            cross_attention_mask = build_cross_attention_mask(
                runtime=self.runtime,
                stack_module=self.stack_module,
                config=self.model_config,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )

        for block in self.blocks:
            block_kwargs = {
                'hidden_states': hidden_states,
                'attention_mask': self_attention_mask,
                'position_bias': position_bias,
                'encoder_hidden_states': encoder_hidden_states,
                'encoder_attention_mask': cross_attention_mask,
                'use_cache': False,
                'output_attentions': False,
            }
            if self.supports_return_dict:
                block_kwargs['return_dict'] = False
            if self.supports_layer_head_mask:
                block_kwargs['layer_head_mask'] = None
            if self.supports_cross_attn_layer_head_mask:
                block_kwargs['cross_attn_layer_head_mask'] = None
            if self.supports_encoder_decoder_position_bias:
                block_kwargs['encoder_decoder_position_bias'] = encoder_decoder_position_bias
            if self.supports_cache_position:
                block_kwargs['cache_position'] = cache_position

            outputs = block(**block_kwargs)
            hidden_states = outputs[0]
            position_bias = outputs[1] if len(outputs) > 1 else None
            encoder_decoder_position_bias = outputs[2] if len(outputs) > 2 else None

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)
            hidden_states = self.final_dropout(hidden_states)
            if self.lm_head is not None:
                if getattr(self.model_config, 'scale_decoder_outputs', False) or (
                    not hasattr(self.model_config, 'scale_decoder_outputs') and
                    getattr(self.model_config, 'tie_word_embeddings', False)
                ):
                    hidden_states = hidden_states * (float(self.model_dim) ** -0.5)
                hidden_states = self.lm_head(hidden_states)

        return hidden_states, position_bias, encoder_decoder_position_bias

class FlanT5SST2PartitionRunner(object):
    def __init__(self, model_name=DEFAULT_MODEL_NAME, device=DEFAULT_DEVICE, max_input_length=DEFAULT_MAX_INPUT_LENGTH):
        self.runtime = get_transformers_runtime()
        self.model_name = model_name
        self.device = device
        self.max_input_length = int(max_input_length)
        if self.max_input_length <= 0:
            raise ValueError('max_input_length must be positive, got {}'.format(self.max_input_length))
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('device=cuda was requested, but torch.cuda.is_available() is False.')

        self.full_model = self.runtime.T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        self.full_model.eval()
        self.tokenizer = self.runtime.AutoTokenizer.from_pretrained(model_name)
        self._validate_model_layout()
        self.encoder_nodes = self._build_encoder_nodes()
        self.decoder_nodes = self._build_decoder_nodes()
        self.decoder_start_token_id = self._resolve_decoder_start_token_id()
        self.compressor_cache = {}

    def _validate_model_layout(self):
        encoder_layers = len(self.full_model.encoder.block)
        decoder_layers = len(self.full_model.decoder.block)
        if encoder_layers != 12 or decoder_layers != 12:
            raise ValueError(
                'This script assumes a 12+12 T5 layout for the fixed 3+1 split, but got '
                'encoder_layers={} decoder_layers={} for model {}'.format(
                    encoder_layers,
                    decoder_layers,
                    self.model_name,
                )
            )

    def _resolve_decoder_start_token_id(self):
        if getattr(self.full_model.config, 'decoder_start_token_id', None) is not None:
            return int(self.full_model.config.decoder_start_token_id)
        if self.tokenizer.pad_token_id is not None:
            return int(self.tokenizer.pad_token_id)
        raise RuntimeError('Could not resolve decoder_start_token_id for {}'.format(self.model_name))

    def _build_encoder_nodes(self):
        nodes = []
        for index, (start, end) in enumerate(ENCODER_RANGES):
            nodes.append(FlanT5EncoderPartition(
                full_model=self.full_model,
                layer_indices=list(range(start, end)),
                is_first_partition=(index == 0),
                is_last_partition=(index == len(ENCODER_RANGES) - 1),
                runtime=self.runtime,
            ))
        return nodes

    def _build_decoder_nodes(self):
        nodes = []
        for index, (start, end) in enumerate(DECODER_RANGES):
            nodes.append(FlanT5DecoderPartition(
                full_model=self.full_model,
                layer_indices=list(range(start, end)),
                is_first_partition=(index == 0),
                is_last_partition=(index == len(DECODER_RANGES) - 1),
                runtime=self.runtime,
            ))
        return nodes

    def clear_decoder_cache(self, task_id=None):
        for node in self.decoder_nodes:
            if task_id is None:
                node.clear_all_cache()
            else:
                node.clear_cache(task_id)

    def _get_compressor(self, compressor_name):
        name = str(compressor_name or '').strip().lower()
        if name in ('', 'identity', 'none'):
            return None
        if name not in self.compressor_cache:
            self.compressor_cache[name] = get_compressor(name)
        return self.compressor_cache[name]

    def encode_prompt(self, prompt_text):
        encoded = self.tokenizer(
            prompt_text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_input_length,
        )
        return {
            'input_ids': encoded.input_ids.detach().cpu(),
            'attention_mask': normalize_attention_mask(
                encoded.input_ids,
                encoded.get('attention_mask'),
                encoded.input_ids.device,
            ).detach().cpu(),
        }

    def run_partitioned_first_step(self, input_ids, attention_mask, task_id, compressor_name, tp_compression_params):
        compressor = self._get_compressor(compressor_name)
        input_ids = input_ids.to(self.device)
        attention_mask = normalize_attention_mask(input_ids, attention_mask, self.device)
        self.clear_decoder_cache(task_id)

        decoder_input_ids = torch.tensor(
            [[self.decoder_start_token_id]],
            dtype=torch.long,
            device=self.device,
        )
        hidden_states = input_ids
        position_bias = None
        transfer_stats = {}

        try:
            with torch.no_grad():
                for node_index, node in enumerate(self.encoder_nodes):
                    hidden_states, position_bias = node(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        position_bias=position_bias,
                    )
                    if node_index < len(self.encoder_nodes) - 1:
                        tp_name = 'tp{}'.format(node_index)
                        hidden_states, transfer_stats[tp_name] = apply_transport_compression(
                            hidden_states,
                            compressor=compressor,
                            compression_param=tp_compression_params[node_index],
                            device=self.device,
                        )

                encoder_hidden_states = hidden_states
                encoder_hidden_states, transfer_stats['tp2'] = apply_transport_compression(
                    encoder_hidden_states,
                    compressor=compressor,
                    compression_param=tp_compression_params[2],
                    device=self.device,
                )

                hidden_states = decoder_input_ids
                position_bias = None
                encoder_decoder_position_bias = None

                for node_index, node in enumerate(self.decoder_nodes):
                    hidden_states, position_bias, encoder_decoder_position_bias = node(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        task_id=task_id,
                        encoder_attention_mask=attention_mask,
                        position_bias=position_bias,
                        encoder_decoder_position_bias=encoder_decoder_position_bias,
                    )

            return {
                'logits': hidden_states[:, -1, :],
                'transfer_stats': transfer_stats,
            }
        finally:
            self.clear_decoder_cache(task_id)

def _safe_ratio(numerator, denominator, default=1.0):
    denominator = float(denominator)
    if denominator <= 0.0:
        return float(default)
    return float(numerator) / denominator

def _describe_candidate(candidate):
    if candidate['compressor'] == 'topk':
        return 'topk k={}'.format(_format_k_values(candidate['compression_params']))
    if candidate['compressor'] == 'quantization':
        return 'quantization k={} prec={}'.format(
            _format_k_values(candidate['compression_params']),
            ','.join(str(item) for item in candidate.get('quant_precision_values', [])),
        )
    if candidate['compressor'] == 'llmint8':
        return 'llmint8 policy={} outlier={}'.format(
            candidate['llm_policy'],
            _format_k_values(candidate['outlier_values']),
        )
    return str(candidate)

def generate_k_candidates(k_levels=K_LEVELS):
    return [list(item) for item in itertools.product(k_levels, repeat=NUM_TRANSFER_POINTS)]

def generate_llmint8_candidates(outlier_levels, policy_names):
    candidates = []
    for policy_name in policy_names:
        policy_spec = LLMINT8_POLICY_SPECS[policy_name]
        for outlier_values in itertools.product(outlier_levels, repeat=NUM_TRANSFER_POINTS):
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

def build_sweep_candidates(compressor_names, k_levels, llm_policies, llm_outlier_levels, quant_k_levels):
    candidates = []
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
    if 'llmint8' in compressor_names:
        candidates.extend(generate_llmint8_candidates(llm_outlier_levels, llm_policies))
    return candidates

def _expected_compressor_policies(compressor_names, llm_policies):
    policies = []
    if 'topk' in compressor_names:
        policies.append('topk')
    if 'quantization' in compressor_names:
        policies.append('quantization')
    if 'llmint8' in compressor_names:
        for policy_name in llm_policies:
            policies.append('llmint8_{}'.format(policy_name))
    return policies

def _policy_raw_csv_path(base_csv_path, compressor_policy):
    directory = os.path.dirname(base_csv_path)
    stem, ext = os.path.splitext(os.path.basename(base_csv_path))
    filename = '{}_{}{}'.format(stem, _sanitize_tag(compressor_policy), ext or '.csv')
    return os.path.join(directory, filename)

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

def _encode_samples(runner, samples, prompt_template):
    encoded_samples = []
    for sample in samples:
        prompt_text = build_prompt(sample.sentence, prompt_template)
        encoded = runner.encode_prompt(prompt_text)
        encoded_samples.append({
            'sample': sample,
            'prompt_text': prompt_text,
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
        })
    return encoded_samples

def _extract_two_way_logits(logits, verbalizers):
    return torch.stack([
        logits[:, verbalizers['negative_id']],
        logits[:, verbalizers['positive_id']],
    ], dim=1)

def _collect_candidate_rows(runner, encoded_samples, candidates, verbalizers):
    rows = []
    candidate_progress = tqdm(candidates, desc='config-sweep', unit='policy')

    for candidate_idx, candidate in enumerate(candidate_progress):
        total_correct = 0
        total_samples = 0
        total_latency = 0.0
        total_ce_loss = 0.0
        total_margin = 0.0
        tp_original_bytes = [0.0] * NUM_TRANSFER_POINTS
        tp_compressed_bytes = [0.0] * NUM_TRANSFER_POINTS

        sample_progress = tqdm(
            encoded_samples,
            desc=_sanitize_tag(candidate['compressor_policy']),
            unit='sample',
            leave=False,
        )
        for sample_item in sample_progress:
            sample = sample_item['sample']
            t_start = time.perf_counter()
            result = runner.run_partitioned_first_step(
                input_ids=sample_item['input_ids'],
                attention_mask=sample_item['attention_mask'],
                task_id='cand{}_sample{}'.format(candidate_idx, sample.sample_id),
                compressor_name=candidate['compressor'],
                tp_compression_params=candidate['compression_params'],
            )
            latency = time.perf_counter() - t_start
            logits = result['logits']
            pred_label = decide_label(logits[0], verbalizers)
            total_correct += int(pred_label == int(sample.label))
            total_samples += 1
            total_latency += latency

            label_logits = _extract_two_way_logits(logits, verbalizers)
            labels = torch.tensor([int(sample.label)], dtype=torch.long, device=label_logits.device)
            total_ce_loss += float(torch.nn.functional.cross_entropy(label_logits, labels, reduction='mean').item())
            positive_logit = float(logits[0, verbalizers['positive_id']].item())
            negative_logit = float(logits[0, verbalizers['negative_id']].item())
            total_margin += abs(positive_logit - negative_logit)

            for tp_idx in range(NUM_TRANSFER_POINTS):
                stats = result['transfer_stats']['tp{}'.format(tp_idx)]
                tp_original_bytes[tp_idx] += float(stats['original_bytes'])
                tp_compressed_bytes[tp_idx] += float(stats['compressed_bytes'])

            sample_progress.set_postfix(
                acc='{:.4f}'.format(_safe_ratio(total_correct, total_samples, default=0.0)),
                tp0='{:.3f}'.format(_safe_ratio(tp_compressed_bytes[0], tp_original_bytes[0])),
                tp1='{:.3f}'.format(_safe_ratio(tp_compressed_bytes[1], tp_original_bytes[1])),
                tp2='{:.3f}'.format(_safe_ratio(tp_compressed_bytes[2], tp_original_bytes[2])),
            )
        sample_progress.close()

        avg_accuracy = _safe_ratio(total_correct, total_samples, default=0.0)
        avg_latency = _safe_ratio(total_latency, total_samples, default=0.0)
        avg_ce_loss = _safe_ratio(total_ce_loss, total_samples, default=0.0)
        avg_margin = _safe_ratio(total_margin, total_samples, default=0.0)
        avg_payload_ratio = _safe_ratio(sum(tp_compressed_bytes), sum(tp_original_bytes), default=1.0)
        avg_compression_ratio = _safe_ratio(sum(tp_original_bytes), sum(tp_compressed_bytes), default=1.0)

        if candidate['compressor'] in ('topk', 'quantization'):
            realized_k_values = [_round4(_cap_k(item)) for item in candidate['feature_k_values']]
        else:
            realized_k_values = [
                _round4(_cap_k(_safe_ratio(tp_compressed_bytes[idx], tp_original_bytes[idx], default=1.0)))
                for idx in range(NUM_TRANSFER_POINTS)
            ]

        outlier_values = candidate.get('outlier_values', [None, None, None])
        quant_precision_values = candidate.get('quant_precision_values', ['', '', ''])
        outlier_values_text = (
            _format_k_values(outlier_values)
            if all(value is not None for value in outlier_values) else ''
        )

        row = {
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
            'num_samples': total_samples,
            'correct_count': total_correct,
            'avg_accuracy': avg_accuracy,
            'avg_latency': avg_latency,
            'avg_ce_loss': avg_ce_loss,
            'avg_margin': avg_margin,
            'avg_payload_ratio': avg_payload_ratio,
            'avg_compression_ratio': avg_compression_ratio,
            'tp0_original_bytes': tp_original_bytes[0],
            'tp1_original_bytes': tp_original_bytes[1],
            'tp2_original_bytes': tp_original_bytes[2],
            'tp0_compressed_bytes': tp_compressed_bytes[0],
            'tp1_compressed_bytes': tp_compressed_bytes[1],
            'tp2_compressed_bytes': tp_compressed_bytes[2],
            'tp0_payload_ratio': _round4(_cap_k(_safe_ratio(tp_compressed_bytes[0], tp_original_bytes[0], default=1.0))),
            'tp1_payload_ratio': _round4(_cap_k(_safe_ratio(tp_compressed_bytes[1], tp_original_bytes[1], default=1.0))),
            'tp2_payload_ratio': _round4(_cap_k(_safe_ratio(tp_compressed_bytes[2], tp_original_bytes[2], default=1.0))),
        }
        rows.append(row)

        candidate_progress.set_postfix(
            policy=candidate['compressor_policy'],
            acc='{:.4f}'.format(avg_accuracy),
            tp0='{:.3f}'.format(row['tp0_payload_ratio']),
            tp1='{:.3f}'.format(row['tp1_payload_ratio']),
            tp2='{:.3f}'.format(row['tp2_payload_ratio']),
        )
        logging.info(
            'Completed candidate: %s | realized_k=%s | acc=%.6f | latency=%.6f | payload_ratio=%.4f',
            _describe_candidate(candidate),
            _format_k_values(realized_k_values),
            avg_accuracy,
            avg_latency,
            avg_payload_ratio,
        )

    candidate_progress.close()
    return rows

def collect_raw_accuracy_data(output_csv=RAW_ACCURACY_CSV,
                              model_name=DEFAULT_MODEL_NAME,
                              dataset_path=None,
                              split=DEFAULT_SPLIT,
                              device=DEFAULT_DEVICE,
                              max_samples=DEFAULT_MAX_SAMPLES,
                              compressor_name=DEFAULT_COMPRESSOR_NAME,
                              k_levels=None,
                              quant_k_levels=None,
                              llm_policies=None,
                              llm_outlier_levels=None,
                              prompt_template=DEFAULT_PROMPT_TEMPLATE,
                              positive_token=None,
                              negative_token=None,
                              target_policies=None):
    resolved_compressors = _resolve_compressor_names(compressor_name)
    resolved_policies = _resolve_llmint8_policies(llm_policies or DEFAULT_LLMINT8_POLICY_NAMES)
    resolved_outlier_levels = _parse_float_levels(llm_outlier_levels, DEFAULT_LLMINT8_OUTLIER_LEVELS)
    resolved_k_levels = _parse_float_levels(k_levels, K_LEVELS)
    resolved_quant_k_levels = _parse_float_levels(quant_k_levels, DEFAULT_QUANTIZATION_K_LEVELS)

    runner = FlanT5SST2PartitionRunner(model_name=model_name, device=device)
    samples = load_sst2_samples(dataset_path=dataset_path, split=split)
    if max_samples is not None and int(max_samples) > 0:
        samples = samples[: int(max_samples)]
    verbalizers = resolve_single_token_verbalizers(
        runner.tokenizer,
        positive_text=positive_token,
        negative_text=negative_token,
    )
    encoded_samples = _encode_samples(runner, samples, prompt_template)

    candidates = build_sweep_candidates(
        compressor_names=resolved_compressors,
        k_levels=resolved_k_levels,
        llm_policies=resolved_policies,
        llm_outlier_levels=resolved_outlier_levels,
        quant_k_levels=resolved_quant_k_levels,
    )
    if target_policies:
        target_policy_set = {item.strip() for item in str(target_policies).split(',') if item.strip()}
        candidates = [item for item in candidates if item['compressor_policy'] in target_policy_set]

    logging.info(
        'Offline sweep: %d candidates x %d samples | compressors=%s',
        len(candidates),
        len(samples),
        resolved_compressors,
    )
    rows = _collect_candidate_rows(
        runner=runner,
        encoded_samples=encoded_samples,
        candidates=candidates,
        verbalizers=verbalizers,
    )
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df['model_name'] = model_name
    df['split'] = split
    df['positive_token'] = verbalizers['positive_text']
    df['negative_token'] = verbalizers['negative_text']
    df['positive_token_id'] = verbalizers['positive_id']
    df['negative_token_id'] = verbalizers['negative_id']
    df['prompt_template'] = prompt_template

    df = df.sort_values(['compressor_policy', 'avg_accuracy', 'avg_latency'], ascending=[True, False, True]).reset_index(drop=True)
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    _prepare_accuracy_df_for_save(df).to_csv(output_csv, index=False)
    logging.info('Saved combined raw accuracy CSV to %s', output_csv)

    for compressor_policy, group_df in _ordered_policy_groups(
        df,
        required_policies=_expected_compressor_policies(resolved_compressors, resolved_policies),
    ):
        policy_csv_path = _policy_raw_csv_path(output_csv, compressor_policy)
        _prepare_accuracy_df_for_save(group_df).to_csv(policy_csv_path, index=False)
        logging.info('Saved raw accuracy CSV for %s to %s', compressor_policy, policy_csv_path)

    mapping_output_path = _save_llmint8_eta_mapping(df, output_csv)
    if mapping_output_path is not None:
        logging.info('Saved llmint8 eta mapping to %s', mapping_output_path)

    return df

def _print_collection_summary(df):
    if df is None or df.empty:
        print('No raw rows collected.')
        return

    print('=' * 80)
    print('Raw Collection Summary')
    print('=' * 80)
    for compressor_policy, group_df in df.groupby('compressor_policy', sort=False):
        best_row = group_df.sort_values(['avg_accuracy', 'avg_latency'], ascending=[False, True]).iloc[0]
        print(
            '{:<24s} rows={:<4d} best_acc={:.6f} best_k={} avg_payload={:.4f}'.format(
                str(compressor_policy),
                len(group_df),
                float(best_row['avg_accuracy']),
                str(best_row['k_values']),
                float(best_row['avg_payload_ratio']),
            )
        )

def _print_training_summary(summary_df):
    if summary_df is None or summary_df.empty:
        print('No estimators trained.')
        return

    print('=' * 80)
    print('Estimator Training Summary')
    print('=' * 80)
    for _, row in summary_df.iterrows():
        print(
            '{policy:<24s} {model:<18s} test_rmse={rmse:.6f} test_mae={mae:.6f} test_r2={r2:.6f}'.format(
                policy=str(row['compressor_policy']),
                model=str(row['model_type']),
                rmse=float(row.get('test_rmse', 0.0)),
                mae=float(row.get('test_mae', 0.0)),
                r2=float(row.get('test_r2', 0.0)),
            )
        )

def main():
    parser = argparse.ArgumentParser(
        description='Local Flan-T5 SST-2 3-cutpoint experiment and estimator training.'
    )
    parser.add_argument('--mode', choices=['collect', 'train', 'all'], default='all')
    parser.add_argument('--model_name', default=DEFAULT_MODEL_NAME)
    parser.add_argument('--device', default=DEFAULT_DEVICE)
    parser.add_argument('--dataset_path', default=None)
    parser.add_argument('--split', default=DEFAULT_SPLIT)
    parser.add_argument('--max_samples', type=int, default=DEFAULT_MAX_SAMPLES, help='0 or negative means use the full dataset.')
    parser.add_argument(
        '--compressor_name',
        default=DEFAULT_COMPRESSOR_NAME,
        help="Compressor sweep to run. Defaults to '{}'.".format(DEFAULT_COMPRESSOR_NAME),
    )
    parser.add_argument('--k_levels', default=','.join(str(item) for item in K_LEVELS))
    parser.add_argument('--quant_k_levels', default=','.join(str(item) for item in DEFAULT_QUANTIZATION_K_LEVELS))
    parser.add_argument('--llm_policies', default=DEFAULT_LLMINT8_POLICY_NAMES)
    parser.add_argument(
        '--llm_outlier_levels',
        default=','.join(str(item) for item in DEFAULT_LLMINT8_OUTLIER_LEVELS),
    )
    parser.add_argument('--target_policies', default=None)
    parser.add_argument('--prompt_template', default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument('--positive_token', default=None)
    parser.add_argument('--negative_token', default=None)
    parser.add_argument('--csv_path', default=RAW_ACCURACY_CSV)
    parser.add_argument('--output_dir', default=ESTIMATOR_MODEL_DIR)
    parser.add_argument('--model_types', default=DEFAULT_MODEL_TYPES)
    parser.add_argument('--log_level', default='INFO')
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format='[%(levelname)s] %(message)s',
    )

    resolved_compressors = _resolve_compressor_names(args.compressor_name)
    resolved_policies = _resolve_llmint8_policies(args.llm_policies)
    if args.target_policies:
        required_policies = [item.strip() for item in str(args.target_policies).split(',') if item.strip()]
    else:
        required_policies = _expected_compressor_policies(resolved_compressors, resolved_policies)

    collected_df = None
    summary_df = None

    if args.mode in ('collect', 'all'):
        collected_df = collect_raw_accuracy_data(
            output_csv=args.csv_path,
            model_name=args.model_name,
            dataset_path=args.dataset_path,
            split=args.split,
            device=args.device,
            max_samples=args.max_samples,
            compressor_name=args.compressor_name,
            k_levels=args.k_levels,
            quant_k_levels=args.quant_k_levels,
            llm_policies=args.llm_policies,
            llm_outlier_levels=args.llm_outlier_levels,
            prompt_template=args.prompt_template,
            positive_token=args.positive_token,
            negative_token=args.negative_token,
            target_policies=args.target_policies,
        )
        _print_collection_summary(collected_df)

    if args.mode in ('train', 'all'):
        _, summary_df = train_estimators_from_csv(
            csv_path=args.csv_path,
            output_dir=args.output_dir,
            model_types=args.model_types,
            required_policies=required_policies,
        )
        _print_training_summary(summary_df)

    return 0
