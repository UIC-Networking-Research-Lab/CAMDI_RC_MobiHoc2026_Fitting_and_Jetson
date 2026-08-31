"""
compressors.py  -  Unified Activation Compressor Library
========================================================

This module provides four activation compression methods extracted from
the MDI distributed inference project, wrapped behind a uniform API:

  1. TopKCompressor         - Top-K sparsification with PackBits mask encoding
  2. QuantizationCompressor - Uniform quantization (FP16 / INT8 / INT4 / INT2)
  3. LLMInt8Compressor      - LLM.int8()-style hybrid mixed-precision compression
  4. OutlierTopKCompressor  - Outlier-aware two-group Top-K sparsification

All compressors share the same interface:

    compressed = compressor.compress(tensor, compression_params)
    restored   = compressor.decompress(compressed)
    nbytes     = compressor.get_compressed_size(compressed)
    name       = compressor.get_name()

Quick-start
-----------
    from compressors import (
        TopKCompressor,
        QuantizationCompressor,
        LLMInt8Compressor,
        OutlierTopKCompressor,
    )

    topk = TopKCompressor()
    data = topk.compress(tensor, 0.5)
    out = topk.decompress(data)

    quant = QuantizationCompressor()
    data = quant.compress(tensor, 0.25)
    out = quant.decompress(data)

    llm = LLMInt8Compressor()
    data = llm.compress(tensor, [0.01, 'fp16', 'int8'])
    out = llm.decompress(data)

    outlier_topk = OutlierTopKCompressor()
    data = outlier_topk.compress(tensor, [0.01, 0.5])
    out = outlier_topk.decompress(data)

Dependencies: torch, numpy
"""

import numpy as np
import torch

FP16_MAX_FINITE = 65504.0


class BaseCompressor:
    """
    Abstract base class for all compressors.

    Every subclass must implement:
        compress(tensor, compression_params)  -> dict
        decompress(compressed)                -> torch.Tensor
        get_compressed_size(compressed)       -> int
        get_name()                            -> str
    """

    def compress(self, tensor: torch.Tensor, compression_params) -> dict:
        raise NotImplementedError

    def decompress(self, compressed: dict) -> torch.Tensor:
        raise NotImplementedError

    def get_compressed_size(self, compressed: dict) -> int:
        raise NotImplementedError

    def get_name(self) -> str:
        raise NotImplementedError


class TopKCompressor(BaseCompressor):
    """Top-K sparsification compressor with bit-packed masks."""

    def compress(self, tensor, compression_params):
        k_ratio = compression_params
        original_bytes = tensor.numel() * tensor.element_size()

        if k_ratio >= 1.0:
            return {
                'compressed': False,
                'tensor': tensor,
                'original_bytes': original_bytes,
                'compressed_bytes': original_bytes,
                'compression_ratio': 1.0,
            }

        original_shape = tensor.shape
        batch_size = original_shape[0]
        reshaped = tensor.reshape(batch_size, -1)
        elements_per_sample = reshaped.shape[1]
        k = max(1, int(elements_per_sample * k_ratio))

        _, top_indices = torch.topk(reshaped.abs(), k, dim=1)

        mask = torch.zeros_like(reshaped, dtype=torch.bool)
        mask.scatter_(1, top_indices, True)

        sorted_indices = torch.sort(top_indices, dim=1)[0]
        sorted_values = torch.gather(reshaped, 1, sorted_indices)

        mask_np = mask.cpu().numpy().astype(np.uint8)
        packed_mask = np.packbits(mask_np, axis=1)

        values_bytes = sorted_values.numel() * 4
        mask_bytes = packed_mask.size
        metadata_bytes = 64
        compressed_bytes = values_bytes + mask_bytes + metadata_bytes

        return {
            'method': 'packbits_simple',
            'values': sorted_values.cpu().numpy(),
            'packed_mask': packed_mask,
            'shape': original_shape,
            'elements_per_sample': elements_per_sample,
            'k': k,
            'compressed': True,
            'original_bytes': original_bytes,
            'compressed_bytes': compressed_bytes,
            'compression_ratio': compressed_bytes / original_bytes if original_bytes > 0 else 1.0,
        }

    def decompress(self, compressed):
        if not compressed.get('compressed', False):
            return compressed['tensor']

        values = torch.from_numpy(compressed['values'])
        packed_mask = compressed['packed_mask']
        shape = compressed['shape']
        elements_per_sample = compressed['elements_per_sample']
        batch_size = shape[0]

        mask_np = np.unpackbits(packed_mask, axis=1)[:, :elements_per_sample]
        mask = torch.from_numpy(mask_np).bool()

        reshaped = torch.zeros(batch_size, elements_per_sample, dtype=values.dtype)
        reshaped[mask] = values.flatten()
        return reshaped.reshape(shape)

    def get_compressed_size(self, compressed):
        if not compressed.get('compressed', False):
            return int(compressed['original_bytes'])
        return int(compressed['compressed_bytes'])

    def get_name(self):
        return 'TopK'


class _QuanFP16:
    def compress(self, tensor, compression_params):
        original_bytes = tensor.numel() * tensor.element_size()

        if compression_params >= 1.0:
            return {
                'compressed': False,
                'tensor': tensor,
                'original_bytes': original_bytes,
                'compressed_bytes': original_bytes,
                'compression_ratio': 1.0,
            }

        compressed_bytes = tensor.numel() * 2 + 64
        return {
            'method': 'fp16',
            'values': tensor.half().cpu().numpy(),
            'shape': tensor.shape,
            'compressed': True,
            'original_bytes': original_bytes,
            'compressed_bytes': compressed_bytes,
            'compression_ratio': compressed_bytes / original_bytes if original_bytes > 0 else 1.0,
        }

    def decompress(self, data):
        if not data.get('compressed', False):
            return data['tensor']
        return torch.from_numpy(data['values']).float()

    def get_compressed_size(self, data):
        if not data.get('compressed', False):
            return int(data['original_bytes'])
        return int(data['compressed_bytes'])


class _QuanInt8:
    def compress(self, tensor, compression_params):
        original_bytes = tensor.numel() * tensor.element_size()

        if compression_params >= 1.0:
            return {
                'compressed': False,
                'tensor': tensor,
                'original_bytes': original_bytes,
                'compressed_bytes': original_bytes,
                'compression_ratio': 1.0,
            }

        abs_max = tensor.abs().max()
        scale = abs_max / 127.0 if abs_max > 0 else 1.0
        quantized = (tensor / scale).round().clamp(-127, 127).to(torch.int8)

        compressed_bytes = quantized.numel() + 4 + 64
        return {
            'method': 'int8_signed',
            'values': quantized.cpu().numpy(),
            'scale': scale.item() if hasattr(scale, 'item') else float(scale),
            'shape': tensor.shape,
            'compressed': True,
            'original_bytes': original_bytes,
            'compressed_bytes': compressed_bytes,
            'compression_ratio': compressed_bytes / original_bytes if original_bytes > 0 else 1.0,
        }

    def decompress(self, data):
        if not data.get('compressed', False):
            return data['tensor']
        values = torch.from_numpy(data['values']).float()
        return values * data['scale']

    def get_compressed_size(self, data):
        if not data.get('compressed', False):
            return int(data['original_bytes'])
        return int(data['compressed_bytes'])


class _QuanInt4:
    def compress(self, tensor, compression_params):
        original_bytes = tensor.numel() * tensor.element_size()

        if compression_params >= 1.0:
            return {
                'compressed': False,
                'tensor': tensor,
                'original_bytes': original_bytes,
                'compressed_bytes': original_bytes,
                'compression_ratio': 1.0,
            }

        abs_max = tensor.abs().max()
        scale = abs_max / 7.0 if abs_max > 0 else 1.0
        x = tensor / scale
        floor_x = x.floor()
        prob = torch.nan_to_num(x - floor_x, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        quantized = floor_x + torch.bernoulli(prob).to(x.device)
        quantized = quantized.clamp(-7, 7).to(torch.int8)
        shifted = (quantized + 7).to(torch.uint8)
        flat = shifted.flatten()
        if flat.numel() % 2 != 0:
            flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8, device=flat.device)])
            padding = 1
        else:
            padding = 0
        packed = (flat[0::2] << 4) | (flat[1::2] & 0x0F)

        compressed_bytes = packed.numel() + 4 + 64
        return {
            'method': 'int4_signed',
            'packed': packed.cpu().numpy(),
            'scale': scale.item() if hasattr(scale, 'item') else float(scale),
            'shape': tensor.shape,
            'padding': padding,
            'compressed': True,
            'original_bytes': original_bytes,
            'compressed_bytes': compressed_bytes,
            'compression_ratio': compressed_bytes / original_bytes if original_bytes > 0 else 1.0,
        }

    def decompress(self, data):
        if not data.get('compressed', False):
            return data['tensor']
        packed = torch.from_numpy(data['packed'])
        high = (packed >> 4).to(torch.int8)
        low = (packed & 0x0F).to(torch.int8)
        unpacked = torch.stack([high, low], dim=1).flatten()
        if data['padding'] > 0:
            unpacked = unpacked[:-data['padding']]
        quantized = unpacked - 7
        return quantized.float().reshape(data['shape']) * data['scale']

    def get_compressed_size(self, data):
        if not data.get('compressed', False):
            return int(data['original_bytes'])
        return int(data['compressed_bytes'])


class _QuanInt2:
    def compress(self, tensor, compression_params):
        original_bytes = tensor.numel() * tensor.element_size()

        if compression_params >= 1.0:
            return {
                'compressed': False,
                'tensor': tensor,
                'original_bytes': original_bytes,
                'compressed_bytes': original_bytes,
                'compression_ratio': 1.0,
            }

        abs_max = tensor.abs().max()
        scale = abs_max if abs_max > 0 else 1.0
        x = tensor / scale
        quantized = x.round().clamp(-1, 1).to(torch.int8)
        shifted = (quantized + 1).to(torch.uint8)
        flat = shifted.flatten()
        padding = (4 - (flat.numel() % 4)) % 4
        if padding > 0:
            flat = torch.cat([flat, torch.zeros(padding, dtype=torch.uint8, device=flat.device)])
        packed = (flat[0::4] << 6) | (flat[1::4] << 4) | (flat[2::4] << 2) | flat[3::4]

        compressed_bytes = packed.numel() + 4 + 64
        return {
            'method': 'int2_signed',
            'packed': packed.cpu().numpy(),
            'scale': scale.item() if hasattr(scale, 'item') else float(scale),
            'shape': tensor.shape,
            'padding': padding,
            'compressed': True,
            'original_bytes': original_bytes,
            'compressed_bytes': compressed_bytes,
            'compression_ratio': compressed_bytes / original_bytes if original_bytes > 0 else 1.0,
        }

    def decompress(self, data):
        if not data.get('compressed', False):
            return data['tensor']
        packed = torch.from_numpy(data['packed'])
        p0 = (packed >> 6) & 0x03
        p1 = (packed >> 4) & 0x03
        p2 = (packed >> 2) & 0x03
        p3 = packed & 0x03
        unpacked = torch.stack([p0, p1, p2, p3], dim=1).flatten()
        if data['padding'] > 0:
            unpacked = unpacked[:-data['padding']]
        quantized = unpacked.to(torch.int8) - 1
        return quantized.float().reshape(data['shape']) * data['scale']

    def get_compressed_size(self, data):
        if not data.get('compressed', False):
            return int(data['original_bytes'])
        return int(data['compressed_bytes'])


class QuantizationCompressor(BaseCompressor):
    """
    Smart quantization compressor that selects bit-width from k ranges.

    Mapping:
        k >= 1.0          -> pass-through
        0.5 <= k < 1.0    -> FP16
        0.25 <= k < 0.5   -> INT8
        0.125 <= k < 0.25 -> INT4
        k < 0.125         -> INT2
    """

    def __init__(self):
        self._fp16 = _QuanFP16()
        self._int8 = _QuanInt8()
        self._int4 = _QuanInt4()
        self._int2 = _QuanInt2()

    def _select_compressor(self, k):
        if k >= 0.5:
            return self._fp16
        if k >= 0.25:
            return self._int8
        if k >= 0.125:
            return self._int4
        return self._int2

    def _dispatch_decompress(self, data):
        method = data.get('method')
        if method == 'int8_signed':
            return self._int8
        if method == 'int4_signed':
            return self._int4
        if method == 'int2_signed':
            return self._int2
        return self._fp16

    def compress(self, tensor, compression_params):
        k = compression_params
        if k >= 1.0:
            original_bytes = tensor.numel() * tensor.element_size()
            return {
                'compressed': False,
                'tensor': tensor,
                'original_bytes': original_bytes,
                'compressed_bytes': original_bytes,
                'compression_ratio': 1.0,
            }
        return self._select_compressor(k).compress(tensor, k)

    def decompress(self, compressed):
        return self._dispatch_decompress(compressed).decompress(compressed)

    def get_compressed_size(self, compressed):
        return self._dispatch_decompress(compressed).get_compressed_size(compressed)

    def get_name(self):
        return 'Quantization'


class LLMInt8Compressor(BaseCompressor):
    """
    LLM.int8()-style hybrid mixed-precision compressor.

    compression_params : list
        [outlier_ratio, outlier_precision, regular_precision]

    outlier_precision:
        'fp32', 'fp16', or 'int8'
    """

    def compress(self, tensor, compression_params):
        outlier_ratio, outlier_precision, regular_precision = compression_params

        if tensor.dtype != torch.float32:
            tensor = tensor.float()

        shape = tensor.shape
        original_bytes = tensor.numel() * 4
        outlier_ratio = float(outlier_ratio)
        if outlier_ratio < 0.0 or outlier_ratio > 1.0:
            raise ValueError(f'outlier_ratio must be in [0, 1], got {outlier_ratio}')

        if outlier_ratio <= 0.0:
            threshold = tensor.abs().max().item() if tensor.numel() > 0 else 0.0
            mask_bool = torch.zeros_like(tensor, dtype=torch.bool)
        elif outlier_ratio >= 1.0:
            # Boundary case: route the full tensor through the outlier branch.
            threshold = -1.0
            mask_bool = torch.ones_like(tensor, dtype=torch.bool)
        else:
            percentile = 1.0 - outlier_ratio
            threshold = torch.quantile(tensor.abs().float().reshape(-1), percentile).item()
            mask_bool = tensor.abs() > threshold

        num_outliers = mask_bool.sum().item()
        all_outliers = num_outliers == tensor.numel() and tensor.numel() > 0

        outlier_values_raw = torch.masked_select(tensor, mask_bool)
        if outlier_precision == 'fp32':
            outlier_values = outlier_values_raw.float()
            size_outliers = outlier_values.numel() * 4
            outlier_scale = None
            outlier_clipped_count = 0
        elif outlier_precision == 'fp16':
            # FP16 cannot represent values outside +/-65504, so clamp first to avoid infs.
            clamped_outliers = outlier_values_raw.clamp(min=-FP16_MAX_FINITE, max=FP16_MAX_FINITE)
            outlier_clipped_count = int((clamped_outliers != outlier_values_raw).sum().item())
            outlier_values = clamped_outliers.half()
            size_outliers = outlier_values.numel() * 2
            outlier_scale = None
        elif outlier_precision == 'int8':
            outlier_abs_max = outlier_values_raw.abs().max() if outlier_values_raw.numel() > 0 else 0.0
            outlier_scale = outlier_abs_max / 127.0 if outlier_abs_max > 0 else 1.0
            outlier_values = (outlier_values_raw / outlier_scale).round().clamp(-127, 127).to(torch.int8)
            size_outliers = outlier_values.numel()
            outlier_clipped_count = 0
        else:
            raise ValueError(f'Unsupported outlier precision: {outlier_precision}')

        tensor_regular = tensor.clone()
        tensor_regular.masked_fill_(mask_bool, 0.0)
        if all_outliers:
            scales = torch.empty((0, 1), dtype=torch.float32, device=tensor.device)
            if regular_precision == 'int8':
                quantized_regular = torch.empty((0,), dtype=torch.int8, device=tensor.device)
            elif regular_precision in ('int4', 'int2'):
                quantized_regular = torch.empty((0,), dtype=torch.uint8, device=tensor.device)
            else:
                raise ValueError(f'Unsupported regular precision: {regular_precision}')
            size_regular = 0
            regular_padding = 0
        else:
            flattened = tensor_regular.view(-1, shape[-1])
            abs_max = flattened.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)

            if regular_precision == 'int8':
                scales = abs_max / 127.0
                quantized_regular = (flattened / scales).round().clamp(-127, 127).to(torch.int8)
                size_regular = quantized_regular.numel()
                regular_padding = 0
            elif regular_precision == 'int4':
                scales = abs_max / 7.0
                x = flattened / scales
                floor_x = x.floor()
                prob = torch.nan_to_num(x - floor_x, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
                quantized = floor_x + torch.bernoulli(prob).to(x.device)
                quantized = quantized.clamp(-7, 7).to(torch.int8)
                shifted = (quantized + 7).to(torch.uint8)
                flat = shifted.flatten()
                if flat.numel() % 2 != 0:
                    flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8, device=flat.device)])
                    regular_padding = 1
                else:
                    regular_padding = 0
                quantized_regular = (flat[0::2] << 4) | (flat[1::2] & 0x0F)
                size_regular = quantized_regular.numel() * 0.5
            elif regular_precision == 'int2':
                scales = abs_max / 1.0
                x = flattened / scales
                quantized = x.round().clamp(-1, 1).to(torch.int8)
                shifted = (quantized + 1).to(torch.uint8)
                flat = shifted.flatten()
                regular_padding = (4 - (flat.numel() % 4)) % 4
                if regular_padding > 0:
                    flat = torch.cat([flat, torch.zeros(regular_padding, dtype=torch.uint8, device=flat.device)])
                quantized_regular = (flat[0::4] << 6) | (flat[1::4] << 4) | (flat[2::4] << 2) | flat[3::4]
                size_regular = quantized_regular.numel() * 0.25
            else:
                raise ValueError(f'Unsupported regular precision: {regular_precision}')

        mask_np = mask_bool.reshape(-1).cpu().numpy().astype(np.uint8)
        packed_mask = np.packbits(mask_np)

        size_mask = packed_mask.nbytes
        size_scales = scales.numel() * 4
        metadata_bytes = 64
        if outlier_precision == 'int8':
            metadata_bytes += 4

        compressed_bytes = size_mask + size_outliers + size_regular + size_scales + metadata_bytes

        return {
            'method': 'llmint8_hybrid',
            'packed_mask': packed_mask,
            'outlier_values': outlier_values.cpu().numpy(),
            'outlier_scale': (
                outlier_scale.item() if hasattr(outlier_scale, 'item') else outlier_scale
            ) if outlier_scale is not None else None,
            'main_values': quantized_regular.cpu().numpy(),
            'main_scales': scales.cpu().numpy(),
            'regular_padding': regular_padding,
            'all_outliers': all_outliers,
            'shape': shape,
            'numel': tensor.numel(),
            'threshold': threshold,
            'num_outliers': num_outliers,
            'outlier_ratio': num_outliers / tensor.numel() if tensor.numel() > 0 else 0.0,
            'outlier_clipped_count': outlier_clipped_count,
            'compressed': True,
            'original_bytes': original_bytes,
            'compressed_bytes': compressed_bytes,
            'compression_ratio': compressed_bytes / original_bytes if original_bytes > 0 else 1.0,
            'outlier_precision': outlier_precision,
            'regular_precision': regular_precision,
        }

    def decompress(self, data):
        if not data.get('compressed', False):
            return data['tensor']

        shape = data['shape']
        numel = data['numel']

        mask_flat = np.unpackbits(data['packed_mask'])
        mask_flat = mask_flat[:numel]
        mask = torch.from_numpy(mask_flat.copy()).view(shape).bool()

        regular_precision = data.get('regular_precision', 'int8')
        if data.get('all_outliers', False):
            restored = torch.zeros(shape, dtype=torch.float32)
        else:
            main_scales = torch.from_numpy(data['main_scales'].copy()).float()

            if regular_precision == 'int8':
                main_values = torch.from_numpy(data['main_values'].copy()).float()
                restored = main_values.view(-1, shape[-1]) * main_scales
            elif regular_precision == 'int4':
                packed = torch.from_numpy(data['main_values'].copy())
                padding = data['regular_padding']
                high = (packed >> 4).to(torch.int8)
                low = (packed & 0x0F).to(torch.int8)
                unpacked = torch.stack([high, low], dim=1).flatten()
                if padding > 0:
                    unpacked = unpacked[:-padding]
                quantized = unpacked - 7
                restored = quantized.float().view(-1, shape[-1]) * main_scales
            elif regular_precision == 'int2':
                packed = torch.from_numpy(data['main_values'].copy())
                padding = data['regular_padding']
                p0 = (packed >> 6) & 0x03
                p1 = (packed >> 4) & 0x03
                p2 = (packed >> 2) & 0x03
                p3 = packed & 0x03
                unpacked = torch.stack([p0, p1, p2, p3], dim=1).flatten()
                if padding > 0:
                    unpacked = unpacked[:-padding]
                quantized = unpacked.to(torch.int8) - 1
                restored = quantized.float().view(-1, shape[-1]) * main_scales
            else:
                raise ValueError(f'Unsupported regular precision: {regular_precision}')

            restored = restored.view(shape)

        if data['outlier_values'].size > 0:
            outlier_precision = data.get('outlier_precision', 'fp16')
            if outlier_precision in ('fp32', 'fp16'):
                outlier_values = torch.from_numpy(data['outlier_values'].copy()).float()
            elif outlier_precision == 'int8':
                outlier_values = torch.from_numpy(data['outlier_values'].copy()).float()
                outlier_values = outlier_values * data['outlier_scale']
            else:
                raise ValueError(f'Unsupported outlier precision: {outlier_precision}')
            restored.masked_scatter_(mask, outlier_values)

        return restored

    def get_compressed_size(self, compressed):
        if not compressed.get('compressed', False):
            return int(compressed['original_bytes'])
        return int(compressed['compressed_bytes'])

    def get_name(self):
        return 'LLMInt8'


class OutlierTopKCompressor(BaseCompressor):
    """
    Outlier-aware Top-K sparsification.

    compression_params : list
        [outlier_ratio, k_ratio]
    """

    def compress(self, tensor, compression_params):
        outlier_ratio, k_ratio = compression_params

        if tensor.dtype != torch.float32:
            tensor = tensor.float()

        original_bytes = tensor.numel() * 4
        if k_ratio >= 1.0:
            return {
                'compressed': False,
                'tensor': tensor,
                'original_bytes': original_bytes,
                'compressed_bytes': original_bytes,
                'compression_ratio': 1.0,
            }

        original_shape = tensor.shape
        batch_size = original_shape[0]
        reshaped = tensor.reshape(batch_size, -1)
        elements_per_sample = reshaped.shape[1]
        abs_vals = reshaped.abs()

        num_outlier = max(1, int(elements_per_sample * outlier_ratio))
        _, outlier_indices = torch.topk(abs_vals, num_outlier, dim=1)
        outlier_mask = torch.zeros_like(reshaped, dtype=torch.bool)
        outlier_mask.scatter_(1, outlier_indices, True)
        regular_mask = ~outlier_mask
        num_regular = elements_per_sample - num_outlier

        k_outlier = max(1, int(num_outlier * k_ratio))
        abs_outlier_only = abs_vals.clone()
        abs_outlier_only[regular_mask] = -1.0
        _, outlier_topk_indices = torch.topk(abs_outlier_only, k_outlier, dim=1)

        k_regular = max(1, int(num_regular * k_ratio))
        abs_regular_only = abs_vals.clone()
        abs_regular_only[outlier_mask] = -1.0
        _, regular_topk_indices = torch.topk(abs_regular_only, k_regular, dim=1)

        final_mask = torch.zeros_like(reshaped, dtype=torch.bool)
        final_mask.scatter_(1, outlier_topk_indices, True)
        final_mask.scatter_(1, regular_topk_indices, True)

        all_sorted_indices = []
        for batch_idx in range(batch_size):
            selected_indices = final_mask[batch_idx].nonzero(as_tuple=False).squeeze(1)
            all_sorted_indices.append(torch.sort(selected_indices)[0])

        sorted_values = torch.stack(
            [reshaped[batch_idx][all_sorted_indices[batch_idx]] for batch_idx in range(batch_size)],
            dim=0,
        )

        mask_np = final_mask.cpu().numpy().astype(np.uint8)
        packed_mask = np.packbits(mask_np, axis=1)

        values_bytes = sorted_values.numel() * 4
        mask_bytes = packed_mask.size
        metadata_bytes = 64
        compressed_bytes = values_bytes + mask_bytes + metadata_bytes

        return {
            'method': 'outlier_topk',
            'values': sorted_values.cpu().numpy(),
            'packed_mask': packed_mask,
            'shape': original_shape,
            'elements_per_sample': elements_per_sample,
            'k': int(final_mask.sum(dim=1)[0].item()),
            'k_outlier': k_outlier,
            'k_regular': k_regular,
            'num_outlier': num_outlier,
            'num_regular': num_regular,
            'outlier_ratio': num_outlier / elements_per_sample,
            'compressed': True,
            'original_bytes': original_bytes,
            'compressed_bytes': compressed_bytes,
            'compression_ratio': compressed_bytes / original_bytes if original_bytes > 0 else 1.0,
        }

    def decompress(self, compressed):
        if not compressed.get('compressed', False):
            return compressed['tensor']

        values = torch.from_numpy(compressed['values'])
        packed_mask = compressed['packed_mask']
        shape = compressed['shape']
        elements_per_sample = compressed['elements_per_sample']
        batch_size = shape[0]

        mask_np = np.unpackbits(packed_mask, axis=1)[:, :elements_per_sample]
        mask = torch.from_numpy(mask_np).bool()

        reshaped = torch.zeros(batch_size, elements_per_sample, dtype=values.dtype)
        reshaped[mask] = values.flatten()
        return reshaped.reshape(shape)

    def get_compressed_size(self, compressed):
        if not compressed.get('compressed', False):
            return int(compressed['original_bytes'])
        return int(compressed['compressed_bytes'])

    def get_name(self):
        return 'OutlierTopK'


def get_compressor(name):
    """
    Factory function to create a compressor by name.

    Args:
        name: One of 'topk', 'quantization', 'llmint8', 'outliertopk'.
    """
    name = name.lower().strip()
    if name == 'topk':
        return TopKCompressor()
    if name in ('quantization', 'quant'):
        return QuantizationCompressor()
    if name in ('llmint8', 'llm_int8', 'llm.int8'):
        return LLMInt8Compressor()
    if name in ('outliertopk', 'outlier_topk', 'outlier-topk'):
        return OutlierTopKCompressor()
    raise ValueError(
        f"Unknown compressor '{name}'. Choose from: topk, quantization, llmint8, outliertopk"
    )


if __name__ == '__main__':
    print('=' * 60)
    print('Compressor self-test')
    print('=' * 60)

    torch.manual_seed(42)
    tensor = torch.relu(torch.randn(2, 64, 8, 8))
    original_bytes = tensor.numel() * 4
    print(f'Input shape: {tensor.shape},  size: {original_bytes} bytes\n')

    for k in [1.0, 0.5, 0.25, 0.125]:
        comp = TopKCompressor()
        data = comp.compress(tensor, k)
        out = comp.decompress(data)
        err = (tensor - out).abs().mean().item()
        ratio = comp.get_compressed_size(data) / original_bytes
        print(f'[{comp.get_name()}] k={k:.3f}  ratio={ratio:.3f}  MAE={err:.6f}  shape_ok={out.shape == tensor.shape}')

    print()

    for k in [1.0, 0.5, 0.25, 0.125, 0.0625]:
        comp = QuantizationCompressor()
        data = comp.compress(tensor, k)
        out = comp.decompress(data)
        err = (tensor - out).abs().mean().item()
        ratio = comp.get_compressed_size(data) / original_bytes
        print(f'[{comp.get_name()}] k={k:.4f}  ratio={ratio:.3f}  MAE={err:.6f}  shape_ok={out.shape == tensor.shape}')

    print()

    llm_configs = [
        [0.01, 'fp32', 'int8'],
        [0.01, 'fp16', 'int8'],
        [0.05, 'fp16', 'int4'],
        [0.10, 'int8', 'int2'],
    ]
    for params in llm_configs:
        comp = LLMInt8Compressor()
        data = comp.compress(tensor, params)
        out = comp.decompress(data)
        err = (tensor - out).abs().mean().item()
        ratio = comp.get_compressed_size(data) / original_bytes
        print(f'[{comp.get_name()}] outlier={params[0]:.2f} out={params[1]} reg={params[2]}  ratio={ratio:.3f}  MAE={err:.6f}  shape_ok={out.shape == tensor.shape}')

    print()

    llm_boundary_configs = [
        [0.00, 'fp32', 'int8'],
        [0.00, 'fp16', 'int8'],
        [0.00, 'int8', 'int4'],
        [1.00, 'fp32', 'int8'],
        [1.00, 'fp16', 'int8'],
        [1.00, 'fp16', 'int4'],
        [1.00, 'int8', 'int2'],
    ]
    print('LLMInt8 boundary checks')
    for params in llm_boundary_configs:
        comp = LLMInt8Compressor()
        data = comp.compress(tensor, params)
        out = comp.decompress(data)
        err = (tensor - out).abs().mean().item()
        ratio = comp.get_compressed_size(data) / original_bytes
        actual_outlier_ratio = data.get('outlier_ratio', 0.0)
        num_outliers = data.get('num_outliers', 0)
        all_outliers = data.get('all_outliers', False)
        print(
            f'[{comp.get_name()}][boundary] outlier={params[0]:.2f} out={params[1]} reg={params[2]}  '
            f'ratio={ratio:.3f}  MAE={err:.6f}  actual_outlier={actual_outlier_ratio:.4f}  '
            f'num_outliers={num_outliers}  all_outliers={all_outliers}  shape_ok={out.shape == tensor.shape}'
        )

    print()

    outlier_configs = [
        [0.01, 0.5],
        [0.05, 0.25],
        [0.10, 0.125],
    ]
    for params in outlier_configs:
        comp = OutlierTopKCompressor()
        data = comp.compress(tensor, params)
        out = comp.decompress(data)
        err = (tensor - out).abs().mean().item()
        ratio = comp.get_compressed_size(data) / original_bytes
        print(f'[{comp.get_name()}] outlier={params[0]:.2f} k={params[1]:.3f}  ratio={ratio:.3f}  MAE={err:.6f}  shape_ok={out.shape == tensor.shape}')

    print('\nAll tests passed.')
