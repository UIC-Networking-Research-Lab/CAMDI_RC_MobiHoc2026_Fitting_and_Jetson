"""activation payloads, compression profiles and eta execution plans."""

from jetson_inference.common.compressors import get_compressor


def _identity_activation_payload(tensor_cpu):
    return {"mode": "identity", "tensor": tensor_cpu}

def build_activation_payload(tensor, compression_param, compressor_name, feature_k_value):
    tensor_cpu = tensor.detach().cpu()
    original_bytes = int(tensor_cpu.numel() * tensor_cpu.element_size())
    if compressor_name in ("identity", None) or compression_param is None:
        payload = _identity_activation_payload(tensor_cpu)
        compressed_bytes = original_bytes
    else:
        compressor = get_compressor(compressor_name)
        compressed = compressor.compress(tensor_cpu, compression_param)
        compressed_bytes = None
        if hasattr(compressor, "get_compressed_size"):
            try:
                compressed_bytes = int(compressor.get_compressed_size(compressed))
            except Exception:
                compressed_bytes = None
        if compressed_bytes is None:
            compressed_bytes = int(compressed.get("compressed_bytes", original_bytes))
        payload = {
            "mode": "compressed",
            "compressor_name": compressor_name,
            "payload": compressed,
        }
    ratio = float(compressed_bytes) / float(original_bytes) if original_bytes > 0 else 1.0
    return payload, {
        "original_bytes": original_bytes,
        "compressed_bytes": compressed_bytes,
        "compression_ratio": ratio,
        "k_value": float(feature_k_value),
    }

def restore_activation_payload(payload, device):
    mode = payload.get("mode", "compressed")
    if mode == "identity":
        restored = payload["tensor"]
    else:
        compressor = get_compressor(payload["compressor_name"])
        restored = compressor.decompress(payload["payload"])
    return restored.to(device)
