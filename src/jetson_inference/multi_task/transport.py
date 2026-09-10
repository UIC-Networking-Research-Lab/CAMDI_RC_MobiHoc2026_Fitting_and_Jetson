"""thread-safe sending and transport lifecycle helpers."""

from jetson_inference.common.http_transport_dynamic import (
    MessageType,
    create_transport_dynamic,
)
from jetson_inference.multi_task.config import (
    NODE_IPS,
    NODE_PORTS,
)
from typing import (
    Any,
    Dict,
    Optional,
)


class ThreadSafeTransportSender:
    def __init__(self, transport):
        self.transport = transport

    def send(self, peer_id, msg_type, payload, logical_task_id: Optional[int] = None, link_idx: Optional[int] = None, rate_bps: Optional[float] = None, handshake_meta: Optional[Dict[str, Any]] = None):
        if (
            logical_task_id is None
            or link_idx is None
            or rate_bps is None
            or not hasattr(self.transport, "send_dynamic")
        ):
            bytes_sent, send_sec = self.transport.send(peer_id, msg_type, payload, handshake_meta=handshake_meta)
            return bytes_sent, float(send_sec)
        bytes_sent, send_sec = self.transport.send_dynamic(
            peer_id,
            msg_type,
            payload,
            logical_task_id=int(logical_task_id),
            link_idx=int(link_idx),
            rate_bps=float(rate_bps),
            handshake_meta=handshake_meta,
        )
        return bytes_sent, float(send_sec)

def _send_shutdown(transport):
    for peer_id, msg_type in [("B", MessageType.TASK_INPUT), ("C", MessageType.ENCODER_OUTPUT), ("D", MessageType.DECODER_STEP)]:
        try:
            transport.send(peer_id, msg_type, {"cmd": "shutdown"})
        except Exception:
            continue

def _create_multitask_transport(
    node_id,
    transport_backend,
    tx_bucket_capacity_bytes,
):
    backend = str(transport_backend).strip().lower()
    if backend != "http_dynamic":
        raise ValueError(
            "jetson_multi_task_pipeline_async.py only supports transport_backend='http_dynamic', got '{}'".format(
                transport_backend
            )
        )
    return create_transport_dynamic(
        node_id,
        NODE_IPS,
        NODE_PORTS,
        tx_bucket_capacity_bytes=tx_bucket_capacity_bytes,
    )
