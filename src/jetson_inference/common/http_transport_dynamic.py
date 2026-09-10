import logging
import pickle
import struct
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass

import requests
from flask import Flask, Response, request
from werkzeug.serving import make_server

from jetson_inference.common.http_transport import (
    DEFAULT_TX_BUCKET_CAPACITY_BYTES,
    FRAME_HEADER,
    FRAME_VERSION,
    MessageType,
)


class _ServerThread(threading.Thread):
    def __init__(self, app, host, port):
        super().__init__(daemon=True)
        self._server = make_server(host, port, app, threaded=True)

    def run(self):
        self._server.serve_forever()

    def shutdown(self):
        self._server.shutdown()


class DynamicTokenBucket:
    """Token bucket with runtime-updatable rate."""

    def __init__(self, capacity_bytes=None):
        if capacity_bytes is None:
            capacity_bytes = DEFAULT_TX_BUCKET_CAPACITY_BYTES
        self.capacity_bytes = float(capacity_bytes)
        if self.capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        self.rate_bytes_per_sec = 1e-6
        self._tokens = self.capacity_bytes
        self._last_refill = time.perf_counter()
        self._lock = threading.Lock()

    def _refill_locked(self, now):
        elapsed = max(0.0, now - self._last_refill)
        self._last_refill = now
        self._tokens = min(
            self.capacity_bytes,
            self._tokens + elapsed * max(float(self.rate_bytes_per_sec), 1e-6),
        )

    def update_rate(self, rate_bytes_per_sec):
        with self._lock:
            now = time.perf_counter()
            self._refill_locked(now)
            self.rate_bytes_per_sec = max(float(rate_bytes_per_sec), 1e-6)

    def acquire(self, num_bytes):
        amount = float(num_bytes)
        if amount <= 0:
            return 0.0

        with self._lock:
            now = time.perf_counter()
            self._refill_locked(now)
            deficit = max(0.0, amount - self._tokens)
            self._tokens -= amount
            rate = max(float(self.rate_bytes_per_sec), 1e-6)

        wait_sec = deficit / rate if deficit > 0 else 0.0
        if wait_sec > 0:
            time.sleep(wait_sec)
        return wait_sec


@dataclass
class _ChannelState:
    lock: threading.Lock
    bucket: DynamicTokenBucket


class HttpTransportDynamic:
    def __init__(
        self,
        node_id,
        node_ips,
        node_ports,
        request_timeout=30.0,
        tx_bucket_capacity_bytes=None,
    ):
        self.node_id = node_id
        self.node_ips = dict(node_ips)
        self.node_ports = dict(node_ports)
        self.request_timeout = request_timeout
        self._session = requests.Session()
        self._thread_local = threading.local()
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._inbox = deque()
        self._pending_handshakes = {}
        self._server_thread = None
        self._app = Flask(f"http_transport_dynamic_{node_id}")
        self._tx_bucket_capacity_bytes = (
            DEFAULT_TX_BUCKET_CAPACITY_BYTES
            if tx_bucket_capacity_bytes is None
            else int(tx_bucket_capacity_bytes)
        )
        self._channel_lock = threading.Lock()
        self._channel_states = {}
        self._configure_routes()

    def _configure_routes(self):
        @self._app.get("/health")
        def _health():
            return {"status": "ok", "node": self.node_id}

        @self._app.post("/hs")
        def _handshake():
            payload = request.get_json(force=True, silent=False) or {}
            handshake_id = str(payload.get("handshake_id", "")).strip()
            if not handshake_id:
                return Response(status=400)
            now = time.perf_counter()
            handshake_meta = payload.get("handshake_meta")
            if handshake_meta is not None and not isinstance(handshake_meta, dict):
                handshake_meta = None
            with self._lock:
                self._purge_stale_handshakes_locked(now)
                self._pending_handshakes[handshake_id] = {
                    "started_at": now,
                    "src": str(payload.get("src", "")),
                    "dst": str(payload.get("dst", "")),
                    "msg_type": int(payload.get("msg_type", 0)),
                    "handshake_meta": dict(handshake_meta or {}),
                }
            return {"status": "ok", "node": self.node_id, "handshake_id": handshake_id}

        @self._app.post("/m")
        def _message():
            handshake_id = request.headers.get("X-Handshake-ID", "").strip()
            raw = request.get_data(cache=False)
            receive_completed_at = time.perf_counter()
            if len(raw) < FRAME_HEADER.size:
                return Response(status=400)

            version, msg_type, src_code, dst_code, payload_len = FRAME_HEADER.unpack(
                raw[: FRAME_HEADER.size]
            )
            if version != FRAME_VERSION:
                return Response(status=400)

            payload_bytes = raw[FRAME_HEADER.size :]
            if payload_len != len(payload_bytes):
                return Response(status=400)

            transfer_started_at = None
            handshake_entry = None
            if handshake_id:
                with self._lock:
                    handshake_entry = self._pending_handshakes.pop(handshake_id, None)
                if handshake_entry is not None:
                    transfer_started_at = float(handshake_entry["started_at"])
                else:
                    logging.warning(
                        "[%s] Missing handshake metadata for handshake_id=%s; falling back to receive completion timing",
                        self.node_id,
                        handshake_id,
                    )
            if transfer_started_at is None:
                transfer_started_at = receive_completed_at
            transfer_elapsed_sec = max(receive_completed_at - transfer_started_at, 1e-9)

            envelope = {
                "msg_type": MessageType(msg_type),
                "src": chr(src_code),
                "dst": chr(dst_code),
                "payload": pickle.loads(payload_bytes),
                "received_at": receive_completed_at,
                "frame_bytes": len(raw),
                "payload_bytes": len(payload_bytes),
                "transfer_started_at": transfer_started_at,
                "transfer_completed_at": receive_completed_at,
                "transfer_elapsed_sec": transfer_elapsed_sec,
                "handshake_id": handshake_id or None,
                "handshake_started_at": (
                    float(handshake_entry["started_at"])
                    if handshake_id and handshake_entry is not None and "started_at" in handshake_entry
                    else None
                ),
                "handshake_meta": (
                    dict(handshake_entry.get("handshake_meta", {}))
                    if handshake_id and handshake_entry is not None
                    else {}
                ),
            }

            with self._cv:
                self._inbox.append(envelope)
                self._cv.notify_all()
            return Response(status=200)

    def _get_send_session(self):
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            self._thread_local.session = session
        return session

    def _peer_url(self, peer_id, path):
        return f"http://{self.node_ips[peer_id]}:{self.node_ports[peer_id]}{path}"

    def _purge_stale_handshakes_locked(self, now, ttl_sec=120.0):
        stale_ids = [
            handshake_id
            for handshake_id, meta in self._pending_handshakes.items()
            if (now - float(meta.get("started_at", now))) > float(ttl_sec)
        ]
        for handshake_id in stale_ids:
            self._pending_handshakes.pop(handshake_id, None)

    def _get_channel_state(self, peer_id, logical_task_id, link_idx):
        key = (self.node_id, str(peer_id), int(logical_task_id), int(link_idx))
        with self._channel_lock:
            state = self._channel_states.get(key)
            if state is None:
                state = _ChannelState(
                    lock=threading.Lock(),
                    bucket=DynamicTokenBucket(capacity_bytes=self._tx_bucket_capacity_bytes),
                )
                self._channel_states[key] = state
            return state

    def start(self):
        if self._server_thread is not None:
            return
        port = self.node_ports[self.node_id]
        self._server_thread = _ServerThread(self._app, "0.0.0.0", port)
        self._server_thread.start()
        logging.info("[%s] HTTP dynamic transport listening on %s", self.node_id, port)

    def close(self):
        if self._server_thread is not None:
            self._server_thread.shutdown()
            self._server_thread.join(timeout=2.0)
            self._server_thread = None
        self._session.close()
        thread_session = getattr(self._thread_local, "session", None)
        if thread_session is not None:
            thread_session.close()

    def wait_for_peers(self, peer_ids, timeout=60.0, poll_interval=1.0):
        deadline = time.time() + timeout
        pending = set(peer_ids)
        while pending:
            if time.time() > deadline:
                raise TimeoutError(f"Peers not ready: {sorted(pending)}")
            ready = []
            for peer_id in list(pending):
                url = self._peer_url(peer_id, "/health")
                try:
                    response = self._session.get(url, timeout=min(3.0, poll_interval))
                    if response.ok:
                        ready.append(peer_id)
                except requests.RequestException:
                    continue
            for peer_id in ready:
                pending.discard(peer_id)
            if pending:
                time.sleep(poll_interval)

    def send(self, peer_id, msg_type, payload, timeout=None, handshake_meta=None):
        t0 = time.perf_counter()
        payload_bytes = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        handshake_id = uuid.uuid4().hex
        session = self._get_send_session()
        handshake_response = session.post(
            self._peer_url(peer_id, "/hs"),
            json={
                "handshake_id": handshake_id,
                "src": self.node_id,
                "dst": peer_id,
                "msg_type": int(msg_type),
                "handshake_meta": (dict(handshake_meta) if isinstance(handshake_meta, dict) else {}),
            },
            timeout=timeout or self.request_timeout,
        )
        handshake_response.raise_for_status()
        frame = FRAME_HEADER.pack(
            FRAME_VERSION,
            int(msg_type),
            ord(self.node_id),
            ord(peer_id),
            len(payload_bytes),
        ) + payload_bytes
        response = session.post(
            self._peer_url(peer_id, "/m"),
            data=frame,
            timeout=timeout or self.request_timeout,
            headers={
                "Content-Type": "application/octet-stream",
                "X-Handshake-ID": handshake_id,
            },
        )
        response.raise_for_status()
        return len(payload_bytes), float(time.perf_counter() - t0)

    def send_dynamic(
        self,
        peer_id,
        msg_type,
        payload,
        logical_task_id,
        link_idx,
        rate_bps,
        timeout=None,
        handshake_meta=None,
    ):
        t0 = time.perf_counter()
        payload_bytes = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        frame = FRAME_HEADER.pack(
            FRAME_VERSION,
            int(msg_type),
            ord(self.node_id),
            ord(peer_id),
            len(payload_bytes),
        ) + payload_bytes
        channel_state = self._get_channel_state(peer_id, logical_task_id, link_idx)
        session = self._get_send_session()
        with channel_state.lock:
            channel_state.bucket.update_rate(float(rate_bps) / 8.0)
            handshake_id = uuid.uuid4().hex
            handshake_response = session.post(
                self._peer_url(peer_id, "/hs"),
                json={
                    "handshake_id": handshake_id,
                    "src": self.node_id,
                    "dst": peer_id,
                    "msg_type": int(msg_type),
                    "handshake_meta": (dict(handshake_meta) if isinstance(handshake_meta, dict) else {}),
                },
                timeout=timeout or self.request_timeout,
            )
            handshake_response.raise_for_status()
            channel_state.bucket.acquire(len(frame))
            response = session.post(
                self._peer_url(peer_id, "/m"),
                data=frame,
                timeout=timeout or self.request_timeout,
                headers={
                    "Content-Type": "application/octet-stream",
                    "X-Handshake-ID": handshake_id,
                },
            )
            response.raise_for_status()
        return len(payload_bytes), float(time.perf_counter() - t0)

    def broadcast(self, peer_ids, msg_type, payload, timeout=None):
        results = {}
        for peer_id in peer_ids:
            results[peer_id] = self.send(peer_id, msg_type, payload, timeout=timeout)
        return results

    def receive(self, msg_type=None, source=None, predicate=None, timeout=None):
        deadline = None if timeout is None else time.time() + timeout
        with self._cv:
            while True:
                for envelope in list(self._inbox):
                    if msg_type is not None and envelope["msg_type"] != msg_type:
                        continue
                    if source is not None and envelope["src"] != source:
                        continue
                    if predicate is not None and not predicate(envelope["payload"]):
                        continue
                    self._inbox.remove(envelope)
                    return envelope

                if deadline is not None:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        raise TimeoutError("receive timeout")
                    self._cv.wait(timeout=remaining)
                else:
                    self._cv.wait()


def create_transport_dynamic(
    node_id,
    node_ips,
    node_ports,
    request_timeout=30.0,
    tx_bucket_capacity_bytes=None,
):
    return HttpTransportDynamic(
        node_id=node_id,
        node_ips=node_ips,
        node_ports=node_ports,
        request_timeout=request_timeout,
        tx_bucket_capacity_bytes=tx_bucket_capacity_bytes,
    )
