import json
import os
import threading
import time
import resource
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple


class PrimitiveTracer:
    """
    Lightweight tracing utility to map high-level Orion operators to backend
    primitives (HAdd, HMult, HRot, Bootstrap, etc.). When enabled, high-level
    modules wrap their forward passes in ``op_context`` and backend calls
    register their primitive name via ``log_primitive``. The tracer then keeps
    a flat list of operator metadata and primitive events that share an
    ``op_id``.
    """

    def __init__(self, enabled: bool = False, output_path: Optional[str] = None):
        self.enabled = enabled
        self.output_path = output_path or "orion_primitive_trace.json"
        self._local = threading.local()
        self.reset()

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def reset(self):
        self.high_level_ops = []
        self.primitive_calls = []
        self._local.current_op_id = None
    
    @staticmethod
    def _get_rss_bytes():
        """
        Best-effort current RSS in bytes.
        Prefer /proc/self/statm (Linux), fall back to resource.ru_maxrss.
        """
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            with open("/proc/self/statm", "r", encoding="utf-8") as f:
                parts = f.readline().split()
                if len(parts) >= 2:
                    rss_pages = int(parts[1])
                    return rss_pages * page_size
        except Exception:
            pass

        try:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            peak = usage.ru_maxrss
            return int(peak * 1024) if peak and peak < 10**9 else int(peak)
        except Exception:
            return None

    @staticmethod
    def _get_peak_mem_bytes():
        """Best-effort peak RSS in bytes; returns None if unavailable."""
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            # ru_maxrss is kilobytes on Linux, bytes on macOS; normalize to bytes.
            peak = usage.ru_maxrss
            return int(peak * 1024) if peak and peak < 10**9 else int(peak)
        except Exception:
            return None
    
    def _sample_peak(self, stop_event: threading.Event) -> Tuple[int, int]:
        """
        Background sampler to capture the maximum observed RSS during an op.
        Returns (peak_bytes, samples_taken).
        """
        peak = 0
        samples = 0
        while not stop_event.is_set():
            rss = self._get_rss_bytes() or peak
            peak = max(peak, rss)
            samples += 1
            time.sleep(0.001)  # 1ms sampling; adjust for overhead vs fidelity
        return peak, samples

    @contextmanager
    def op_context(
        self,
        op_type: str,
        name: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        Context manager to denote execution of a high-level Orion operator.
        All primitives logged within this context inherit the generated
        ``op_id`` to build a mapping of op -> primitives.
        """
        if not self.enabled:
            yield None
            return

        op_id = len(self.high_level_ops)
        start_rss = self._get_rss_bytes()
        stop_evt = threading.Event()
        peak_container = {}
        # Start a lightweight sampler thread to capture true peak during op.
        def _runner():
            peak, samples = self._sample_peak(stop_evt)
            peak_container["peak"] = peak
            peak_container["samples"] = samples

        sampler = threading.Thread(target=_runner, daemon=True)
        sampler.start()
        start = time.time()
        op_entry = {
            "op_id": op_id,
            "op_type": op_type,
            "name": name,
            "params": params or {},
        }
        self.high_level_ops.append(op_entry)

        parent = getattr(self._local, "current_op_id", None)
        self._local.current_op_id = op_id
        try:
            yield op_id
        finally:
            op_entry["duration_sec"] = time.time() - start

            # Stop sampler and compute peak delta.
            stop_evt.set()
            sampler.join(timeout=0.1)

            peak = peak_container.get("peak")
            end_rss = self._get_rss_bytes()
            baseline = start_rss if start_rss is not None else end_rss
            if peak is None and end_rss is not None:
                peak = end_rss
            if peak is not None and baseline is not None:
                op_entry["peak_mem_bytes"] = max(0, peak - baseline)
                op_entry["peak_mem_samples"] = peak_container.get("samples", 0)
            self._local.current_op_id = parent

    def log_primitive(self, primitive: str, params: Optional[Dict[str, Any]] = None):
        """Record a backend primitive occurrence."""
        if not self.enabled:
            return

        op_id = getattr(self._local, "current_op_id", None)
        self.primitive_calls.append(
            {
                "op_id": op_id,
                "primitive": primitive,
                "params": params or {},
                "timestamp": time.time(),
            }
        )

    def snapshot(self) -> Dict[str, Any]:
        return {"ops": self.high_level_ops, "primitives": self.primitive_calls}

    def save(self, path: Optional[str] = None) -> Optional[str]:
        """
        Persist the trace to disk as JSON. Returns the path when tracing is
        enabled, or None if tracing was disabled.
        """
        if not self.enabled:
            return None

        out_path = path or self.output_path
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.snapshot(), f, indent=2)
        return out_path
