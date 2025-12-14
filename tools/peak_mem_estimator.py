#!/usr/bin/env python3
import argparse
import json
import math
import csv
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import yaml


class CkksParams:
    """
    Minimal CKKS parameter helper to estimate plaintext/ciphertext footprint.

    Assumptions:
      - ciphertext stores two polynomials
      - each polynomial uses all moduli up to `level` (inclusive)
      - plaintext uses a single polynomial with the active modulus set, modeled
        as using the sum of active LogQ up to `level`
      - a small overhead factor accounts for metadata/allocator slack
    """

    def __init__(
        self,
        logn: int,
        logq: List[int],
        logp: List[int] = None,
        overhead: float = 1.05,
    ):
        self.logn = logn
        self.logq = logq
        self.logp = logp or []
        self.ring_degree = 1 << logn
        self.overhead = overhead

    @classmethod
    def from_config(cls, cfg: Union[str, Dict]) -> "CkksParams":
        """
        Accepts a YAML path or a config dictionary with CKKS parameters under
        the `ckks_params` key (mirrors Orion's config structure).
        """
        if isinstance(cfg, str):
            with open(cfg, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)

        ckks_cfg = cfg.get("ckks_params", cfg)
        return cls(
            logn=ckks_cfg["LogN"],
            logq=ckks_cfg["LogQ"],
            logp=ckks_cfg.get("LogP", []),
        )

    def _clamp_level(self, level: int) -> int:
        if level is None:
            return len(self.logq) - 1
        return max(0, min(level, len(self.logq) - 1))

    def ct_bytes(self, level: int) -> int:
        """
        Estimate ciphertext bytes at a given level.
        2 polys * N coefficients * sum(logQi) bits, times overhead.
        """
        lvl = self._clamp_level(level)
        bitwidth = 2 * self.ring_degree * sum(self.logq[: lvl + 1])
        return math.ceil(bitwidth / 8 * self.overhead)

    def pt_bytes(self, level: int) -> int:
        """
        Estimate plaintext bytes at a given level.
        1 poly * N coefficients * sum(logQi) bits, times overhead.
        """
        lvl = self._clamp_level(level)
        bitwidth = self.ring_degree * sum(self.logq[: lvl + 1])
        return math.ceil(bitwidth / 8 * self.overhead)


# -------------------------- Trace loading -------------------------- #


def load_trace(path: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Load a combined ops + primitives trace produced by the Orion backend
    primitive logger.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("ops", []), data.get("primitives", [])


def load_ops(path: str) -> List[Dict]:
    """
    Load ops from a JSON file. Expected format:
      { "ops": [ ... ] }
    or directly a list of ops.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("ops", data)


def group_primitives_by_op(primitives: List[Dict]) -> Dict[int, List[Dict]]:
    """
    Group primitive records by op_id, ignoring primitives with op_id = null.
    """
    grouped = defaultdict(list)
    for prim in primitives:
        op_id = prim.get("op_id")
        if op_id is None:
            continue
        grouped[op_id].append(prim)
    return grouped


# ---------------------- Primitive-level model ---------------------- #


def prim_peak_buffers(primitive: Dict) -> Tuple[int, int]:
    """
    Return an approximate (num_ciphertexts, num_plaintexts) live at the
    peak of this primitive's execution.

    These are conservative templates and can be refined over time.
    """
    prim_type = primitive.get("primitive")
    params = primitive.get("params", {}) or {}

    # base mapping
    mapping = {
        "HAdd": (3, 0),       # in1, in2, out
        "HMult": (4, 0),      # in1, in2, temp, out
        "PMult": (2, 1),      # ct + pt
        "HRot": (3, 0),       # in, rotated, maybe accumulator
        "Rescale": (2, 0),    # in + out
        "MatVec": (4, 0),     # BSGS block (very rough)
        "Bootstrap": (8, 4),  # large workspace
    }

    ct_cnt, pt_cnt = mapping.get(prim_type, (2, 0))

    # Simple handling of in-place ops: they need fewer temporaries.
    # If the primitive is explicitly marked in_place=true, reduce by 1 ct.
    if params.get("in_place") and ct_cnt > 1:
        ct_cnt -= 1

    return ct_cnt, pt_cnt


def primitive_peak_memory(
    primitive: Dict, params: CkksParams, fallback_level: int = None
) -> int:
    """
    Estimate the peak memory for a single primitive, in bytes, based on the
    number of ciphertexts and plaintexts it holds live and the CKKS params.
    """
    level = fallback_level
    prim_params = primitive.get("params")
    if isinstance(prim_params, dict):
        level = prim_params.get("level", prim_params.get("output_level", level))
        if level is None:
            level = prim_params.get("input_level", fallback_level)

    ct_cnt, pt_cnt = prim_peak_buffers(primitive)
    ct_bytes = params.ct_bytes(level)
    pt_bytes = params.pt_bytes(level)
    return ct_cnt * ct_bytes + pt_cnt * pt_bytes


def estimate_peak_from_primitives(
    ops: List[Dict],
    prims_by_op: Dict[int, List[Dict]],
    ckks_cfg: Union[str, Dict],
) -> List[Dict]:
    """
    Primitive-aware per-op peak memory estimation.
    For each op, we aggregate all its primitives and take the maximum
    primitive peak as the op's peak memory.
    """
    ckks = CkksParams.from_config(ckks_cfg)
    results = []

    for op in ops:
        op_id = op.get("op_id")
        params = op.get("params", {}) if isinstance(op.get("params"), dict) else {}
        level = params.get("level")
        prims = prims_by_op.get(op_id, [])

        peak = 0
        for p in prims:
            prim_peak = primitive_peak_memory(p, ckks, fallback_level=level)
            if prim_peak > peak:
                peak = prim_peak

        if peak == 0:
            ct_cnt, pt_cnt = op_peak_buffers(op.get("op_type"))
            peak = ct_cnt * ckks.ct_bytes(level) + pt_cnt * ckks.pt_bytes(level)

        results.append(
            {
                "op_id": op_id,
                "name": op.get("name"),
                "op_type": op.get("op_type"),
                "level": level,
                "peak_mem_bytes": peak,
            }
        )
    return results


# --------------------- Fallback operator-level model --------------------- #

_OP_HEURISTIC_BUFFERS = {
    "Conv2d": (4, 0),
    "Linear": (3, 0),
    "Add": (3, 0),
    "ReLU": (2, 0),
    "SiLU": (2, 0),
    "AvgPool2d": (3, 0),
}


def op_peak_buffers(op_type: str) -> Tuple[int, int]:
    return _OP_HEURISTIC_BUFFERS.get(op_type, (2, 0))


def estimate_peak_from_ops(
    ops: List[Dict],
    ckks_cfg: Union[str, Dict],
) -> List[Dict]:
    """
    Legacy operator-level-only peak memory estimator.
    Uses coarse heuristics per op_type and CKKS params.
    """
    ckks = CkksParams.from_config(ckks_cfg)
    results = []
    for op in ops:
        params = op.get("params", {}) if isinstance(op.get("params"), dict) else {}
        level = params.get("level")
        ct_cnt, pt_cnt = op_peak_buffers(op.get("op_type"))
        peak = ct_cnt * ckks.ct_bytes(level) + pt_cnt * ckks.pt_bytes(level)
        results.append(
            {
                "op_id": op.get("op_id"),
                "name": op.get("name"),
                "op_type": op.get("op_type"),
                "level": level,
                "peak_mem_bytes": peak,
            }
        )
    return results


# -------------------------- CLI + pretty printing -------------------------- #


def fmt_bytes(num: int) -> str:
    if num is None:
        return "n/a"
    n = float(num)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:,.2f} {unit}"
        n /= 1024.0
    return f"{n:,.2f} TB"


def main():
    parser = argparse.ArgumentParser(
        description="Estimate CKKS peak memory usage from Orion traces."
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to Orion config (YAML) with ckks_params.",
    )
    parser.add_argument(
        "--trace",
        help="Path to primitive trace JSON (ops + primitives). "
        "If provided, uses primitive-based estimator.",
    )
    parser.add_argument(
        "--ops",
        help="Path to ops JSON (if no --trace). Should contain 'ops' list or be a list.",
    )
    parser.add_argument(
        "--csv-out",
        help="Optional path to write per-op results as CSV. "
        "If omitted, results are printed to stdout.",
    )
    args = parser.parse_args()

    # Decide which estimator to use
    if args.trace:
        ops, prims = load_trace(args.trace)
        prims_by_op = group_primitives_by_op(prims)
        results = estimate_peak_from_primitives(ops, prims_by_op, args.config)
        mode = "primitive-aware"
    else:
        if not args.ops:
            raise SystemExit("Provide --trace or --ops for estimation.")
        ops = load_ops(args.ops)
        results = estimate_peak_from_ops(ops, args.config)
        mode = "operator-only (heuristic)"

    # CSV output if requested
    if args.csv_out:
        fieldnames = ["op_id", "name", "op_type", "level", "peak_mem_bytes"]
        with open(args.csv_out, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow({k: row.get(k) for k in fieldnames})
        print(f"Wrote {mode} peak-memory estimates to CSV: {args.csv_out}")
        return

    # Otherwise pretty-print to stdout
    print(f"Estimation mode: {mode}")
    print(f"{'op_id':>5}  {'name':20}  {'op_type':12}  {'level':>5}  {'peak':>16}")
    print("-" * 68)
    for row in results:
        print(
            f"{str(row['op_id']):>5}  "
            f"{str(row['name'] or ''):20.20}  "
            f"{str(row['op_type'] or ''):12.12}  "
            f"{str(row['level']):>5}  "
            f"{fmt_bytes(row['peak_mem_bytes']):>16}"
        )


if __name__ == "__main__":
    main()
