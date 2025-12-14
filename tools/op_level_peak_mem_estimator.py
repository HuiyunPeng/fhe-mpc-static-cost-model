import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml

import orion
import orion.models as models
import orion.nn as on
from orion.core.tracer import OrionTracer, StatsTracker

@dataclass
class CkksParams:
    logn: int
    logq: list  # full chain
    overhead: float = 1.05  # metadata cushion

    @property
    def n(self): return 2 ** self.logn

    def ct_bytes(self, level: int) -> int:
        logq_active = sum(self.logq[: level + 1])
        return int(2 * self.n * logq_active / 8 * self.overhead)  # 2 polys

    def pt_bytes(self, level: int) -> int:
        logq_active = sum(self.logq[: level + 1])
        return int(self.n * logq_active / 8 * self.overhead)

def linear_mem(op, params: CkksParams):
    ct = params.ct_bytes(op["level"])
    # input + accumulator + rotations + output
    return ct * (2 + op.get("output_rotations", 0) + 1)

def poly_mem(op, params: CkksParams):
    ct = params.ct_bytes(op["level"])
    poly_depth = op.get("degree", op.get("depth", 1))
    return ct * (1 + poly_depth + 1)  # in + temps + out

ESTIMATORS = {
    "Linear": linear_mem,
    "Conv2d": linear_mem,
    "Activation": poly_mem,
    "Quad": lambda op, p: poly_mem({**op, "degree": 2}, p),
    "BatchNorm1d": lambda op, p: p.ct_bytes(op["level"]) * 2 + p.pt_bytes(op["level"]),
    "BatchNorm2d": lambda op, p: p.ct_bytes(op["level"]) * 2 + p.pt_bytes(op["level"]),
}

def estimate_ops(ops, ckks_cfg):
    params = CkksParams(logn=ckks_cfg["LogN"], logq=ckks_cfg["LogQ"])
    results = []
    for op in ops:
        fn = ESTIMATORS.get(op["op_type"])
        if not fn:
            continue
        level = op.get("params", {}).get("level")
        if level is None:
            # Skip ops without assigned level (non-FHE ops like Flatten)
            continue
        results.append({
            "op_id": op["op_id"],
            "op_type": op["op_type"],
            "name": op.get("name"),
            "level": level,
            "peak_mem_bytes": fn({**op, **op.get("params", {})}, params),
        })
    return results


def _collect_ops_from_model(net) -> List[Dict[str, Any]]:
    """
    Collect per-op metadata (shapes, level, depth) from leaf Orion modules.
    Assumes scheme.fit/compile has already populated module attributes.
    """
    ops = []
    for name, module in net.named_modules():
        if not isinstance(module, on.Module):
            continue
        if len(list(module.children())) > 0:
            continue  # only leaves (actual ops)

        params = {
            "input_shape": tuple(getattr(module, "input_shape", []) or []),
            "output_shape": tuple(getattr(module, "output_shape", []) or []),
            "fhe_input_shape": tuple(getattr(module, "fhe_input_shape", []) or []),
            "fhe_output_shape": tuple(getattr(module, "fhe_output_shape", []) or []),
            "level": getattr(module, "level", None),
            "depth": getattr(module, "depth", None),
            "output_rotations": getattr(module, "output_rotations", 0),
        }

        ops.append(
            {
                "op_id": len(ops),
                "op_type": module.__class__.__name__,
                "name": name,
                "params": params,
            }
        )
    return ops


def _parse_shape(shape_str: str):
    try:
        return tuple(int(x) for x in shape_str.split(","))
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid shape '{shape_str}'. Use comma-separated ints, e.g., 1,1,28,28."
        ) from exc


def _load_ckks_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ckks = cfg.get("ckks_params", {})
    if not {"LogN", "LogQ"} <= set(ckks):
        raise ValueError("Config must include ckks_params.LogN and ckks_params.LogQ")
    return ckks


def main():
    parser = argparse.ArgumentParser(
        description="Estimate per-op peak memory from CKKS params and synthetic shapes."
    )
    parser.add_argument("--config", required=True, help="Path to Orion YAML config")
    parser.add_argument("--model", required=True, help="Model class name in orion.models (e.g., MLP)")
    parser.add_argument(
        "--input-shape",
        required=True,
        type=_parse_shape,
        help="Comma-separated input shape, e.g., 1,1,28,28",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        help="Optional path to save per-op peak memory CSV (defaults beside config)",
    )
    args = parser.parse_args()

    ckks_cfg = _load_ckks_cfg(args.config)

    scheme = orion.init_scheme(args.config)
    try:
        model_cls = getattr(models, args.model)
    except AttributeError:
        raise SystemExit(f"Unknown model '{args.model}' in orion.models")

    net = model_cls()
    net.eval()

    dummy = torch.zeros(args.input_shape)

    # Populate shapes/statistics without real data (plain torch forward)
    scheme.fit(net, dummy)
    scheme.compile(net)

    ops = _collect_ops_from_model(net)
    results = estimate_ops(ops, ckks_cfg)

    print("\nPer-op peak memory estimate (bytes):")
    for entry in results:
        level = entry.get("level")
        mem = entry["peak_mem_bytes"]
        print(
            f"- {entry['op_id']:02d} {entry['name']} ({entry['op_type']}), "
            f"level={level}: {mem:,d}"
        )

    # Optionally persist results to CSV for downstream analysis/reporting.
    csv_path = args.csv_out
    if csv_path is None:
        cfg_path = Path(args.config)
        csv_path = cfg_path.with_name(f"{cfg_path.stem}_{args.model}_peak_mem.csv")

    Fieldnames = ["op_id", "name", "op_type", "level", "peak_mem_bytes", "peak_mem_mib"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=Fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "op_id": row["op_id"],
                    "name": row.get("name"),
                    "op_type": row.get("op_type"),
                    "level": row.get("level"),
                    "peak_mem_bytes": row["peak_mem_bytes"],
                    "peak_mem_mib": row["peak_mem_bytes"] / (1024 * 1024),
                }
            )
    print(f"\nSaved CSV to {csv_path}")


if __name__ == "__main__":
    main()