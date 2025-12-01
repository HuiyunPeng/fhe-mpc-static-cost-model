#!/usr/bin/env python3
"""
Create a grouped bar chart comparing estimated peak memory against the
measured peak memory from an FHE primitive trace.

Example usage:
    python tools/plot_peak_mem_bar.py \
        --trace-json data/mlp_primitive_trace.json \
        --estimate-csv data/mlp_peak_estimated.csv \
        --pdf-out data/mlp_peak_mem_bar.pdf
"""

import argparse
from pathlib import Path
from typing import Dict, Hashable, List, Tuple

from trace_report import load_estimates, load_ops


def op_key(op: Dict) -> Tuple[str, Hashable]:
    """
    Generate a stable key to align estimate rows with trace rows.
    Preference order: op_id -> name -> op_type.
    """
    if op.get("op_id") is not None:
        return ("id", op["op_id"])
    if op.get("name"):
        return ("name", op["name"])
    return ("type", op.get("op_type"))


def op_label(op: Dict) -> str:
    op_id = op.get("op_id")
    name = op.get("name") or op.get("op_type") or "op"
    return f"{op_id}: {name}" if op_id is not None else name


def to_mib(bytes_val: int) -> float:
    return (bytes_val or 0) / (1024 * 1024)


def build_series(ops: List[Dict]) -> Dict[Tuple[str, Hashable], Dict]:
    series: Dict[Tuple[str, Hashable], Dict] = {}
    for op in ops:
        series[op_key(op)] = {
            "label": op_label(op),
            "mib": to_mib(op.get("peak_mem_bytes", 0)),
        }
    return series


def build_figure(labels: List[str], actual: List[float], estimated: List[float]):
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise SystemExit(
            "plotly is required to generate the chart. "
            "Install it with `pip install plotly`."
        ) from exc

    fig = go.Figure()
    fig.add_bar(
        name="Trace Peak Memory (MiB)",
        x=labels,
        y=actual,
        marker_color="#1f77b4",
    )
    fig.add_bar(
        name="Estimated Peak Memory (MiB)",
        x=labels,
        y=estimated,
        marker_color="#ff7f0e",
        opacity=0.85,
    )

    fig.update_layout(
        barmode="group",
        yaxis_title="Peak Memory (MiB)",
        legend_title="",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.4,
            xanchor="center",
            x=0.5,
            font=dict(size=14),
        ),
        font=dict(size=14),
        template="plotly_white",
        bargap=0.25,
        margin=dict(b=140),
    )
    fig.update_xaxes(
        tickangle=-30,
        title_font=dict(size=18),
        tickfont=dict(size=16),
        title="Operator",
    )
    fig.update_yaxes(title_font=dict(size=16), tickfont=dict(size=12))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot estimated vs measured peak memory as a grouped bar chart."
    )
    parser.add_argument(
        "--trace-json",
        required=True,
        type=Path,
        help="Path to the primitive trace JSON (e.g., data/mlp_primitive_trace.json).",
    )
    parser.add_argument(
        "--estimate-csv",
        required=True,
        type=Path,
        help="Path to the estimated peak memory CSV (e.g., data/mlp_peak_estimated.csv).",
    )
    parser.add_argument(
        "--pdf-out",
        type=Path,
        help="Where to write the PDF figure (defaults beside the trace JSON).",
    )
    args = parser.parse_args()

    trace_ops = load_ops(args.trace_json, source=args.trace_json.stem, series="actual")
    est_ops = load_estimates(args.estimate_csv, source=args.estimate_csv.stem)

    trace_series = build_series(trace_ops)
    est_series = build_series(est_ops)

    all_keys = list({*trace_series.keys(), *est_series.keys()})
    all_keys.sort(key=lambda item: (item[0], item[1]))

    labels: List[str] = []
    actual_mib: List[float] = []
    estimated_mib: List[float] = []
    for key in all_keys:
        actual_entry = trace_series.get(key) or {}
        est_entry = est_series.get(key) or {}
        label = actual_entry.get("label") or est_entry.get("label") or str(key[1])
        labels.append(label)
        actual_mib.append(actual_entry.get("mib", 0.0))
        estimated_mib.append(est_entry.get("mib", 0.0))

    fig = build_figure(labels, actual_mib, estimated_mib)

    pdf_path = args.pdf_out or args.trace_json.with_name(f"{args.trace_json.stem}_peak_mem_bar.pdf")

    try:
        fig.write_image(str(pdf_path), format="pdf")
        print(f"Wrote PDF to {pdf_path}")
    except Exception as exc:
        raise SystemExit(
            "Creating a PDF requires the Plotly static image engine "
            "(install with `pip install -U plotly[kaleido]`)."
        ) from exc


if __name__ == "__main__":
    main()
