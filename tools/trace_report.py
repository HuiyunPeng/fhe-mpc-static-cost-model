import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def load_ops(trace_path: Path, source: str, series: str = "trace") -> List[Dict[str, Any]]:
    with trace_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if "ops" not in payload or not isinstance(payload["ops"], list):
        raise ValueError(f"{trace_path} does not contain an 'ops' list")

    ops: List[Dict[str, Any]] = []
    for op in payload["ops"]:
        op_copy = dict(op)
        op_copy["_source"] = source
        op_copy["_series"] = series
        ops.append(op_copy)

    return ops


def load_estimates(csv_path: Path, source: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            peak_mem_bytes_raw = row.get("peak_mem_bytes")
            peak_mem_bytes = int(float(peak_mem_bytes_raw)) if peak_mem_bytes_raw else 0
            rows.append(
                {
                    "op_id": int(row["op_id"]) if row.get("op_id") else None,
                    "name": row.get("name"),
                    "op_type": row.get("op_type"),
                    "peak_mem_bytes": peak_mem_bytes,
                    "peak_mem_mib": peak_mem_bytes / (1024 * 1024),
                    "_source": source,
                    "_series": "estimate",
                }
            )
    return rows


def write_csv(ops: List[Dict[str, Any]], csv_path: Path) -> None:
    fieldnames = [
        "series",
        "source",
        "op_id",
        "name",
        "op_type",
        "peak_mem_bytes",
        "peak_mem_mib",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for op in ops:
            peak_mem_bytes = op.get("peak_mem_bytes", 0) or 0
            entry = {
                "series": op.get("_series"),
                "source": op.get("_source"),
                "op_id": op.get("op_id"),
                "name": op.get("name"),
                "op_type": op.get("op_type"),
                "peak_mem_bytes": peak_mem_bytes,
                "peak_mem_mib": peak_mem_bytes / (1024 * 1024),
            }
            writer.writerow(entry)


def build_figure(ops: List[Dict[str, Any]]):
    try:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
    except ImportError as exc:
        raise SystemExit(
            "plotly is required to generate the chart. "
            "Install it with `pip install plotly`."
        ) from exc

    def op_label(op: Dict[str, Any]) -> str:
        return f"{op.get('op_id', '?')}: {op.get('name', '')}".strip()

    trace_ops = [op for op in ops if op.get("_series") != "estimate"]
    est_ops = [op for op in ops if op.get("_series") == "estimate"]

    labels: List[str] = []
    seen = set()
    for op in trace_ops + est_ops:
        label = op_label(op)
        if label not in seen:
            seen.add(label)
            labels.append(label)

    trace_mem_by_label = {
        op_label(op): (op.get("peak_mem_bytes", 0) or 0) / (1024 * 1024)
        for op in trace_ops
    }
    est_mem_by_label = {
        op_label(op): (op.get("peak_mem_bytes", 0) or 0) / (1024 * 1024)
        for op in est_ops
    }

    x_labels = labels
    trace_mem = [trace_mem_by_label.get(label, 0.0) for label in labels]
    est_mem = [est_mem_by_label.get(label, 0.0) for label in labels]

    fig = make_subplots(
        rows=1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Peak Memory (MiB)",),
    )

    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=trace_mem,
            name="Trace Peak Memory (MiB)",
            mode="lines+markers",
            line=dict(color="#1f77b4"),
        ),
        row=1,
        col=1,
    )

    if est_ops:
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=est_mem,
                name="Estimated Peak Memory (MiB)",
                mode="lines+markers",
                line=dict(color="#ff7f0e", dash="dash"),
            ),
            row=1,
            col=1,
        )

    fig.update_layout(
        title="Operator Peak Memory",
        xaxis_title="Operator",
        legend_title="Metric",
        template="plotly_white",
    )
    fig.update_yaxes(title_text="Peak Memory (MiB)", row=1, col=1)

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an FHE trace JSON to CSV and Plotly chart."
    )
    parser.add_argument(
        "trace_jsons",
        type=Path,
        nargs="+",
        help="Path(s) to trace JSON file(s) containing an 'ops' array.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        help="Where to write the operator metrics CSV. "
        "Defaults to <trace_json stem>_ops.csv in the same directory.",
    )
    parser.add_argument(
        "--estimate-csv",
        type=Path,
        action="append",
        help="Optional estimated peak memory CSV(s) to overlay on the memory chart.",
    )
    parser.add_argument(
        "--pdf-out",
        type=Path,
        help="Where to write the Plotly PDF file. "
        "Defaults to <trace_json stem>_ops.pdf in the same directory.",
    )

    args = parser.parse_args()

    ops: List[Dict[str, Any]] = []
    for path in args.trace_jsons:
        ops.extend(load_ops(path, source=path.stem))

    if args.estimate_csv:
        for est_path in args.estimate_csv:
            ops.extend(load_estimates(est_path, source=f"{est_path.stem} (est)"))

    primary = args.trace_jsons[0]
    csv_path = args.csv_out or primary.with_name(
        f"{primary.stem}_ops.csv"
    )
    pdf_path = args.pdf_out or primary.with_name(
        f"{primary.stem}_ops.pdf"
    )

    write_csv(ops, csv_path)
    fig = build_figure(ops)
    try:
        fig.write_image(str(pdf_path), format="pdf")
    except Exception as exc:
        raise SystemExit(
            "Creating a PDF requires the Plotly static image engine "
            "(install with `pip install -U plotly[kaleido]`)."
        ) from exc

    print(f"Wrote CSV to {csv_path}")
    print(f"Wrote Plotly PDF to {pdf_path}")


if __name__ == "__main__":
    main()
