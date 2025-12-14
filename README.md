# Static Cost Model for FHE

This repository hosts our static cost model for fully homomorphic
encryption (FHE) inference. It builds directly on top of Orion’s CKKS
compiler to gather primitive traces, reason about packing/levels, and
produce reproducible memory estimates for FHE deployments. The goal is
to predict peak ciphertext and plaintext usage before execution, compare
those estimates with measured traces, and iterate on schemes or model
designs without running costly encrypted inference every time.

## What’s inside

- **Orion runtime** – We vendor Orion’s scheme initialization, automatic
  bootstrap placement, and neural network modules as the functional
  baseline for compiling PyTorch models to CKKS.
- **Primitive tracing** – Examples configure `PrimitiveTracer` so every
  encrypted op produces a structured JSON trace that records levels,
  buffer counts, and timing.
- **Static cost model tools** – Scripts in `tools/` consume those traces
  and CKKS configs to estimate per-operator peak memory, export CSV
  reports, and generate comparison plots. Two levels of estimation are
  available: `primitive_level_peak_mem_estimator.py` for detailed
  primitive-granularity analysis, and `op_level_peak_mem_estimator.py`
  for coarser operator-level estimates. Supporting utilities include
  `trace_report.py` and `plot_peak_mem_bar.py`.
- **Sample data & configs** – `configs/*.yml` reproduce LoLA, MLP, and
  ResNet runs; `data/` stores cached diagonals, keys, and example trace
  outputs to experiment with the estimator.
- **Tests & examples** – Minimal pytest coverage ensures modules import
  correctly, while `examples/run_*.py` run end-to-end encrypted
  inference with tracing enabled.

## Repository layout

| Path | Description |
| --- | --- |
| `orion/` | Orion core runtime, backends, NN ops, and models. |
| `configs/` | CKKS + Orion YAML configs used by the static analysis. |
| `examples/` | Scripts that train/compile LoLA, MLP, ResNet and emit traces. |
| `tools/` | Cost-model tooling (`peak_mem_estimator`, `trace_report`, plotting helpers). |
| `data/` | Default location for datasets, keys, diagonals, and produced traces/CSVs. |
| `tests/` | Smoke tests executed via pytest. |

## Installation

We test on Ubuntu 22.04 with Python 3.9–3.12.

```bash
sudo apt update && sudo apt install -y \
    build-essential git wget curl ca-certificates \
    python3 python3-pip python3-venv \
    unzip pkg-config libgmp-dev libssl-dev
```

Install Go for the Lattigo backend:

```bash
cd /tmp
wget https://go.dev/dl/go1.22.3.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.22.3.linux-amd64.tar.gz
echo 'export PATH=/usr/local/go/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
go version    # go version go1.22.3 linux/amd64
```

Clone and install this repository (editable install compiles the Go
bindings through `tools/build_lattigo.py`):

```bash
cd fhe-mpc-static-cost-model
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

## Static cost modeling workflow

1. **Generate a primitive trace**

   Each example script configures tracing:

   ```bash
   cd examples
   python run_lola.py
   ```

   The run downloads MNIST, fits polynomial activations, compiles the
   model, executes encrypted inference, and writes
   `data/lola_primitive_trace.json`. Adjust
   `scheme = orion.init_scheme("configs/*.yml")` to switch models.

2. **Estimate peak memory statically**

   Two levels of peak memory estimation are available:

   **Primitive-level estimation** (detailed, trace-based):
   ```bash
   python tools/primitive_level_peak_mem_estimator.py \
       --config configs/lola.yml \
       --trace data/lola_primitive_trace.json \
       --csv-out data/lola_peak_estimated.csv
   ```
   This approach analyzes individual primitive operations from the trace
   for fine-grained memory accounting.

   **Operator-level estimation** (static, synthetic shapes):
   ```bash
   python tools/op_level_peak_mem_estimator.py \
       --config configs/mlp.yml \
       --model MLP \
       --input-shape 1,784 \
       --csv-out data/mlp_op_peak_mem.csv
   ```
   This approach estimates per-operator memory directly from Orion model 
   metadata without requiring a trace, making it faster for preliminary analysis.
   Adjust `--model` and `--input-shape` to match your architecture.

3. **Compare against measured traces**

   Convert traces/estimates into a merged CSV and visualization:

   ```bash
   python tools/trace_report.py \
       data/lola_primitive_trace.json \
       --estimate-csv data/lola_peak_estimated.csv \
       --csv-out data/lola_ops_report.csv \
       --pdf-out data/lola_ops_plot.pdf

   python tools/plot_peak_mem_bar.py \
       --trace-json data/lola_primitive_trace.json \
       --estimate-csv data/lola_peak_estimated.csv \
       --pdf-out data/lola_peak_mem_bar.pdf
   ```

   The resulting CSV and PDF (or fallback HTML) show where the static
   model over/underestimates peak ciphertext buffers, enabling rapid
   iteration on packing strategies without repeatedly executing the FHE
   workload.

## Configuration tips

YAML files in `configs/` carry two sections:

- `ckks_params` – ring dimension (`LogN`), modulus ladder (`LogQ`,
  `LogP`), scale, secret-key Hamming weight, and ring type that match
  the target backend.
- `orion` – framework knobs: `margin` for activation range expansion,
  data embedding strategy, backend selection, file paths for persisted
  diagonals/keys, I/O mode, and trace logging location.

Use the supplied configs as templates when designing new circuits or
running what-if experiments for different modulus chains.

## Sample data

`data/` contains cached MNIST/CIFAR downloads, serialized diagonals and
keys to skip regeneration, plus pre-generated primitive traces and Chart
outputs. Delete or override files there if you prefer a clean run.

