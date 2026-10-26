# Tail-Latency Amplification and Coupled Thread-Pool Saturation in Cloud Microservice Call Graphs, with Deadline-Aware Overload Control
### Official Artifact & Replication Package for Springer Nature Journal Submission

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-93%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Manuscript: Springer Nature](https://img.shields.io/badge/manuscript-Springer%20Nature%20Q1-blue.svg)](paper/main.pdf)

This repository contains the complete experimental testbed, analytical queueing models, measurement datasets (540 accepted campaign runs + 200 retained sensitivity runs), offline verification scripts, and LaTeX manuscript source for the study:

> **Tail-latency amplification and coupled thread-pool saturation in cloud microservice call graphs, with deadline-aware overload control**  
> **Authors:**  
> - **Devarsh Patel** (Contributing Author) — [devarsh.patel2024@vitstudent.ac.in](mailto:devarsh.patel2024@vitstudent.ac.in)  
> - **Ranjithkumar S** (Corresponding Author) — [ranjithkumar.s@vit.ac.in](mailto:ranjithkumar.s@vit.ac.in)  
> *School of Computer Science and Engineering (SCOPE), Vellore Institute of Technology (VIT), Vellore, Tamil Nadu, India*

---

## 🔬 Overview & Key Findings

Microservice requests fan out across directed acyclic dependency graphs, where each synchronous upstream call holds a worker thread until its downstream dependencies return. This study establishes a coupled $G/G/m$ holding-time model for bounded call graphs and evaluates it on a process-per-vertex gRPC benchmark testbed across 540 completed runs in five phases:

1. **Thread-Pool vs. Processor Saturation:** Synchronous service chains saturate their root worker pools ($\rho_{\text{pool}} \to 1$) while host processors remain only ~10% busy ($0.75\text{--}0.79$ pool utilization with CPU idling), whereas asynchronous and hybrid chains avoid pool starvation up to 191 req/s.
2. **Tail Amplification ($\Gamma_{\text{crit}}$ & $\Gamma_{\text{load}}$):** P99.9 latency at the highest stable load amplifies by $2.8\times$ to $8.1\times$ the critical-path median sum for 17 of 18 shapes, and reaches $20.5\times$ for width-eight shared-leaf in-cast topologies.
3. **Observability & Interception Overhead:** Distributed tracing ($S \ge 0.25$) adds $+4.3$ to $+10.6\text{ ms}$ at P99 under high load (pooled median $+5.50\text{ ms}$, exact sign test $p = 0.03125$), representing a conservative lower bound given higher baseline background CPU load.
4. **Overload Control Under Injected Faults:** Under capacity faults (slowdown and hang), a deadline-aware controller (**DAC-Shed**) restricts P99.9 of successful requests below $260\text{ ms}$, where static circuit breakers and naive retries explode to $>2.7\text{ s}$.
5. **Budget-Gated Retry Rule:** Under transient callee errors, an offline trace replay of a budget-and-congestion-gated single retry recovers $70.9\%$ of first-attempt errors, lifting goodput to $147.7\text{ req/s}$ with only $+2.7\%$ extra hop calls, while blocking 100% of retries under capacity faults.
6. **Multi-Phase Host Gate Invariance:** Relaxing the quiet-host admission gate from $1.0$ to $1.2\text{ cores}$ across 62 borderline runs leaves all 18 Phase 1 saturation brackets $\Theta^*$ and all Phase 3–5 controller rankings 100% invariant.

---

## 📁 Repository Structure

```
.
├── paper/                    # Springer Nature manuscript source and build scripts
│   ├── main.tex              # Master LaTeX document (sn-jnl.cls template)
│   ├── sn-jnl.cls            # Official Springer Nature journal class
│   ├── compile_paper.sh      # Automated Tectonic / PDFLaTeX compilation script
│   ├── figures/              # Vectorized architecture and performance figures
│   └── sections/             # Modular section LaTeX files (01_intro to 10_conclusion)
├── src/                      # Core testbed harness, analytical models, and controllers
│   ├── analytical_dag_model.py  # Coupled G/G/m holding-time model & Kingman approximations
│   ├── dag_topology_builder.py  # Topologies: Serial Chains, Trees, Diamond In-Cast, DeathStarBench
│   ├── dac_shed_interceptor.py  # Deadline-budget and Vegas queueing-gradient interceptor
│   ├── overload_controller.py   # DAC-Shed, Token Bucket, and Static Circuit Breaker
│   ├── node_retry.py            # Budget- and gradient-gated retry gate
│   ├── host_gate.py             # Quiet-host continuous contamination gate
│   ├── result_loader.py         # Fast Parquet telemetry and request record loaders
│   ├── make_tables.py           # Reproduces LaTeX tables from stored metrics
│   └── plot_journal_figures.py  # Generates publication PDF figures
├── scripts/                  # Deterministic verification and sensitivity scripts
│   ├── run_journal_final_checks.py        # All 5 journal post-hoc checks & metrics
│   ├── relaxed_gate_sensitivity.py        # Phase 1 quiet-host sensitivity sweep
│   ├── compute_phase4_bootstrap.py        # Phase 4 high-load tracing bootstrap & sign test
│   ├── compute_phase5_spreads.py          # Phase 5 replicate spreads and goodput CIs
│   └── compute_offline_retry_bound.py     # Offline trace-replay counterfactual evaluation
├── proto/                    # Wire protocol specifications
│   └── mesh_service.proto    # gRPC service definition with deadline metadata
├── results/                  # Empirical evaluation metrics and raw traces
│   ├── metrics/              # CSV and JSON summary tables
│   ├── figures/              # Compiled vector publication figures (Fig 1–8)
│   └── raw_traces/           # Per-run Parquet traces for Phases 1–5 (540 accepted + 200 sensitivity)
├── tables/                   # Generated LaTeX tables (tab01 to tab14)
├── literature/               # Peer-reviewed BibTeX literature database
├── tests/                    # Automated pytest test suite (93 unit & regression tests)
└── requirements.txt          # Python runtime dependencies
```

---

## 🚀 Quick Start & Reproduction Guide

### 1. Environment Setup

Python 3.12 or newer is recommended. Install required dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Compile protocol buffer definitions (if modifying `.proto` files):
```bash
python3 -m grpc_tools.protoc -Iproto --python_out=src --grpc_python_out=src proto/mesh_service.proto
```

### 2. Run Automated Test Suite

Verify testbed models, overload controllers, telemetry parsers, and statistical routines:
```bash
python3 -m pytest tests/
```
*Expected output: `93 passed`.*

### 3. Run Deterministic Verification Checks

Execute the offline verification script that computes all five post-hoc empirical checks directly from the stored trace corpus:
```bash
python3 scripts/run_journal_final_checks.py
```
This script computes:
- Phase 5 request-level binomial bootstrap 95% confidence intervals across all 45 factorial cells.
- Phase 5 retry gate evaluation across all three fault modes (`errors`, `slowdown`, `hang`).
- Multi-phase relaxed quiet-host gate sensitivity ($U = 1.2\text{ cores}$) across Phases 3, 4, and 5.
- Phase 4 high-load background-adjusted regression and exact sign test.
- Phase 2 shared-leaf process CPU and local work / pool wait correlation on `diamond_D5_W8`.

Outputs are written deterministically to `results/metrics/journal_final_checks.json` and `results/metrics/sensitivity_gate_phases35.csv`.

### 4. Regenerate Manuscript Tables & Figures

Regenerate summary tables and publication figures from the metrics:
```bash
python3 src/make_tables.py
python3 src/plot_journal_figures.py
```

### 5. Compile the LaTeX Manuscript

Compile `paper/main.tex` to `paper/main.pdf` using Tectonic or PDFLaTeX:
```bash
bash paper/compile_paper.sh
```
The compiled PDF is generated at `paper/main.pdf` (44 pages, official Springer Nature `sn-jnl.cls` template).

---

## 📊 Dataset Description

The trace corpus comprises:
- **540 Accepted Campaign Runs:** Complete telemetry, per-request latency logs, and sampled hop records across 5 factorial phases:
  - Phase 1: 18 shapes $\times$ 5 load levels $\times$ 2 replicates = 180 runs.
  - Phase 2: 3 shapes $\times$ 3 runtimes $\times$ 5 load levels $\times$ 3 replicates = 135 runs.
  - Phase 3: 3 depths $\times$ 5 payload sizes $\times$ 5 replicates = 75 runs.
  - Phase 4: 5 tracing sampling rates $\times$ 2 exporter modes $\times$ 2 load levels = 60 runs.
  - Phase 5: 3 fault modes $\times$ 3 overload controllers $\times$ 5 load levels $\times$ 2 replicates = 90 runs.
- **200 Retained Contaminated Runs:** Stored run directories with full telemetry used for empirical sensitivity auditing of the 1.0-core quiet-host admission gate.

All trace files are stored in Apache Parquet format (`requests.parquet`, `hops.parquet`, `telemetry.parquet`) with metadata manifests (`meta.json`).

---

## 📄 License & Attribution

This project is licensed under the [MIT License](LICENSE).

```bibtex
@article{patel2026taillatency,
  title={Tail-latency amplification and coupled thread-pool saturation in cloud microservice call graphs, with deadline-aware overload control},
  author={Patel, Devarsh and Ranjithkumar, S},
  journal={Springer Nature Journal Submission},
  year={2026}
}
```
