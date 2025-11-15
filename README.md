# Towards Causal Market Simulators

This repository contains the implementation of the paper **"Towards Causal Market Simulators" (ICAIF 2025)** by Dennis Thumm and Luis Ontaneda Mijares.

## Overview

We propose the **Time-series Neural Causal Model VAE (TNCM-VAE)** — a generative model that combines **variational autoencoders (VAEs)** with **structural causal models (SCMs)** to generate **counterfactual financial time series**.  
The model enforces causal constraints through a **directed acyclic graph (DAG)** in the decoder and uses the **causal Wasserstein distance** during training.

A **Conditional VAE (CVAE)** is included as a benchmark for comparison.

## Repository Structure

```
├── README.md                          # Project README (paper implementation)
├── README_temporal_vae.md             # Notes for temporal VAE
├── requirements.txt                   # Python dependencies
├── data_loader.py                     # Data loading utilities
├── *.log                              # Training logs (financial/synthetic)
│
├── Data examination/                  # Exploratory notebooks & causal graph inspection
│   ├── data_examination.ipynb
│   ├── causal_graph_model.ipynb
│   └── synthetic_data_generation*.ipynb
│
├── Experiments/                       # Training/eval for synthetic & real-data setups
│   ├── CDML/
│   │   ├── synth_CDML.ipynb
│   │   └── test_CDML.py               # CDML tests
│   │
│   ├── Finance_real_data/
│   │   ├── train_time_causal_vae.py   # TNCM-VAE training (real market data)
│   │   ├── train_temporal_vae.py      # Baseline training (CVAE/temporal VAE)
│   │   └── ctf_test_temporal_vae.py   # Counterfactual tests on finance data
│   │
│   └── Sythetic_DAG_data/
│       ├── train_time_causal_vae.py   # TNCM-VAE training (synthetic DAG)
│       ├── train_temporal_vae_synth.py# Baseline training (synthetic)
│       ├── ctf_test_temp_vae_synth.py # Counterfactual tests (TNCM-VAE)
│       ├── ctf_test_temp_vae_synth_c_vae.py  # Counterfactual tests (CVAE)
│       └── *.ipynb / *.csv            # Synthetic generation notebooks & CSVs
│
├── FinanceCPT/                        # External dataset(s) & docs (figures/relationships/returns)
│   ├── figures/                       # Provided plots (PDF/EPS)
│   ├── relationships/                 # Relationship CSVs
│   └── returns/                       # Return series CSVs
│
├── lightning_logs/                    # TensorBoard logs (by model/run)
│   ├── temporal_vae/
│   ├── c_vae/
│   └── financial_vae/
│
├── model_weights/                     # Saved checkpoints
│   ├── temporal_vae/                  # TNCM-VAE (financial & synthetic)
│   ├── c_vae/                         # Conditional VAE (financial & synthetic)
│   └── financial_vae/                 # Other VAE variants
│
├── src/                               # Installable source tree
│   └── scm/
│       ├── ncm/                       # Core models & mappings
│       │   ├── time_causal_vae.py     # TNCM-VAE (main model)
│       │   ├── temporal_vae.py        # Temporal VAE baseline components
│       │   └── causal_maps.py         # Causal/DAG utilities
│       │
│       ├── pipeline/                  # Training/evaluation pipelines
│       │   ├── vae_pipeline.py
│       │   └── c_vae_pipeline.py
│       │
│       └── prior/
│           └── realnvp.py             # Flow-based prior (RealNVP)
│
└── __pycache__/                       # Python caches (auto-generated)
```

## Main Models

### TNCM-VAE
Captures temporal and causal dependencies using GRU/LSTM encoders and DAG-based decoders.  
Enables counterfactual generation through interventions such as:

```
do(X_t = x)
```

### Conditional VAE (Baseline)
A standard conditional generative model used to compare reconstruction and counterfactual accuracy.

## Experiments

The models are trained on synthetic autoregressive (AR) data inspired by the **Ornstein–Uhlenbeck process**, allowing evaluation against analytical ground truth.  
We assess counterfactual probabilities such as:

```
P(Y_{t+1} > 0 | do(X_t = 0))
P(Y_{t+1} > 2 | do(X_t = -2))
```

The TNCM-VAE achieves **L1 distances between 0.03 and 0.10**, outperforming the Conditional VAE baseline.

## 🧪 Reproducing Experiments

### Experiment 1 — Time-series Neural Causal Model VAE (TNCM-VAE)
Evaluate the main model on the synthetic DAG dataset (no training required).

```bash
# From the repository root
# Our model
python Experiments/Sythetic_DAG_data/ctf_test_temp_vae_synth.py --ckpt model_weights/temporal_vae/synth_data_vae/last.ckpt --out results/synth_tncm_repro --thres-x 0 --thres-y 0

# Benchmark
python Experiments/Sythetic_DAG_data/ctf_test_temp_vae_synth_c_vae.py --ckpt model_weights/c_vae/synth_data_vae/last.ckpt --out results/synth_cvae_repro --thres-x 0 --thres-y 0
```

## Citation

If you use this code, please cite:

```
@conference{thumm2025towards,
    author = {Thumm, Dennis and Mijares, Luis Ontaneda},
    booktitle = {ICAIF 2025 Workshop on Rethinking Financial Time-Series},
    title = {Towards Causal Market Simulators},
    year = {2025},
    address   = {Singapore},
    url = {https://icaif-25-rfts.github.io}
}
```

---

© 2025 Dennis Thumm & Luis Ontaneda Mijares

