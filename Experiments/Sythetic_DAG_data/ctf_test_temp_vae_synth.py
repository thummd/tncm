#!/usr/bin/env python3
import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch as T
import networkx as nx

# Allow "src/" imports when called from repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.scm.ncm.temporal_vae import TemporalVAE
from src.scm.prior.realnvp import FlowPrior


def build_graph(relation_csv: str) -> nx.DiGraph:
    df = pd.read_csv(relation_csv).astype(int)
    df = df[df["source"] != df["target"]]
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row["source"], row["target"])
    return G


def load_model(ckpt_path: str, G: nx.DiGraph,
               latent_dim=16, input_dim=1, seq_length=20, batch_size=500,
               num_flows=5, hidden_dim=128, device="cpu") -> TemporalVAE:
    #prior = {
    #    node: FlowPrior(num_flows=num_flows, latent_dim=latent_dim, hidden_dim=hidden_dim)
    #    for node in range(input_dim + 1)
    #}
    prior = "normal"
    model = TemporalVAE(
        latent_dim=latent_dim,
        G=G,
        input_dim=input_dim,
        seq_length=seq_length,
        batch_size=batch_size,
        prior=prior
    )
    checkpoint = T.load(ckpt_path, map_location=device)
    sd = checkpoint.get("state_dict", checkpoint)
    new_state_dict = {}
    for k, v in sd.items():
        new_state_dict[k.replace("vae.", "")] = v
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model


def give_ctf_gt_probs(idx, seed, n_steps, thres_x, thres_y, model, n_samples, device):

    do = {"0": {idx-1: thres_x}}
    ctf_samp, _ = model.ctf_decoder(do, n_samples, n_steps, device, seed)
    ctf_prob = np.mean(ctf_samp[:, idx, 1].numpy() > thres_y)

    np.random.seed(seed)

    theta_x, sigma_x = 0.2, 0.5
    theta_y, sigma_y, beta = 0.3, 0.6, 0.5

    x_t = lambda x_prev, eta: x_prev - theta_x * x_prev + sigma_x * eta
    y_t = lambda x_prev, y_prev, eps: y_prev - theta_y * y_prev + beta * x_prev + sigma_y * eps

    x_sam_t_1 = ctf_samp[:, idx-1, 0]
    y_sam_t_1 = ctf_samp[:, idx-1, 1]
    g_t_samp = y_t(x_sam_t_1, y_sam_t_1, np.random.normal(0,1)).numpy()
    g_t_prob = np.mean(g_t_samp > thres_y)

    return g_t_prob, ctf_prob, ctf_samp


def run_eval(args):
    os.makedirs(args.out, exist_ok=True)
    G = build_graph(args.relation)

    model = load_model(
        ckpt_path=args.ckpt,
        G=G,
        latent_dim=args.latent_dim,
        input_dim=args.input_dim,
        seq_length=args.seq_len,
        batch_size=args.batch_size,
        num_flows=args.num_flows,
        hidden_dim=args.hidden_dim,
        device="cpu"
    )

    all_gt_probs, all_ctf_probs = [], []
    thres_x, thres_y = args.thres_x, args.thres_y
    n_steps = args.steps

    for trial in range(args.trials):
        gt_row, ctf_row = [], []
        for idx in range(1, n_steps):
            g_p, c_p, _ = give_ctf_gt_probs(
                idx=idx,
                seed=idx * (trial + 1),
                n_steps=idx + 1,
                thres_x=thres_x,
                thres_y=thres_y,
                model=model,
                n_samples=args.n_samples,
                device=args.device
            )
            gt_row.append(g_p)
            ctf_row.append(c_p)
        all_gt_probs.append(gt_row)
        all_ctf_probs.append(ctf_row)

    real_prob = [float(np.mean(col)) for col in zip(*all_gt_probs)]
    avg_ctf_prob = [float(np.mean(col)) for col in zip(*all_ctf_probs)]
    l1_per_step = [abs(x - y) for x, y in zip(real_prob, avg_ctf_prob)]

    print("[L1] per-step:", l1_per_step)

    metrics = []
    for s, l1 in enumerate(l1_per_step, start=1):
        q = f"P(Y_(t+1) > {thres_y} | do(X_t = {thres_x})) @ step={s}"
        metrics.append({"query": q, "L1": l1})

    out_path = os.path.join(args.out, "metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("[OK] wrote:", out_path)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate TNCM-VAE (default mode) with L1 metric.")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint (.ckpt)")
    p.add_argument("--out", default="results/synth_tncm", help="Output directory")
    p.add_argument("--relation", default="Experiments/Sythetic_DAG_data/synth_data_back_door_relation.csv")
    p.add_argument("--data", default="Experiments/Sythetic_DAG_data/synth_data_back_door.csv")
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--input-dim", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--num-flows", type=int, default=5)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    p.add_argument("--n-samples", type=int, default=10000)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--thres-x", type=float, default=0)
    p.add_argument("--thres-y", type=float, default=0)
    return p.parse_args()


if __name__ == "__main__":
    run_eval(parse_args())
