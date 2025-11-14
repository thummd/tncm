#!/usr/bin/env python3
import json
import pandas as pd
from pathlib import Path

def main():
    # default paths
    tncm_path = Path("results/synth_tncm_repro/metrics.json")
    cvae_path = Path("results/synth_cvae_repro/metrics.json")
    out_path  = Path("results/compare_tncm_vs_cvae.csv")

    # check existence
    if not tncm_path.exists() or not cvae_path.exists():
        print("[ERROR] Missing metrics files.")
        print("Expected:")
        print(f" - {tncm_path}")
        print(f" - {cvae_path}")
        return

    # load metrics
    tncm = json.load(open(tncm_path))
    cvae = json.load(open(cvae_path))

    # build comparison table
    rows = []
    for t, c in zip(tncm, cvae):
        rows.append({
            "query": t["query"],
            "TNCM_L1": t["L1"],
            "CVAE_L1": c["L1"],
            "ΔL1 (CVAE−TNCM)": c["L1"] - t["L1"]
        })

    df = pd.DataFrame(rows)
    df["Better Model"] = df.apply(lambda r: "TNCM-VAE" if r["TNCM_L1"] < r["CVAE_L1"] else "CVAE", axis=1)

    avg_tncm = df["TNCM_L1"].mean()
    avg_cvae = df["CVAE_L1"].mean()

    # print summary
    print("\n=== Counterfactual Comparison ===")
    print(df[["query", "TNCM_L1", "CVAE_L1", "Better Model"]])
    print("\nAverage L1:")
    print(f" - TNCM-VAE: {avg_tncm:.4f}")
    print(f" - CVAE:     {avg_cvae:.4f}")

    # save results
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n[OK] Saved comparison table to: {out_path}")

if __name__ == "__main__":
    main()
