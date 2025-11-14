import os
import sys
import gc
import psutil
import torch as T
import pytorch_lightning as pl
import pandas as pd
from src.scm.prior.realnvp import FlowPrior

from src.scm.ncm.temporal_vae import TemporalVAE
from src.scm.pipeline.vae_pipeline import VAE_Pipeline
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Set random seed for reproducibility
    pl.seed_everything(42)

    # Data paths
    import pandas as pd

    relation = pd.read_csv("FinanceCPT/relationships/random-rels_20_1A.csv", header=None)
    relation.columns = ["source", "target", "lag"]
    relation = relation[relation["source"] != relation["target"]]

    import networkx as nx

    G = nx.DiGraph()
    for _, row in relation.iterrows():
        G.add_edge(row['source'], row['target'])

    from src.scm.ncm.temporal_vae import TemporalVAE
    from src.scm.prior.realnvp import FlowPrior

    latent_dim = 64
    input_dim = 25
    seq_length = 5000
    batch_size = 300

    num_flows = 5 
    hidden_dim = 128 

    prior = {node: FlowPrior(num_flows=num_flows, 
                                    latent_dim=latent_dim, 
                                    hidden_dim=hidden_dim) for node in range(input_dim)}

    # Initialize model
    model = TemporalVAE(
        latent_dim=latent_dim,
        G = G,
        input_dim=input_dim,
        seq_length=seq_length,
        batch_size=batch_size,
        prior=prior
    )
        

    import torch as T

    ckpt_path = "model_weights/temporal_vae/financial_vae/financial_vae-epoch=999-val_loss=0.20.ckpt"
    checkpoint = T.load(ckpt_path, map_location="cpu")
    new_state_dict = {}
    for key in checkpoint['state_dict'].keys():
        new_key = key.replace("vae.", "")
        new_state_dict[new_key] = checkpoint['state_dict'][key]

    # Then replace the old state_dict with the new one
    checkpoint['state_dict'] = new_state_dict

    model.load_state_dict(checkpoint['state_dict'])

    #ctf_samp = model.temporal_maps.ctf_decoder(z)
    # intervention in feature 4, time step 10, value 1
    do = {"4": {10: 1},
          "7": {5: 1}}
    n_samples = 100
    n_steps = 20

    ctf_samp = model.ctf_decoder(do, n_samples, n_steps)

    g = 5

       
if __name__ == "__main__":
    main()