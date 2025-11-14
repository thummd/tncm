import pytorch_lightning as pl
import torch as T
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import numpy as np
import pandas as pd

from src.ds.causal_graph import CausalGraph
from src.scm import Cont_dat, Cont_dat_time_series
from src.scm.ncm import VAE
from src.scm.pipeline import VAE_Pipeline
from data_loader import SCMDataModule
from src.scm.prior.realnvp import FlowPrior

import pickle
import os

cg_graph = "CDML_cg"
cg_file = "{}.cg" .format(cg_graph)
cg = CausalGraph.read(cg_file)

with open('CDML_dataset_3_variables.pkl', 'rb') as file:
    CDML_dfs = pickle.load(file)

lag_matrix = lambda x,lag: np.array([[x[j+i] for j in range(lag)] for i in range(x.shape[0]-lag+1)])

def create_lags(n_lag, var_name, vec):
    temp = lag_matrix(vec, n_lag)
    cols = [var_name + "_t_" + str(i) for i in range(n_lag-1, -1, -1)]
    temp_df = pd.DataFrame(temp, columns=cols)
    return temp_df

def build_all_datasets(datset_df):
    datset = datset_df.to_numpy()
    df_X_1_lags = create_lags(3, "X1", datset[:, 0])
    df_X_2_lags = create_lags(3, "X2", datset[:, 1])
    df_Y_1_lags = create_lags(3, "Y1", datset[:, 2])

    temp = [df_X_1_lags, df_X_2_lags, df_Y_1_lags]
    all_df = {col: i[col].values.reshape(-1, 1).astype(np.float32) for i in temp for col in i.columns}

    return all_df

all_df = [build_all_datasets(i) for i in CDML_dfs]
#all_df = [{col: i[col].values.reshape(-1, 1).astype(np.float32) for col in i.columns} for i in CDML_dfs]
#all_df = [{"Y1": i["Y1"].values.reshape(-1, 1).astype(np.float32)} for i in CDML_dfs]

T.save(all_df, 'matrix_test.pth')
with open('CDML_cg_3_variables_with_lags.pkl', 'wb') as file:
    pickle.dump(all_df, file)

tb_logger =  TensorBoardLogger(save_dir="logs/", name="my_model")

runner = Trainer(
    logger=tb_logger,  # Logger instance
    callbacks=[
        LearningRateMonitor(logging_interval='epoch'),  # Tracks the learning rate over epochs
        ModelCheckpoint(
            save_top_k=2,  # Saves the top 2 best models
            dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),  # Where to save checkpoints
            filename="{epoch}-{loss:.2f}",  # Filename format (includes epoch and val_loss)
            #monitor="val_loss",  # Metric to monitor
            monitor="loss",
            mode="min",  # 'min' because we're monitoring loss (lower is better)
            save_last=True,  # Save the last model as well
        ),
 #       early_stopping
    ],
    #terminate_on_nan=True,
    max_epochs=5000,  # Max number of epochs (example)
)


# the dataset is separated in batches
batch_size = 300
latent_dim = 50
#input_size = len(all_df[0]["X1_t_0"])
#input_size = len(all_df[0].keys())
input_size = 3
#seq_length = len(all_df[0]["Y1"])
seq_length = 20
temp_unsqueeze = input_size*seq_length

device = T.device("cuda" if T.cuda.is_available() else "cpu")

prior = T.nn.ModuleDict({i: FlowPrior(1, latent_dim, 250) for i in ["X1", "X2", "Y1"]})

vae = VAE(latent_dim, cg, input_size, seq_length, batch_size, prior, device)

model_pipeline = VAE_Pipeline(vae, batch_size)

all_df = [T.tensor(i.values, dtype=T.float32) for i in CDML_dfs]
data_module = SCMDataModule(all_df, batch_size=batch_size)
#data_module = SCMDataModule(CDML_dfs, batch_size=batch_size)

runner.fit(model_pipeline, datamodule=data_module)

log_folder = "logs/my_model"

T.save(vae.state_dict(), 'model_weights\CDML\model_weights.pth')

best_model_path = runner.checkpoint_callback.best_model_path