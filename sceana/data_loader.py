import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class SlidingWindowDataset(Dataset):
    def __init__(self, df: pd.DataFrame, window_size: int):
        self.data = torch.tensor(df.values, dtype=torch.float32)
        self.window_size = window_size
        self.num_windows = len(df) - window_size + 1

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        # Return one sequential window of shape [window_size, num_features]
        return self.data[idx : idx + self.window_size]

class SCMDataModule(pl.LightningDataModule):
    def __init__(self, df: pd.DataFrame, window_size: int, batch_size: int):
        super().__init__()
        self.df = df
        self.window_size = window_size
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = SlidingWindowDataset(self.df, self.window_size)

    def train_dataloader(self, num_workers=1):
        # Keep shuffle=False to maintain time order
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=num_workers)

    def val_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)
