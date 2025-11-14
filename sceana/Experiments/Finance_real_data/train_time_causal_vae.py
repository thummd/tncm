import os
import sys
import gc
import psutil
import torch as T
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
import logging
from data_loader import SCMDataModule
import pandas as pd
from src.scm.prior.realnvp import FlowPrior

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('financial_vae_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
from src.scm.ncm.time_causal_vae import C_VAE
from src.scm.pipeline.c_vae_pipeline import C_VAE_Pipeline
import matplotlib.pyplot as plt
import numpy as np

def setup_callbacks(total_epochs: int = 100):
    """Setup training callbacks with improved monitoring"""
    callbacks = []
    
    # Checkpoint saving
    checkpoint_callback = ModelCheckpoint(
        dirpath='model_weights/c_vae/financial_vae',
        filename='financial_vae-{epoch:02d}-{val_loss:.2f}',
        save_top_k=5,  # Save more checkpoints
        monitor='val_loss',
        mode='min',
        save_last=True,  # Always save last model
        every_n_epochs=1  # Save every epoch
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=10,
        verbose=True,
        mode='min'
    )
    callbacks.append(early_stop_callback)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Progress bar with refresh rate
    progress_bar = TQDMProgressBar(
        refresh_rate=20,
        process_position=0
    )
    callbacks.append(progress_bar)
    
    return callbacks


def main():
    try:
        # Set random seed for reproducibility
        pl.seed_everything(42)
        logging.info("Starting VAE training")
    
        # Data paths
        data = pd.read_csv("FinanceCPT/returns/random-rels_20_1A_returns30007000.csv", header=None)
        
        # Initialize data module
        window_size = 20
        batch_size = 500
        data_module = SCMDataModule(data, window_size, batch_size)
        
        # Set up data module to access properties
        data_module.setup()
        
        # Load graph structure
        
        # Model parameters
        latent_dim = 64
        input_dim = data.shape[1]
        seq_length = window_size
        #batch_size = 32
        
        # Prior parameters
        num_flows = 5  # Number of flow layers
        hidden_dim = 128  # Hidden dimension for flow networks

        relation = pd.read_csv("FinanceCPT/relationships/random-rels_20_1A.csv", header=None)
        relation.columns = ["source", "target", "lag"]
        relation = relation[relation["source"] != relation["target"]]
        
        import networkx as nx

        G = nx.DiGraph()
        for _, row in relation.iterrows():
            G.add_edge(row['source'], row['target'])


        prior = {node: FlowPrior(num_flows=num_flows, 
                                latent_dim=latent_dim, 
                                hidden_dim=hidden_dim) for node in range(input_dim)}

        # Initialize model
        model = C_VAE(
            latent_dim=latent_dim,
            input_dim=input_dim,
            seq_length=seq_length,
            batch_size=batch_size,
            prior=prior,
            device="cpu"
        )
        
        # Training parameters
        params = {
            'beta': 0.5,  # KL divergence weight
            'gamma': 0.1,  # Optimal transport weight
        }
        
        # Initialize training pipeline
        pipeline = C_VAE_Pipeline(
            vae_model=model,
            batch_size=32,
            params=params
        )
        
        # Setup logging
        logger = TensorBoardLogger("lightning_logs", name="c_vae/financial_vae")
        
        # Get callbacks
        callbacks = setup_callbacks()
        
        # Initialize trainer with stable configuration
        trainer = pl.Trainer(
            max_epochs=1000,
            accelerator='auto',
            devices=1,
            callbacks=callbacks,
            logger=logger,
            log_every_n_steps=10,
            gradient_clip_val=1.0,  # Enable gradient clipping
            gradient_clip_algorithm="norm",
            precision=32,  # Use 32-bit precision for stability
            detect_anomaly=True,  # Enable anomaly detection
            #val_check_interval=0.25,  # Validate 4 times per epoch
            check_val_every_n_epoch=20,
            profiler="simple",  # Add profiling for performance monitoring
            accumulate_grad_batches=1,  # Default, but explicit for clarity
            deterministic=True,  # Ensure reproducibility
            benchmark=False,  # Disable cuDNN benchmarking for reproducibility
            num_sanity_val_steps=2  # Reduce validation steps
        )
        
        # Train model with memory monitoring
        try:
            # Monitor initial memory usage
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024 / 1024  # GB
            logging.info(f"Initial memory usage: {initial_memory:.2f}GB")
            
            # Start training
            trainer.fit(pipeline, data_module)
            
            # Log final memory usage
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024 / 1024  # GB
            logging.info(f"Final memory usage: {final_memory:.2f}GB")
            logging.info(f"Memory increase: {final_memory - initial_memory:.2f}GB")
            
        except Exception as e:
            logging.error(f"Training interrupted by error: {str(e)}")
            # Force garbage collection
            gc.collect()
            raise
        
        logging.info("\nTraining completed successfully!")
        logging.info(f"Model weights saved in: {callbacks[0].dirpath}")
        
    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
