import pytorch_lightning as pl
import torch as T
from torch import optim
from torch.autograd import grad as torch_grad
from torch.nn import functional as F
from numpy import pi
from src.scm.prior.realnvp import FlowPrior
import numpy as np

def entropy_normal(log_var):
    entropy = -0.5 * (1 + T.log(2.0 * T.tensor(pi)) + log_var)
    entropy = T.sum(entropy, dim=-1)
    return entropy

class C_VAE_Pipeline(pl.LightningModule):

    def __init__(self,
                 vae_model,
                 batch_size,
                 params={}) -> None:
        super(C_VAE_Pipeline, self).__init__()

        self.vae = vae_model
        self.prior = vae_model.prior
        self.beta = .5
        self.params = params
        self.batch_size = batch_size

        self.samples = []


    def forward(self, input, **kwargs):
        return self.vae(input, **kwargs)
    
    def ctf_decoder(self, do_list, n_samples):
        return self.vae.ctf_decoder(do_list, n_samples)
    
    def sample(self, n_samples):
        with T.no_grad():
            return self.vae.sample(n_samples)
     
    def compute_loss(self, X):
        
        X_tilde, mu, log_var, z = self.vae(X)

        recon_loss = T.square(X_tilde - X).mean(dim=0).mean(dim=0).sum()
        kld_loss = -0.5 * T.sum((1 + log_var - mu.pow(2) - log_var.exp()).mean(dim=0).mean(dim=0))

        loss = recon_loss + self.beta * kld_loss
        
        return loss, {
            'loss': loss.item(),
            'recon_loss': recon_loss.item(),
            'kld_loss': kld_loss.item()
        #    ,'transport_loss': transport_loss.item() if self.is_temporal else 0.0
       }

    def training_step(self, batch, batch_idx):
    
        # tensor in batch is "X1", "X2", "Y1"
        # first column is X1 second X2 and third Y1
        loss, losses = self.compute_loss(batch)
        
        # Log all loss components
        for name, value in losses.items():
            self.log(name, value/self.batch_size, 
                    on_epoch=True, prog_bar=True, logger=True)
        
        # Monitor gradient norms
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm()
                self.log(f'grad_norm/{name}', grad_norm, 
                        on_epoch=True, logger=True)
                
                # Alert if gradients are getting too large
                if grad_norm > 10.0:
                    self.log(f'grad_alert/{name}', 1.0, on_epoch=True)
        
        # Print losses periodically
        if (self.current_epoch+1)%15==0:
            print(f"Epoch {self.current_epoch+1}")
            for name, value in losses.items():
                print(f"{name}: {value/self.batch_size:.4f}")
        
        return loss
      
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        # Compute loss without gradients
        with T.no_grad():
            _, losses = self.compute_loss(batch)
        
        for name, value in losses.items():
            self.log(f'val_{name}', value/self.batch_size, 
                    on_epoch=True, prog_bar=True)
        
        return losses['loss']/self.batch_size
    

    def configure_optimizers(self):

        lr = 0.0002
        beta_1 = 0.5
        beta_2 = 0.9

        # Combine all parameters
        params = list(self.vae.parameters())
        
        # Create optimizer with gradient clipping
        optimizer = optim.Adam(params, lr=lr, betas=[beta_1, beta_2])

        return {
            "optimizer": optimizer,
            "gradient_clip_val": 1.0,
            "gradient_clip_algorithm": "norm"
        }
 
