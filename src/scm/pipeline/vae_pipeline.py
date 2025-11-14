import pytorch_lightning as pl
import torch as T
from torch import optim
from torch.autograd import grad as torch_grad
from torch.nn import functional as F
from numpy import pi
from typing import Dict
from src.scm.prior.realnvp import FlowPrior

def entropy_normal(log_var, eps=1e-8, max_value=100):
    """Compute entropy of diagonal Gaussian with given log variance"""
    # Add small epsilon and clamp for numerical stability
    entropy = -0.5 * (1 + T.log(2.0 * T.tensor(T.pi)) + log_var)
    entropy = entropy.sum(dim=2)  # Average over time steps
    return entropy

class VAE_Pipeline(pl.LightningModule):

    def __init__(self,
                 vae_model,
                 batch_size,
                 params={}) -> None:
        super(VAE_Pipeline, self).__init__()

        self.vae = vae_model
        self.prior = vae_model.prior
        if type(self.prior) == str:
            self.prior_name = "normal"
        else:
            self.prior_name = "no normal"
        #self.beta = params.get('beta', 0.5)  # Adjustable beta parameter
        #self.beta = params.get('beta', .5)
        self.beta = .5
        self.gamma = params.get('gamma', 0.1)  # Weight for optimal transport loss
        self.params = params
        self.batch_size = batch_size
        self.curr_device = None
        self.hold_graph = False
        self.automatic_optimization = True
        
        # Temporal specific parameters
        self.is_temporal = hasattr(vae_model, 'temporal_maps')
        self.samples = []

    def forward(self, input, **kwargs):
        return self.vae(input, **kwargs)
    
    def ctf_decoder(self, do_list, n_samples):
        return self.vae.ctf_decoder(do_list, n_samples)
    
    def sample(self, n_samples, n_steps):
        return self.vae.sample(n_samples, n_steps)
    
    def ctf_sampler(self, do_list, n_samples):
        return self.vae.ctf_sampler(do_list, n_samples)

    def compute_loss(self, X):
        
        X_tilde, mu, log_var, z = self.vae(X)        
        recon_loss = T.square(X_tilde - X).mean(dim=0).sum(dim=0).sum()

        if self.prior_name == "normal":
            kld_loss = 0
            for node in self.vae.G.nodes:
                kld_loss += -0.5 * T.sum((1 + log_var[node] - mu[node].pow(2) - log_var[node].exp()).mean(dim=0).mean(dim=0))
            #kld_loss *= .5

        else:            
            # Compute KL divergence with stability        
            posterior_term = {i: entropy_normal(log_var[i]) for i in self.vae.G.nodes}
            prior_term = {i: self.prior[i].log_prob(z[i]) for i in self.vae.G.nodes}
            kld_loss = sum((posterior_term[i] - prior_term[i]).mean(dim=0) for i in self.vae.G.nodes).sum()

        # ctf part

        #do = {"0": {1: thres_x}}
        #self.ctf_sampler()
        
        loss = recon_loss + self.beta * kld_loss
        
        # Final stability check
        if T.isnan(loss) or T.isinf(loss):
            raise ValueError("Loss computation resulted in NaN or Inf values")
        
        #return loss
        return loss, {
            'loss': loss.item(),
            'recon_loss': recon_loss.item(),
            'kld_loss': kld_loss.item()
       }
            
    

    def training_step(self, batch, batch_idx):
        # Compute loss
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
        """Configure optimizer with gradient clipping"""
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
