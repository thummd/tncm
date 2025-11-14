import torch as T
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from src.scm.ncm.causal_maps import TemporalCausalMaps
from collections import defaultdict

def _init_weights(NN):
    """Initialize network weights"""
    for m in NN.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.zeros_(m.bias.data)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(param.data)
                elif 'bias' in name:
                    nn.init.zeros_(param.data)
                        
class TemporalVAE(nn.Module):
    """
    Temporal VAE implementation for time series data.
    Extends base VAE with temporal components and causal maps.
    """
    def __init__(self,
                 latent_dim: int,
                 G,
                 input_dim: int,
                 seq_length: int,
                 batch_size: int,
                 prior,
                 device: str = 'cuda' if T.cuda.is_available() else 'cpu'):
        # Call parent class's __init__
        super().__init__()
        
        # Store parameters
        self.latent_dim = latent_dim
        self.seq_length = seq_length
        self.hidden_dim = 24
        self.batch_size = batch_size
        self.prior = prior
        self.device = device
        self.activation = nn.LeakyReLU()
        self.G = G
        
        # Create ModuleDict components for each node

        self.encoder = nn.ModuleDict()
        self.fc_mu = nn.ModuleDict()
        self.fc_logvar = nn.ModuleDict()
        
        for node in self.G.nodes():
            n_par = len(list(G.predecessors(node)))

            self.encoder[str(node)] = self.build_encoder(input_dim+n_par)
                       
            self.fc_mu[str(node)] = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                self.activation,
                nn.Linear(self.hidden_dim, latent_dim//2),
            )

            self.fc_logvar[str(node)] = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                self.activation,
                nn.Linear(self.hidden_dim, latent_dim//2),
            )
        
        # Temporal components
        self.temporal_maps = TemporalCausalMaps(
            G,
            seq_length = seq_length,
            latent_dim=latent_dim,
            hidden_dim=latent_dim,
            time_window=self.seq_length,
            prior = self.prior
        )
        
        # Initialize weights
        _init_weights(self)

    def build_encoder(self, input_dim):

        encoder = nn.Sequential(
            nn.Linear(input_dim, 24),
            self.activation,
            nn.GRU(input_size=24,
                           hidden_size=24, 
                           num_layers=3, 
                           batch_first=True),
            nn.Linear(24+24, self.hidden_dim),
            self.activation,
            nn.Linear(24+24+24, self.hidden_dim),
            self.activation)

        return encoder

    def encode(self, x0, key):
        
        key = str(key)
        temp = self.encoder[key][0:2](x0)
        x, _ = self.encoder[key][2](temp)
        temp = T.cat([x, temp], dim=2)
        x = self.encoder[key][3:5](temp)
        x = T.cat([x, temp], dim=2)
        x = self.encoder[key][5:](x)

        mu = self.fc_mu[key](x)
        log_var = self.fc_logvar[key](x)

        return mu, log_var
    
    
    def reparameterize(self, mu: T.Tensor, log_var: T.Tensor) -> T.Tensor:
        """
        Reparameterization trick for temporal data
        
        Args:
            mu: Mean tensor [batch_size, time_steps, latent_dim]
            log_var: Log variance tensor [batch_size, time_steps, latent_dim]
        Returns:
            Sampled latent tensor [batch_size, time_steps, latent_dim]
        """
        std = T.exp(0.5 * log_var)
        eps = T.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: Dict[str, T.Tensor]) -> Tuple[Dict[str, T.Tensor], Dict[str, T.Tensor], Dict[str, T.Tensor], Dict[str, T.Tensor]]:
        """
        Forward pass through the temporal VAE with causal maps
        """
        # Encode input
        z = defaultdict()
        mu = defaultdict()
        log_var = defaultdict()

        for node in self.G.nodes:
            temp = list(self.G.predecessors(node))
            temp = temp + [node]            
            mu[node], log_var[node] = self.encode(x[:,:, temp], str(node))
            # Sample latent
            z[node] = self.reparameterize(mu[node], log_var[node])
        
        # Apply temporal causal maps
        X_tilde = self.temporal_maps.decode(z, end_wind=self.seq_length, n_samples=x.shape[0])

        return X_tilde, mu, log_var, z
    
    def sample(self, n_samples: int, n_steps: int, seed, device) -> Dict[str, T.Tensor]:
        """
        Generate samples from the temporal VAE
        
        Args:
            n_samples: Number of samples to generate
        Returns:
            Dictionary of generated samples for each node
        """
        with T.no_grad():
            #z = {key: T.randn(n_samples,
            #            n_steps, 2) for key in self.G.nodes}
            gen = T.Generator(device=device).manual_seed(seed)

            if self.prior == "normal":
                z = {node: T.randn((n_samples, n_steps, self.latent_dim//2), device=device, generator=gen)
                      for node in self.G.nodes}
                                
            else:
                z = {node: self.prior[node].sample(n_samples, n_steps, self.device, gen)
                            for node in self.G.nodes}
            
            sampled_x = self.temporal_maps.decode(z, end_wind=n_steps, n_samples=n_samples)
        
        return sampled_x
    
    def ctf_decoder(self,
                        intervention: Dict[str, T.Tensor],
                        n_samples: int = 1,
                        n_steps: int = 20,
                        device="cpu",
                        seed=0) -> Dict[str, T.Tensor]:
        """
        Generate counterfactual samples
        
        Args:
            x: Dictionary of input tensors for each node
            intervention: Dictionary of interventions per node
            n_samples: Number of counterfactual samples
        Returns:
            Dictionary of counterfactual samples for each node
        """
        # Encode input

        with T.no_grad():
            
            ctf_x = self.temporal_maps.sample_counterfactual(intervention, n_samples, n_steps, self, device, seed)

            return ctf_x

        