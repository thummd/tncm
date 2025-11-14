from torch import nn
import torch as T
import torch.nn.functional as F
import torch.nn.init as init
import pandas as pd
from collections import defaultdict

#conditional VAE

def weights_init_uniform(model, a, b):
    for name, param in model.named_parameters():
        if 'weight' in name:
            init.uniform_(param, a=a, b=b)

class C_VAE(nn.Module):
    def __init__(self,latent_dim, input_dim, seq_length, batch_size, prior, device):
        super().__init__()

        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.hidden_dim = 24
        self.activation = nn.Tanh()
        self.prior = prior
        self.device = device

        self.encoder = self.build_encoder(input_dim)

        self.fc_mu = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.activation,
            nn.Linear(self.hidden_dim, latent_dim),
        )

        self.fc_logvar = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.activation,
            nn.Linear(self.hidden_dim, latent_dim),
        )

        self.decoder = self.build_decoder()

    def build_encoder(self, input_dim):

        encoder = nn.Sequential(
            nn.Linear(input_dim, 24),
            self.activation,
            nn.LSTM(input_size=24, 
                           hidden_size=24, 
                           num_layers=3, 
                           batch_first=True),
            nn.Linear(input_dim+24, self.hidden_dim),
            self.activation,
            nn.Linear(input_dim+24+24, self.hidden_dim),
            self.activation
            )
        

        return encoder
        
    
    def encode(self, x0):

        x = self.encoder[:2](x0)
        x = self.encoder[2](x)[0]

        temp = T.cat([x, x0], dim=2)
        x = self.encoder[3:5](temp)
        y = T.cat([x, temp], dim=2)
        y = self.encoder[5:](y)

        mu = self.fc_mu(y)
        log_var = self.fc_logvar(y)

        return mu, log_var
    
    
    def build_decoder(self):

        decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 24),
            self.activation,
            nn.LSTM(input_size=24, 
                           hidden_size=24, 
                           num_layers=3, 
                           batch_first=True),
            #nn.Linear(self.latent_dim+1, 24),
            nn.Linear(24, 24),
            self.activation,
            nn.Linear(24+self.latent_dim, 24),
            self.activation,
            nn.Linear(24, self.input_dim),
        )

        return decoder
    

    def decode(self, x0):
        
        x = self.decoder[:2](x0)
        x = self.decoder[2](x)[0]
        x = self.decoder[3:5](x)

        y0 = T.cat([x, x0], dim=-1)
        y = self.decoder[5:7](y0)
        z = self.decoder[7](y)
        
        return z
        
    
    def reparameterize(self, mu, logvar):
        std = T.exp(0.5 * logvar)
        eps = T.randn_like(std)
        return mu + eps * std

    def forward(self, input):

        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        decoded_x = self.decode(z)

        return  [decoded_x, mu, log_var, z]

    def sample(self,
               n_samples,
               t_steps=1):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        
        #z = {i: self.prior[i].sample(n_samples, self.seq_length, self.latent_dim) for i in ["X1", "X2", "Y1"]}
        z = T.randn(n_samples, 20, 64)
        sampled_x = self.decode(z)

        return sampled_x
    

    def ctf_decoder(self, intervention, n_samples, n_steps, seed):

        T.manual_seed(seed)

        with T.no_grad():
            all_keys = [key for inner_dict in intervention.values() for key in inner_dict.keys()]
            if n_steps not in all_keys:
                all_keys.append(n_steps)

            decoded = defaultdict(list)

            #sampled x before counterfactual        
            z = T.randn(n_samples, n_steps, 64)
            sam_x_bef_ctf = self.decode(z)

            if decoded == None:
                decoded = defaultdict(list)

            all_keys = sorted(all_keys)
            ctf_x_sample = sam_x_bef_ctf[:,:all_keys[0]+1,:].clone()

            for dim_inter in intervention.keys():
                st_idx = all_keys[0]
                temp = intervention[dim_inter].get(st_idx)
                if temp != None:
                    ctf_x_sample[:,st_idx,int(dim_inter)] = intervention[dim_inter].get(st_idx)

            mu, log_var = self.encode(ctf_x_sample)
            sam_encod_aft = self.reparameterize(mu, log_var)
            temp_z = T.randn(n_samples, n_steps - all_keys[0]-1, 64)

            temp_new_samp = T.concat([sam_encod_aft, temp_z], dim=1)
            ctf_samp = self.decode(temp_new_samp)

            return ctf_samp, sam_x_bef_ctf

