import torch as T
import torch.nn as nn
from typing import Dict, List, Tuple
from collections import defaultdict

class CausalMap(nn.Module):
    """
    Implements causal maps for optimal transport in time series VAE.
    Based on TC-VAE paper: https://arxiv.org/abs/2411.02947
    """
    def __init__(self, n_par, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Inverse map (decoding)
        self.inverse_map = nn.Sequential(
            nn.Linear(24+n_par, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def inverse(self, z: T.Tensor) -> T.Tensor:
        """Apply inverse causal map"""
        return self.inverse_map(z)

class TemporalCausalMaps(nn.Module):
    """
    Implements temporal causal maps for time series counterfactuals.
    Handles both temporal and causal dependencies.
    """
    def __init__(self, 
                 G,
                 seq_length,
                 latent_dim: int,
                 hidden_dim: int = 64,
                 time_window: int = 5,
                 device: str = 'cuda' if T.cuda.is_available() else 'cpu',
                 prior: str = "normal"):
        super().__init__()
        self.num_nodes = len(G.nodes)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.time_window = time_window
        self.G = G
        self.seq_length = seq_length
        all_nums = set(range(self.num_nodes))  # 0 to 25
        self.non_caus_n = sorted(all_nums - set(self.G.nodes))
        self.activation = nn.LeakyReLU()
        self.device = device
        self.prior = prior
        
        # Create causal maps for each variable

        self.causal_maps = nn.ModuleDict()
        for node in self.G.nodes():
            n_par = len(list(G.predecessors(node)))
            #self.causal_maps[str(node)] = CausalMap(n_par, latent_dim, hidden_dim)
            self.causal_maps[str(node)] = CausalMap(0, latent_dim, hidden_dim)

        for node in self.non_caus_n:
            self.causal_maps[str(node)] = CausalMap(0, latent_dim, hidden_dim)
        
        # Set attention dimensions

        input_dim = 1
        self.emb_dim = 8
        self.cond_net_node = nn.ModuleDict({
                                str(key): nn.Sequential(
                                    nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, self.emb_dim)
                                ) for key in self.G.nodes
                            })             
        
        self.input_proj = nn.ModuleDict()
        for node in self.G.nodes():
            n_par = len(list(G.predecessors(node)))
            self.input_proj[str(node)] = nn.Linear((hidden_dim//2)+n_par*self.emb_dim+self.emb_dim, 24)
        
        for node in self.non_caus_n:
            self.input_proj[str(node)] = nn.Linear((hidden_dim//2), 24)

        # Additional projection for GRU input
        self.gru_proj = nn.Linear(24, 24)
                
        self.temporal_gru = nn.ModuleDict({str(i): nn.GRU(
                                                    input_size=24,
                                                    hidden_size=24,
                                                    num_layers=2,
                                                    batch_first=True
                                                ) for i in all_nums})
        
    
    def decode_pipeline(self, z, node, decoded, idx=None):
        
        if idx == None:
            z_temp = z
        else:
            z_temp = z[:,:idx,:]

        z_proj = self.input_proj[str(node)](z_temp)
        z_proj = self.activation(z_proj)
        z_gru = self.gru_proj(z_proj)
        z_temporal, _ = self.temporal_gru[str(node)](z_gru)
        z_decoded = self.causal_maps[str(node)].inverse(z_temporal)
        decoded[node] = z_decoded
    
    def decode(self, z: Dict[str, T.Tensor],
               decoded = None,
               st_idx = 0,
               ret_tensor = True,
               end_wind = 0,
               n_samples = 0) -> Dict[str, T.Tensor]:
        """
        Apply inverse causal maps with temporal dependencies
        
        Args:
            z: Dictionary mapping variable names to latent tensors
               Shape: [batch_size, seq_len, latent_dim]
        """
        max_in_degree = max(dict(self.G.in_degree()).values())
        
        if decoded == None:
            decoded = defaultdict(list)

        real_decoded = {key: T.empty((n_samples, 0, 1)) for key in self.G.nodes()}
                
        for idx in range(st_idx,end_wind):
            copy_nodes = list(self.G.nodes())
            for deg in range(max_in_degree+1):
                to_drop = []
                for node in copy_nodes:
                    if self.G.in_degree(node) == deg:
                        to_drop.append(node)
                        z_mod = z[node][:,:idx+1]
                        if idx != 0:
                            # and later access with str(node)
                            temp_emb = self.cond_net_node[str(node)](decoded[node])     
                            #decoded[node] = T.concat([temp_emb.detach(),T.zeros(n_samples, 1, self.emb_dim, device=self.device)], dim=1)
                            decoded[node] = T.concat([temp_emb,T.zeros(n_samples, 1, self.emb_dim, device=self.device)], dim=1)
                            z_mod = T.concat([z_mod, decoded[node]], dim=2)
                        else:
                            z_mod = T.concat([z_mod, T.zeros(n_samples, z_mod.shape[1], self.emb_dim, device=self.device)], dim=2)
                        
                        if 0 != deg:
                            for pred in self.G.predecessors(node):
                                # 1 is the number of lags, we can change that later
                                if idx != 0:
                                    temp_emb = self.cond_net_node[str(pred)](decoded[pred])   
                                    #z_mod = T.concat([z_mod, temp_emb.detach()], dim=2)
                                    z_mod = T.concat([z_mod, temp_emb], dim=2)
                                else:
                                    z_mod = T.cat([z_mod, T.zeros(n_samples, 1, self.emb_dim, device=self.device)], dim=2)

                        self.decode_pipeline(z_mod, node, decoded)
                        real_decoded[node] = T.concat([real_decoded[node],decoded[node][:,idx:idx+1]], dim=1)
                        
                for d in to_drop:
                    copy_nodes.remove(d)

            # to do: change this so it follows the same proccess as the causal one
            # x_t gets appended to Z
            for node in self.non_caus_n:
                self.decode_pipeline(z, node, decoded)

        if ret_tensor:
            tensor = T.empty((real_decoded[0].shape[0],real_decoded[0].shape[1], self.num_nodes), dtype=T.float32, device=self.device)
            for node in range(self.num_nodes):
                tensor[:, :, node] = real_decoded[node].squeeze(-1)
            
            return tensor
    
    def sample_counterfactual(self,
                            intervention: Dict[str, T.Tensor],
                            n_samples: int,
                            n_steps: int,
                            temp_vae,
                            device,
                            seed) -> Dict[str, T.Tensor]:
                
        all_keys = [key for inner_dict in intervention.values() for key in inner_dict.keys()]
        if n_steps not in all_keys:
            all_keys.append(n_steps)

        gen = T.Generator(device=device).manual_seed(seed)

        node_list = sorted(self.G.nodes)  # see point (2)

        if self.prior == "normal":
            z = {node: T.randn((n_samples, n_steps, self.latent_dim//2), device=device, generator=gen)
                    for node in self.G.nodes}
        else:       
            z = {
                node: temp_vae.prior[node].sample(
                    n_samples, n_steps, device, generator=gen
                )
                for node in node_list
            }
            
        decoded = defaultdict(list)

        #sampled x before counterfactual
        sam_x_bef_ctf = temp_vae.temporal_maps.decode(z, decoded, end_wind=n_steps, n_samples=n_samples)
        
        if decoded == None:
            decoded = defaultdict(list)

        all_keys = sorted(all_keys)

        ctf_x_sample = sam_x_bef_ctf[:,:all_keys[0]+1,:].clone()

        for dim_inter in intervention.keys():
                st_idx = all_keys[0]
                temp = intervention[dim_inter].get(st_idx)
                if temp != None:
                    ctf_x_sample[:,st_idx,int(dim_inter)] = intervention[dim_inter].get(st_idx)
            

        while len(all_keys) > 1:
            st_idx = all_keys.pop(0)
            decoded = {key: val[:,:st_idx+1] for key, val in decoded.items()}

            for dim_inter in intervention.keys():
                temp = intervention[dim_inter].get(st_idx)
                if temp != None:
                    decoded[int(dim_inter)][:,st_idx] = temp

            temp = self.decode(z, decoded, st_idx+1, True, all_keys[0], n_samples=n_samples)
            ctf_x_sample = T.concat([ctf_x_sample,temp], dim=1)
                

        return ctf_x_sample, sam_x_bef_ctf
        