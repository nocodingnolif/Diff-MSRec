import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class DenoiseNetwork(nn.Module):
    def __init__(self, hidden_size):
        super(DenoiseNetwork, self).__init__()

        
        self.time_embedding = nn.Embedding(1000, hidden_size)

        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.LeakyReLU(),
            nn.Linear(hidden_size * 4, hidden_size * 4),
            nn.LeakyReLU(),
            nn.Linear(hidden_size * 4, hidden_size)  
        )

    def forward(self, x_t, h_final, t):
   
        t_emb = self.time_embedding(t)  # [Batch, Hidden]

       
        x_in = torch.cat([x_t, h_final], dim=-1)  # [Batch, Hidden*2]


        x_in_time = torch.cat([x_t + t_emb, h_final], dim=-1)

        predicted_noise = self.mlp(x_in_time)
        return predicted_noise


class DiffusionGenerator(nn.Module):
    def __init__(self, hidden_size, num_steps=100, beta_start=0.0001, beta_end=0.02):
        super(DiffusionGenerator, self).__init__()

        self.hidden_size = hidden_size
        self.num_steps = num_steps

      
        self.betas = torch.linspace(beta_start, beta_end, num_steps)

        
        self.alphas = 1.0 - self.betas

        
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

       
        self.register_buffer('beta_t', self.betas)
        self.register_buffer('alpha_t', self.alphas)
        self.register_buffer('alpha_bar_t', self.alphas_cumprod)

        
        self.denoise_net = DenoiseNetwork(hidden_size)

    def q_sample(self, x_0, t, noise=None):
        
        if noise is None:
            noise = torch.randn_like(x_0)

        
        alpha_bar = self.alpha_bar_t[t].unsqueeze(-1)

        return torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise

    def calc_loss(self, target_item_emb, h_final):
     
        batch_size = target_item_emb.size(0)

       
        t = torch.randint(0, self.num_steps, (batch_size,), device=target_item_emb.device)

        
        noise = torch.randn_like(target_item_emb)

        
        x_t = self.q_sample(target_item_emb, t, noise)

       
        predicted_noise = self.denoise_net(x_t, h_final, t)

      
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def generate(self, h_final):
      
        batch_size = h_final.size(0)

        
        x_t = torch.randn(batch_size, self.hidden_size, device=h_final.device)

        
        for t in reversed(range(self.num_steps)):
            t_batch = torch.full((batch_size,), t, device=h_final.device, dtype=torch.long)

            
            predicted_noise = self.denoise_net(x_t, h_final, t_batch)

            
            alpha = self.alpha_t[t]
            alpha_bar = self.alpha_bar_t[t]
            beta = self.beta_t[t]

           
            noise_factor = (1 - alpha) / (torch.sqrt(1 - alpha_bar) + 1e-9)
            mean = (1 / torch.sqrt(alpha)) * (x_t - noise_factor * predicted_noise)

            
            if t > 0:
                sigma = torch.sqrt(beta)
                z = torch.randn_like(x_t)
                x_t = mean + sigma * z
            else:
                x_t = mean

        return x_t  


