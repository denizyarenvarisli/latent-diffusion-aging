import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentMLP(nn.Module):
    """
    MLP architecture to denoise latent vectors (512-dim) in StyleGAN2's W-space.
    Integrates time-step and class-label embeddings.
    """
    def __init__(self, input_dim=512, hidden_dim=1024, num_layers=4, num_classes=2):
        super().__init__()
        # Embedding for the diffusion time-step
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Embedding for age categories (e.g., Young/Old)
        self.label_emb = nn.Embedding(num_classes, hidden_dim)
        
        # Main MLP body for latent traversal
        layers = []
        layers.append(nn.Linear(input_dim + hidden_dim, hidden_dim))
        layers.append(nn.GELU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.main = nn.Sequential(*layers)

    def forward(self, x, t, labels):
        # Generate combined embedding from time and label
        t_emb = self.time_mlp(t.unsqueeze(-1).float())
        l_emb = self.label_emb(labels)
        h = t_emb + l_emb
        
        # Concatenate noisy latent with conditioning embedding
        x_input = torch.cat([x, h], dim=-1)
        return self.main(x_input)

class GaussianDiffusion:
    """
    Manages the forward and reverse diffusion processes.
    Defines the variance schedule and noise prediction loss.
    """
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.model = model
        self.timesteps = timesteps
        # Linear variance schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion: Adds noise to the latent vector at step t."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod[t]).view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1. - self.alphas_cumprod[t]).view(-1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, t, labels, noise=None):
        """Calculates MSE loss between actual and predicted noise."""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.model(x_noisy, t, labels)
        
        return F.mse_loss(noise, predicted_noise)
