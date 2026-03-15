import torch
from torch.utils.data import Dataset
import numpy as np

class LatentDataset(Dataset):
    """Custom Dataset for StyleGAN2 W-space latent vectors and age labels."""
    def __init__(self, latents, labels):
        self.latents = torch.tensor(latents, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx], self.labels[idx]

def visualize_latent(G, w_vector, device='cuda'):
    """
    Synthesizes an image from a latent vector using the StyleGAN2 generator.
    Expects w_vector in W-space (1x18x512).
    """
    if w_vector.ndim == 1:
        w_vector = w_vector.unsqueeze(0).unsqueeze(0).repeat(1, 18, 1)
    
    with torch.no_grad():
        img = G.synthesis(w_vector, noise_mode='const')
        # Map pixel values from [-1, 1] to [0, 255]
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return img[0].cpu().numpy()
