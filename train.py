import torch
from torch.utils.data import DataLoader
from model import LatentMLP, GaussianDiffusion
from utils import LatentDataset

# Hyperparameters for the latent diffusion process
LATENT_DIM = 512
BATCH_SIZE = 128
LR = 1e-4
EPOCHS = 200

def train(latents, labels):
    """Initializes model and manages the main training loop."""
    dataset = LatentDataset(latents, labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model and move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LatentMLP().to(device)
    diffusion = GaussianDiffusion(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            # Sample random time-steps for each image in the batch
            t = torch.randint(0, diffusion.timesteps, (x.shape[0],)).to(device)
            
            optimizer.zero_grad()
            loss = diffusion.p_losses(x, t, y)
            loss.backward()
            optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
            torch.save(model.state_dict(), f"checkpoints/latent_diffusion_epoch_{epoch}.pt")

if __name__ == "__main__":
    # Placeholder for data loading
    # latents, labels = load_data()
    # train(latents, labels)
    pass
