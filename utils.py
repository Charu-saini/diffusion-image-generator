import torch
import torchvision.transforms as transforms
from PIL import Image
import os

def setup_folders():
    os.makedirs("outputs/graphs", exist_ok=True)
    os.makedirs("outputs/gifs", exist_ok=True)
    os.makedirs("outputs/images", exist_ok=True)

def get_noise_schedule(T=100, beta_start=1e-4, beta_end=0.02, device="cuda"):
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bar
# ------------------------
# Image Loader
# ------------------------
def get_image_tensor(path, size=64):
    """Load and normalize an image to [-1,1]"""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # [-1,1]
    ])
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)  # [1,3,H,W]

# ------------------------
# Forward Diffusion (q_sample)
# ------------------------
def q_sample(x0, t, noise=None, betas=None, alphas=None, alpha_bar=None, beta_start=1e-4, beta_end=0.02, T=100):
    """
    Forward diffusion: add noise to x0 at timestep(s) t
    x0: [B,3,H,W]
    t:  [B] timesteps
    """
    if noise is None:
        noise = torch.randn_like(x0)

    # if betas/alphas not precomputed, build them
    if betas is None or alphas is None or alpha_bar is None:
        betas = torch.linspace(beta_start, beta_end, T, device=x0.device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

    # gather alpha_bar[t] for each item in batch
    alpha_bar_t = alpha_bar[t].view(-1, 1, 1, 1)  # [B,1,1,1]

    xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
    return xt, noise
