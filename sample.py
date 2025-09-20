import torch
import matplotlib.pyplot as plt
import imageio
import numpy as np
from tqdm import tqdm

from model import TinyModelWithTime
from utils import setup_folders, get_noise_schedule

# -----------------------------------
# Config
# -----------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
T = 100
model_size = (1, 3, 64, 64)   # must match training image size

setup_folders()

# -----------------------------------
# Noise Schedule
# -----------------------------------
betas, alphas, alpha_bar = get_noise_schedule(T=T, device=device)

# -----------------------------------
# Reverse Functions
# -----------------------------------
@torch.no_grad()
def p_sample_noise(model, x, t, noise=None, betas=None, alphas=None, alpha_bar=None):
    """
    One reverse diffusion step if model predicts noise.
    Returns: (next_xt, predicted_x0)
    """
    if betas is None or alphas is None or alpha_bar is None:
        raise ValueError("Pass precomputed betas, alphas, alpha_bar")

    pred_noise = model(x, t)

    alpha_t = alphas[t].view(-1, 1, 1, 1)
    alpha_bar_t = alpha_bar[t].view(-1, 1, 1, 1)
    beta_t = betas[t].view(-1, 1, 1, 1)

    # estimate x0
    x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)

    # posterior mean
    mean = torch.sqrt(alpha_t) * x0_pred + torch.sqrt(1 - alpha_t) * pred_noise

    if noise is None:
        noise = torch.randn_like(x)

    sample = mean + torch.sqrt(beta_t) * noise
    return sample, x0_pred


@torch.no_grad()
def p_sample_x0(model, xt, t):
    """Reverse step if model predicts x0"""
    pred_x0 = model(xt, torch.tensor([t], device=device))

    alpha_t = alphas[t]
    beta_t = betas[t]

    mean = torch.sqrt(alpha_t) * pred_x0
    if t > 0:
        z = torch.randn_like(xt)
        sample = mean + torch.sqrt(beta_t) * z
    else:
        sample = pred_x0
    return sample


# -----------------------------------
# Sampling with Frames
# -----------------------------------
@torch.no_grad()
def sample_frames(model, predict_noise=True, T=100):
    """Run reverse process and return all frames for GIF"""
    xt = torch.randn(model_size, device=device)
    frames = []

    for t in tqdm(reversed(range(T)), desc=f"Sampling (predict_noise={predict_noise})"):
        if predict_noise:
            xt, _ = p_sample_noise(
                model, xt, torch.tensor([t], device=device),
                betas=betas, alphas=alphas, alpha_bar=alpha_bar
            )
        else:
            xt = p_sample_x0(model, xt, t)

        # unnormalize [-1,1] -> [0,1]
        img_disp = (xt.squeeze(0) * 0.5 + 0.5).clamp(0, 1)
        img_disp = img_disp.permute(1, 2, 0).cpu().numpy()
        frames.append(img_disp)

    return frames


# -----------------------------------
# Helper to Save Outputs
# -----------------------------------
def save_outputs(frames, name):
    clean_frames = []
    for f in frames:
        # Ensure it's a numpy array
        f = np.array(f)
        # Remove singleton dimensions
        f = np.squeeze(f)

        # If frame is [C,H,W] (3,64,64), convert to [H,W,C]
        if f.ndim == 3 and f.shape[0] == 3:
            f = f.transpose(1, 2, 0)
        # If frame is 2D, convert to 3 channels
        if f.ndim == 2:
            f = np.stack([f]*3, axis=-1)

        # Clip and convert to uint8 for GIF
        f = np.clip(f*255, 0, 255).astype(np.uint8)
        clean_frames.append(f)

    # Save final image
    final_img = clean_frames[-1]
    plt.imshow(final_img)
    plt.axis("off")
    plt.title(name)
    plt.savefig(f"outputs/images/{name}.png")
    plt.close()

    # Save GIF
    imageio.mimsave(f"outputs/gifs/{name}.gif", clean_frames, fps=10)



# -----------------------------------
# Run Both Models
# -----------------------------------
def main():
    # --- Load noise-predicting model ---
    model_noise = TinyModelWithTime(predict_noise=True).to(device)
    model_noise.load_state_dict(torch.load("outputs/model_noise.pt", map_location=device))
    model_noise.eval()

    # --- Load x0-predicting model ---
    model_x0 = TinyModelWithTime(predict_noise=False).to(device)
    model_x0.load_state_dict(torch.load("outputs/model_x0.pt", map_location=device))
    model_x0.eval()

    # --- Sampling ---
    frames_noise = sample_frames(model_noise, predict_noise=True, T=T)
    frames_x0 = sample_frames(model_x0, predict_noise=False, T=T)

    # Save individual outputs
    save_outputs(frames_noise, "generated_noise")
    save_outputs(frames_x0, "generated_x0")

    # --- Create Side-by-Side GIF ---
    # --- Create Side-by-Side GIF ---
    side_by_side = []
    for fn, fx in zip(frames_noise, frames_x0):
        if fn.shape != fx.shape:
            fx = np.resize(fx, fn.shape)

        combined = np.concatenate([fn, fx], axis=1)  # horizontal concat

        # --- Convert combined frame to uint8 ---
        combined = np.clip(combined * 255, 0, 255).astype(np.uint8)
        side_by_side.append(combined)

    imageio.mimsave("outputs/gifs/comparison.gif", side_by_side, fps=10)

    print("✅ Done. Generated images + GIFs saved in outputs/")


if __name__ == "__main__":
    main()
