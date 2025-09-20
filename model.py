import torch
import torch.nn as nn
import math


class TinyModel(nn.Module):
    """
    Simple CNN that maps noisy image -> predicted noise or clean image.
    """
    def __init__(self, out_channels=3, predict_noise=True):
        super().__init__()
        self.predict_noise = predict_noise  # if False, predicts x0 instead of noise

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, 3, padding=1)
        )

    def forward(self, x, t=None):
        # right now we ignore timestep (basic version)
        return self.net(x)


# -----------------------------
# Sinusoidal timestep embeddings
# -----------------------------
def sinusoidal_embedding(timesteps, dim=32):
    """
    Create sinusoidal timestep embeddings (like in Transformers/UNet DDPM).
    """
    half_dim = dim // 2
    device = timesteps.device
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb


class TinyModelWithTime(nn.Module):
    """
    CNN + timestep conditioning using sinusoidal embeddings.
    """
    def __init__(self, out_channels=3, time_dim=32, predict_noise=True):
        super().__init__()
        self.predict_noise = predict_noise

        # project time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )

        # conv net
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, out_channels, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        # compute time embeddings
        t_emb = sinusoidal_embedding(t, dim=32)
        t_emb = self.time_mlp(t_emb)  # [batch, 32]

        # reshape to add to conv activations
        t_emb = t_emb[:, :, None, None]  # [batch, 32, 1, 1]

        # conv + add time conditioning
        h = self.relu(self.conv1(x) + t_emb)
        h = self.relu(self.conv2(h))
        out = self.conv3(h)
        return out
