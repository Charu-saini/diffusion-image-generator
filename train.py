import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader   
from glob import glob  

from model import TinyModelWithTime
from utils import setup_folders, q_sample, get_image_tensor

# -----------------------------------
# Config
# -----------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
T = 100
epochs = 500
lr = 1e-3
img_path = "your_image.png"   # <-- replace with your image path

setup_folders()

os.makedirs("outputs/checkpoints", exist_ok=True)
# -----------------------------------
# Dataset Loader
# -----------------------------------
class ImageDataset(Dataset):
    def __init__(self, folder="data/", size=64):
        self.files = glob(f"{folder}/*.png") + glob(f"{folder}/*.jpg") + glob(f"{folder}/*.jpeg")
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return get_image_tensor(self.files[idx], size=self.size).squeeze(0)  # [3,H,W]


# Create dataset + dataloader
dataset = ImageDataset("data_images/", size=64)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# -----------------------------------
# Noise Schedule
# -----------------------------------
from utils import get_noise_schedule
betas, alphas, alpha_bar = get_noise_schedule(T=T, device=device)



# -----------------------------------
# Training Function
# -----------------------------------
def train_model(predict_noise=True, save_name="model_noise.pt", save_interval=50):
    model = TinyModelWithTime(predict_noise=predict_noise).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    losses = []
    
    checkpoint_path = "outputs/checkpoints/model_noise_epoch100.pt"  # example
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f"🔄 Resumed training from epoch {start_epoch}")
    else:
        start_epoch = 0

    
    for epoch in tqdm(range(start_epoch, epochs), desc=f"Training ({save_name})"):

        for x0 in dataloader:
            x0 = x0.to(device)  # [B,3,H,W]

            # random timestep per batch
            t = torch.randint(0, T, (x0.size(0),), device=device)

            # forward diffusion
            xt, noise = q_sample(x0, t, betas=betas, alphas=alphas, alpha_bar=alpha_bar)

            # training target
            target = noise if predict_noise else x0

            # forward pass
            pred = model(xt, t)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # 🔹 Save checkpoint every `save_interval` epochs
        if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
            ckpt_name = save_name.replace(".pt", f"_epoch{epoch+1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "losses": losses,
            }, f"outputs/checkpoints/{ckpt_name}")
            print(f"💾 Saved checkpoint: {ckpt_name}")

    # save final model
    torch.save(model.state_dict(), f"outputs/{save_name}")

    # save loss curve
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"Training Loss ({'Noise' if predict_noise else 'X0'} Prediction)")
    plt.savefig(f"outputs/graphs/{save_name.replace('.pt', '')}_loss.png")
    plt.close()

    print(f"✅ Finished training {save_name}. Saved model + loss curve.")
    return model


# -----------------------------------
# Train Both Models
# -----------------------------------
if __name__ == "__main__":
    train_model(predict_noise=True, save_name="model_noise.pt")
    train_model(predict_noise=False, save_name="model_x0.pt")
