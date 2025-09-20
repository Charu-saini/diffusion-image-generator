# 🌀 Tiny Diffusion Model (PyTorch)

This project is a **minimal, educational implementation of a Diffusion Model** built from scratch in PyTorch.  
It includes **training, sampling, and checkpointing**, and demonstrates how forward and reverse diffusion work.

---

## 📌 Features
- Train a **Tiny UNet-like model** for image generation
- Supports both **noise prediction** and **x₀ prediction**
- Forward diffusion (`q_sample`) and reverse diffusion (`p_sample`)
- **Checkpointing**: save & resume training
- Loss curve plotting for monitoring
- Batch image sampling from a trained model

---

## 📂 Project Structure
.
├── train.py # Training loop with checkpointing

├── sample.py # Reverse diffusion sampling

├── model.py # Tiny model with timestep conditioning

├── utils.py # Helper functions (q_sample, noise schedule, image loading)

├── data_images/ # Training images

└── outputs/ # Models, graphs, checkpoints


## ⚙️ Installation

git clone https://github.com/your-username/tiny-diffusion.git
cd tiny-diffusion
pip install -r requirements.txt
🚀 Training
Place your training images in the data_images/ folder, then run:

python train.py
This will train two models:

model_noise.pt → predicts noise

model_x0.pt → predicts original image

Checkpoints are saved in:

outputs/checkpoints/
Loss curves are saved in:

outputs/graphs/
🎨 Sampling
After training, generate new images with:

python sample.py
Samples are saved inside:

outputs/samples/
🔄 Resuming Training
To resume from a checkpoint, set the checkpoint path in train.py:

checkpoint_path = "outputs/checkpoints/model_noise_epoch100.pt"
📈 Example Output
(Loss curves and generated images can be added here)

🛠️ Requirements
Python 3.8+

PyTorch

Matplotlib

tqdm

Pillow

Install all with:

pip install -r requirements.txt

