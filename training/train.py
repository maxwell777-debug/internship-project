from pathlib import Path
import torch
import torch.optim as optim
from torch.nn import BCELoss, L1Loss
from torch.utils.data import DataLoader
from loguru import logger

from datasets.pet_gan_dataset import CtPetGanPatchDataset
from models.discriminator import Discriminator3D
from models.generator import Generator3D

# === Config ===
data_root = Path("data/processed")
use_ct = True
patch_size = (128, 128, 128)
batch_size = 2
num_epochs = 100
lr = 2e-4


# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    logger.warning("⚠️ CUDA not available — using CPU")

# === Dataset & DataLoader ===
logger.info("Loading dataset...")
dataset = CtPetGanPatchDataset(data_root, use_ct=use_ct, patch_size=patch_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
logger.info(f"Loaded {len(dataset)} patients.")

# === Models ===
in_channels_G = 2 if use_ct else 1
in_channels_D = in_channels_G + 1

generator = Generator3D(in_channels=in_channels_G).to(device)
discriminator = Discriminator3D(in_channels=in_channels_D).to(device)

# === Optimizers ===
opt_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
opt_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# === Losses ===
bce_loss = BCELoss()
l1_loss = L1Loss()

# === Training loop ===
logger.info("Starting training loop...")
for epoch in range(num_epochs):
    for i, (input_tensor, target_tensor) in enumerate(dataloader):
        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)

        # === Train Discriminator ===
        fake = generator(input_tensor).detach()
        real_pred = discriminator(input_tensor, target_tensor)
        fake_pred = discriminator(input_tensor, fake)

        loss_D = (
            bce_loss(real_pred, torch.ones_like(real_pred)) +
            bce_loss(fake_pred, torch.zeros_like(fake_pred))
        ) * 0.5

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # === Train Generator ===
        fake = generator(input_tensor)
        fake_pred = discriminator(input_tensor, fake)

        loss_G_adv = bce_loss(fake_pred, torch.ones_like(fake_pred))
        loss_G_l1 = l1_loss(fake, target_tensor)
        loss_G = loss_G_adv + 100 * loss_G_l1

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    logger.info(f"[Epoch {epoch+1}/{num_epochs}] Loss_D: {loss_D.item():.4f} | Loss_G: {loss_G.item():.4f}")
