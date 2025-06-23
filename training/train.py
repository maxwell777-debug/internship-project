from pathlib import Path
import torch
import torch.optim as optim
from torch.nn import BCELoss, L1Loss
from torch.utils.data import DataLoader
from loguru import logger
import numpy as np
import nibabel as nib

from datasets.pet_gan_dataset import CtPetGanPatchDataset
from models.discriminator import Discriminator3D
from models.generator import Generator3D

data_root = Path("data/processed")
patch_size = (128, 128, 128)
batch_size = 2
num_epochs = 100
lr = 2e-4
save_interval = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    logger.warning("CUDA not available — using CPU")

logger.info("Loading dataset...")
dataset = CtPetGanPatchDataset(data_root, patch_size=patch_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
logger.info(f"Loaded {len(dataset)} patients.")

in_channels_G = 1
in_channels_D = in_channels_G + 1

generator = Generator3D(in_channels=in_channels_G).to(device)
discriminator = Discriminator3D(in_channels=in_channels_D).to(device)

opt_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
opt_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

bce_loss = BCELoss()
l1_loss = L1Loss()

logger.info("Starting training loop...")
for epoch in range(num_epochs):
    for i, (input_tensor, target_tensor) in enumerate(dataloader):
        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)

        with torch.no_grad():
            fake = generator(input_tensor)

        real_pred = discriminator(input_tensor, target_tensor)
        fake_pred = discriminator(input_tensor, fake)

        loss_D = (
            bce_loss(real_pred, torch.ones_like(real_pred)) +
            bce_loss(fake_pred, torch.zeros_like(fake_pred))
        ) * 0.5

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        fake = generator(input_tensor)
        fake_pred = discriminator(input_tensor, fake)

        loss_G_adv = bce_loss(fake_pred, torch.ones_like(fake_pred))
        loss_G_l1 = l1_loss(fake, target_tensor)
        loss_G = loss_G_adv + 100 * loss_G_l1

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    logger.info(f"[Epoch {epoch+1}/{num_epochs}] Loss_D: {loss_D.item():.4f} | Loss_G: {loss_G.item():.4f}")

    if (epoch + 1) % save_interval == 0:
        generator.eval()
        with torch.no_grad():
            sample_input = input_tensor[0:1]
            sample_target = target_tensor[0:1]
            sample_fake = generator(sample_input)

        output_dir = Path("outputs") / f"epoch_{epoch+1:03d}"
        output_dir.mkdir(parents=True, exist_ok=True)

        def save_nifti(tensor, filename):
            array = tensor.squeeze().cpu().numpy()
            nib.save(nib.Nifti1Image(array, affine=np.eye(4)), filename)

        save_nifti(sample_input, output_dir / "input_pet.nii.gz")
        save_nifti(sample_target, output_dir / "target_pet.nii.gz")
        save_nifti(sample_fake, output_dir / "generated_pet.nii.gz")

        logger.info(f"Saved visual outputs to {output_dir}")
        generator.train()
