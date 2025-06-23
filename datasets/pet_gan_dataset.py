import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Tuple
from loguru import logger
from utils.patch import get_random_patch


class CtPetGanPatchDataset(Dataset):
    def __init__(self, root_dir: Path, patch_size=(128, 128, 128), mode: str = "random"):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.mode = mode
        self.patients = sorted([p for p in self.root_dir.iterdir() if p.is_dir()])

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_dir = self.patients[idx]
        baseline_dir = patient_dir / "baseline"
        normal_dir = patient_dir / "normal"

        pet_baseline = nib.load(baseline_dir / "PET_preprocessed.nii.gz").get_fdata().astype(np.float32)
        pet_normal = nib.load(normal_dir / "PET_preprocessed.nii.gz").get_fdata().astype(np.float32)

        input_tensor = pet_baseline[None, ...]
        target_tensor = pet_normal[None, ...]

        if self.mode == "random":
            input_patch, target_patch = get_random_patch(input_tensor, target_tensor, self.patch_size)
        elif self.mode == "segmentation":
            segmentation_dir = self.root_dir / "segmentation_output"
            logger.error("Segmentation mode is not implemented yet.")
            raise NotImplementedError("Segmentation mode is not implemented yet.")
        else :
            logger.error(f"Unknown mode: {self.mode}. Supported modes are 'random' and 'segmentation'.")
            raise ValueError(f"Unknown mode: {self.mode}. Supported modes are 'random' and 'segmentation'.")
        
        return torch.tensor(input_patch, dtype=torch.float32), torch.tensor(target_patch, dtype=torch.float32)



