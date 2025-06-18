import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
from pathlib import Path

class PetGanDataset(Dataset):

    def __init__(self, root_dir, transform=None, use_ct=False):
        self.root_dir = Path(root_dir)
        self.patients = sorted([p for p in self.root_dir.iterdir() if p.is_dir()])
        self.transform = transform
        self.use_ct = use_ct

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_dir = self.patients[idx]

        pet_pre = nib.load(patient_dir / "PET_pre.nii.gz").get_fdata()
        pet_post = nib.load(patient_dir / "PET_post.nii.gz").get_fdata()

        x = pet_pre.astype(np.float32)
        y = pet_post.astype(np.float32)

        if self.use_ct:
            ct = nib.load(patient_dir / "CT.nii.gz").get_fdata().astype(np.float32)
            input_tensor = np.stack([x, ct], axis=0)
        else:
            input_tensor = x[None, :, :, :]

        target_tensor = y[None, :, :, :]

        if self.transform:
            input_tensor, target_tensor = self.transform(input_tensor, target_tensor)

        return torch.tensor(input_tensor), torch.tensor(target_tensor)
