import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
from pathlib import Path
import random


class CtPetGanPatchDataset(Dataset):
    def __init__(self, root_dir: Path, patch_size=(128, 128, 128), use_ct=False):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.use_ct = use_ct
        self.patients = sorted([p for p in self.root_dir.iterdir() if p.is_dir()])

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_dir = self.patients[idx]
        baseline_dir = patient_dir / "baseline"
        normal_dir = patient_dir / "normal"

        pet_baseline = nib.load(baseline_dir / "PET_preprocessed.nii.gz").get_fdata().astype(np.float32)
        pet_normal = nib.load(normal_dir / "PET_preprocessed.nii.gz").get_fdata().astype(np.float32)

        if self.use_ct:
            ct = nib.load(baseline_dir / "CT_preprocessed.nii.gz").get_fdata().astype(np.float32)

            # Ensure CT has same shape as PET
            assert ct.shape == pet_baseline.shape, f"CT and PET shapes differ: {ct.shape} vs {pet_baseline.shape}"

            input_tensor = np.stack([pet_baseline, ct], axis=0)  # shape: [2, D, H, W]
        else:
            input_tensor = pet_baseline[None, ...]  # shape: [1, D, H, W]

        target_tensor = pet_normal[None, ...]  # shape: [1, D, H, W]

        # Shape checks
        C, D, H, W = input_tensor.shape
        pd, ph, pw = self.patch_size

        if D < pd or H < ph or W < pw:
            raise ValueError(
                f"Volume too small for patch {self.patch_size}: got ({D}, {H}, {W}) for patient {patient_dir.name}"
            )

        # Random center-safe extraction
        margin_d, margin_h, margin_w = pd // 2, ph // 2, pw // 2

        center_d = random.randint(margin_d, D - margin_d)
        center_h = random.randint(margin_h, H - margin_h)
        center_w = random.randint(margin_w, W - margin_w)

        d0 = center_d - margin_d
        h0 = center_h - margin_h
        w0 = center_w - margin_w

        input_patch = input_tensor[:, d0:d0+pd, h0:h0+ph, w0:w0+pw]
        target_patch = target_tensor[:, d0:d0+pd, h0:h0+ph, w0:w0+pw]

        # === Vérification finale des tailles ===
        expected_in = (2 if self.use_ct else 1, pd, ph, pw)
        assert input_patch.shape == expected_in, f"Bad input shape: {input_patch.shape} vs {expected_in}"
        assert target_patch.shape == (1, pd, ph, pw), f"Bad target shape: {target_patch.shape}"

        return torch.tensor(input_patch, dtype=torch.float32), torch.tensor(target_patch, dtype=torch.float32)
