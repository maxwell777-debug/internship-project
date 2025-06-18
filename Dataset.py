import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from pathlib import Path
import random as rd

from DICOMToNiftiConverter import DICOMToNiftiConverter


class PatientDataset(Dataset):
    def __init__(self, data_dir, size=None, transform=None):
        """
        root_dir: dossier racine contenant tous les dossiers PatientXX
        patient_ids: liste des IDs patient (ex: ["Patient0", "Patient1", ...])
        transform: transformations PyTorch/MONAI à appliquer
        """
        self.root_dir = Path(data_dir)
        self.size = size
        self.transform = transform



    def __len__(self):
        return self.size
    
    def load(self):
        nb_patients = len([d for d in self.root_dir.iterdir() if d.is_dir() and d.name.startswith("Patient")])
        patient_ids = [f"Patient{i}" for i in range(nb_patients)]
        if self.size:
            patient_ids = rd.sample(patient_ids, self.size)

        converter = DICOMToNiftiConverter()
        for patient_id in patient_ids:
            converter.extract(
                input_dir=self.root_dir / patient_id / "dcm",
                output_dir=self.root_dir / patient_id / "nii"
            )

    def __getitem__(self, idx):
        patient_id = f"Patient{idx}"
        patient_path = self.root_dir / patient_id / "nii"
 
        # Chargement des volumes
        pet_suv_path = patient_path / "baseline_PET_SUV.nii.gz"
        pet_post_path = patient_path / "followup_PET.nii.gz"
        mask_dir = self.root_dir / patient_id / "segmentation_output"

        pet_suv = nib.load(pet_suv_path).get_fdata().astype(np.float32)
        pet_post = nib.load(pet_post_path).get_fdata().astype(np.float32)

        # Fusion des masques physiologiques
        physio_mask = self._load_and_combine_masks(mask_dir, pet_suv.shape)

        # Optionnel : masquage direct (ex: pet_suv[physio_mask == 1] = 0)
        # pet_suv *= (1 - physio_mask)

        sample = {
            "input": np.stack([pet_suv, physio_mask], axis=0),  # 2 canaux : SUV + masque
            "target": pet_post[np.newaxis, ...]  # 1 canal
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _load_and_combine_masks(self, mask_dir, shape):
        """Fusionne les masques présents dans le dossier donné."""
        combined = np.zeros(shape, dtype=np.uint8)
        for mask_file in os.listdir(mask_dir):
            if mask_file.endswith(".nii.gz"):
                mask = nib.load(mask_dir / mask_file).get_fdata()
                combined |= (mask > 0).astype(np.uint8)
        return combined
