import os
import json
from re import M
import numpy as np
import nibabel as nib
from pathlib import Path
from loguru import logger


def load_pet_metadata(json_file: Path) -> tuple[float, float]:
    with open(json_file, 'r') as f:
        metadata = json.load(f)

    if 'InjectedDose' not in metadata or 'PatientWeight' not in metadata:
        logger.error("JSON must contain 'InjectedDose' and 'PatientWeight'.")
        raise ValueError("Missing required fields in metadata.")

    weight_kg = float(metadata["PatientWeight"])
    dose_bq = float(metadata["InjectedDose"])

    logger.info(f"Metadata loaded: PatientWeight = {weight_kg} kg, InjectedDose = {dose_bq} Bq")
    return weight_kg, dose_bq


def save_image(array: np.ndarray, affine, header, output_path: Path):
    nifti_img = nib.Nifti1Image(array.astype(np.float32), affine, header)
    nib.save(nifti_img, str(output_path))
    logger.info(f"Image saved to: {output_path}")


def convert_pet_to_suv(pet_image: nib.Nifti1Image, weight_kg: float, dose_bq: float) -> nib.Nifti1Image:
    logger.info("Computing SUV from PET image...")

    if weight_kg == 0.0:
        logger.error("Invalid patient weight: cannot be zero.")
        raise ValueError("Patient weight must be non-zero to compute SUV.")
    if dose_bq == 0.0:
        logger.error("Invalid injected dose: cannot be zero.")
        raise ValueError("Injected dose must be non-zero to compute SUV.")

    dose_mbq = dose_bq / 1_000_000  # Convert Bq to MBq
    weight_g = weight_kg * 1000     # Convert kg to g

    suv_factor = dose_mbq / weight_g

    pet_data = pet_image.get_fdata()
    suv_data = pet_data * suv_factor

    return nib.Nifti1Image(suv_data.astype(np.float32), pet_image.affine, pet_image.header)



def normalize_suv_image(suv_image: nib.Nifti1Image, mode: str = "scale", scale_max: float = 20.0) -> nib.Nifti1Image:

    logger.info(f"Normalizing SUV image with mode: {mode}")
    suv_data = suv_image.get_fdata()

    if mode == "scale":
        suv_data = np.clip(suv_data, 0, scale_max)
        suv_data = suv_data / scale_max

    elif mode == "percentile":
        p99 = np.percentile(suv_data, 99)
        suv_data = np.clip(suv_data, 0, p99)
        suv_data = suv_data / p99

    elif mode == "minmax":
        min_val = suv_data.min()
        max_val = suv_data.max()
        if max_val - min_val > 0:
            suv_data = (suv_data - min_val) / (max_val - min_val)
        else:
            logger.warning("SUV image has constant value; skipping normalization.")
            suv_data = np.zeros_like(suv_data)

    else:
        raise ValueError(f"Unknown normalization mode: {mode}")

    return nib.Nifti1Image(suv_data.astype(np.float32), suv_image.affine, suv_image.header)



def normalize_ct_image(ct_image: nib.Nifti1Image, clip_min: int = -200, clip_max: int = 300) -> nib.Nifti1Image:
    logger.info(f"Normalizing CT image with windowing [{clip_min}, {clip_max}]")

    ct_data = ct_image.get_fdata()
    ct_data = np.clip(ct_data, clip_min, clip_max)
    ct_data = (ct_data - clip_min) / (clip_max - clip_min)

    return nib.Nifti1Image(ct_data.astype(np.float32), ct_image.affine, ct_image.header)