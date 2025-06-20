import os
import nibabel as nib
import numpy as np
from pathlib import Path
from loguru import logger


ORGANS_THRESHOLDS = {
    "brain": 10.0,
    "liver": 3.5,
    "kidney_left": 5.0,
    "kidney_right": 5.0,
    "urinary_bladder": 20.0,
    "spleen": 4.0,
    "heart": 7.5,
    "stomach": 4.0,
    "small_bowel": 3.5,
    "colon": 3.5,
}


def generate_physiological_mask(ct_path: Path, mask_dir: Path) -> nib.Nifti1Image:
    ct_image = nib.load(ct_path)
    mask = np.zeros(ct_image.shape, dtype=np.uint8)

    for organ in ORGANS_THRESHOLDS:
        organ_path = mask_dir / f"{organ}.nii.gz"
        if not organ_path.exists():
            logger.warning(f"Missing mask for organ: {organ} -> {organ_path}")
            continue
        organ_data = nib.load(organ_path).get_fdata()
        organ_mask = (organ_data > 0).astype(np.uint8)
        mask = np.logical_or(mask, organ_mask).astype(np.uint8)

    logger.info(f"Generated combined physiological mask for: {ct_path.name}")
    return nib.Nifti1Image(mask, ct_image.affine, ct_image.header)


def save_mask_image(mask_image: nib.Nifti1Image, output_path: Path):
    nib.save(mask_image, output_path)
    logger.success(f"Saved physiological mask to: {output_path}")


def suppress_physiological_uptake_on_pet(pet_path: Path, mask_dir: Path, output_path: Path):
    logger.info(f"Applying physiological suppression on PET: {pet_path.name}")
    tep_image = nib.load(pet_path)
    tep_data = tep_image.get_fdata()

    filtered_voxels = tep_data[(tep_data > 0) & (tep_data <= 20)]
    mean_noise = np.mean(filtered_voxels)
    logger.debug(f"Estimated physiological noise: {mean_noise:.3f}")

    for organ, divisor in ORGANS_THRESHOLDS.items():
        organ_path = mask_dir / f"{organ}.nii.gz"
        if not organ_path.exists():
            logger.warning(f"Missing mask for organ: {organ}")
            continue
        organ_mask = (nib.load(organ_path).get_fdata() > 0)
        tep_data[organ_mask] /= (divisor * mean_noise)

    masked_img = nib.Nifti1Image(tep_data.astype(np.float32), tep_image.affine, tep_image.header)
    nib.save(masked_img, output_path)
    logger.success(f"Saved PET with physiological uptake suppressed to: {output_path}")


# ========== Exemple d'utilisation ==========
if __name__ == "__main__":
    ct_file = Path("processed_data/Agathe/CT_baseline.nii.gz")
    mask_dir = Path("processed_data/segmentation_output")
    output_dir = Path("processed_data/Agathe")

    mask_img = generate_physiological_mask(ct_file, mask_dir)
    save_mask_image(mask_img, output_dir / "physiological_mask.nii.gz")

    suppress_physiological_uptake_on_pet(
        pet_path=Path("processed_data/Agathe/PET_baseline_SUV.nii.gz"),
        mask_dir=mask_dir,
        output_path=output_dir / "PET_baseline_SUV_masked.nii.gz"
    )
