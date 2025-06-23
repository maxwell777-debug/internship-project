from pathlib import Path
import re
import nibabel as nib
from loguru import logger

from resampling import change_spacing, resample_like
from registration import register_image_to_reference
from normalization import (
    convert_pet_to_suv,
    save_image,
    normalize_ct_image,
    normalize_suv_image,
    load_pet_metadata,
)


def preprocess_patient(pet_baseline_path: Path, ct_baseline_path: Path, pet_normal_path: Path, metadata_json_path: Path, output_dir: Path):
    pet_baseline = nib.load(pet_baseline_path)
    pet_normal = nib.load(pet_normal_path)   
    ct_baseline = nib.load(ct_baseline_path)

    pet_baseline_iso = change_spacing(pet_baseline, new_spacing=1.5, interpolator="linear")
    pet_normal_iso = change_spacing(pet_normal, new_spacing=1.5, interpolator="linear")
    ct_baseline_iso = change_spacing(ct_baseline, new_spacing=1.5, interpolator="linear", default_pixel_value=-1000)

    pet_normal_aligned = register_image_to_reference(pet_normal_iso, pet_baseline_iso, transform_type="Rigid")

    ct_baseline_resampled = resample_like(ct_baseline_iso, pet_baseline_iso, interpolator="linear", default_pixel_value=-1000)
    pet_normal_resampled = resample_like(pet_normal_aligned, pet_baseline_iso, interpolator="linear")

    weight_kg, dose_bq = load_pet_metadata(metadata_json_path)
    suv_baseline = convert_pet_to_suv(pet_baseline_iso, weight_kg, dose_bq)
    suv_normal = convert_pet_to_suv(pet_normal_resampled, weight_kg, dose_bq)

    ct_baseline_normalized = normalize_ct_image(ct_baseline_resampled, clip_min=-200, clip_max=300)
    suv_baseline_normalized = normalize_suv_image(suv_baseline, mode="scale", scale_max=20.0)
    suv_normal_normalized = normalize_suv_image(suv_normal, mode="scale", scale_max=20.0)

    reset_nifti_scaling(ct_baseline_normalized)
    reset_nifti_scaling(suv_baseline_normalized)
    reset_nifti_scaling(suv_normal_normalized)

    save_image(ct_baseline_normalized.get_fdata(), ct_baseline_normalized.affine, ct_baseline_normalized.header, output_dir / "CT_baseline_preprocessed.nii.gz")
    save_image(suv_baseline_normalized.get_fdata(), suv_baseline_normalized.affine, suv_baseline_normalized.header, output_dir / "PET_baseline_preprocessed.nii.gz")
    save_image(suv_normal_normalized.get_fdata(), suv_normal_normalized.affine, suv_normal_normalized.header, output_dir / "PET_normal_preprocessed.nii.gz")


def preprocess_patient_from_dir(patient_dir: Path, output_dir: Path):
    logger.info(f"Launching preprocessing for patient: {patient_dir.name}")
    
    pet_baseline_path = patient_dir / "PET_baseline.nii.gz"
    pet_normal_path = patient_dir / "PET_normal.nii.gz"
    ct_baseline_path = patient_dir / "CT_baseline.nii.gz"
    metadata_path = patient_dir / "patient_info.json"

    for file_path in [pet_baseline_path, pet_normal_path, ct_baseline_path, metadata_path]:
        if not file_path.exists():
            logger.error(f"Missing file: {file_path}")
            raise FileNotFoundError(f"Expected file not found: {file_path}")

    preprocess_patient(pet_baseline_path, ct_baseline_path, pet_normal_path, metadata_path, output_dir)
    logger.info(f"Successfully preprocessing for patient {patient_dir.name}")


def preprocess_all_patients(root_processed_dir: Path, output_dir: Path = None):
    logger.info(f"Processing all patients in: {root_processed_dir}")

    if output_dir is None:
        output_dir = root_processed_dir.parent / "preprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)

    required_files = {
        "PET_baseline_preprocessed.nii.gz",
        "PET_normal_preprocessed.nii.gz",
        "CT_baseline_preprocessed.nii.gz"
    }

    for patient_dir in root_processed_dir.iterdir():
        if not patient_dir.is_dir():
            continue

        patient_processed_dir = output_dir / patient_dir.name
        patient_processed_dir.mkdir(parents=True, exist_ok=True)

        if required_files.issubset({f.name for f in patient_processed_dir.glob("*.nii.gz")}):
            logger.info(f"Preprocessing already completed for patient: {patient_dir.name}. Skipping.")
            continue

        logger.info(f"Preprocessing patient: {patient_dir.name}")
        preprocess_patient_from_dir(patient_dir, patient_processed_dir)

    logger.info("Full preprocessing completed for all patients.")


def reset_nifti_scaling(nifti_img):
    nifti_img.header['scl_slope'] = 1.0
    nifti_img.header['scl_inter'] = 0.0
    return nifti_img


if __name__ == "__main__":
    preprocess_all_patients(Path("data/processed"))
