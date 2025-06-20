from pathlib import Path
import nibabel as nib
from loguru import logger

from resampling import change_spacing, resample_like
from normalization import (
    convert_pet_to_suv,
    save_image,
    normalize_ct_image,
    normalize_suv_image,
    load_pet_metadata,
)


def preprocess_ct_pet_pair(ct_path: Path, pet_path: Path, metadata_json_path: Path, output_dir: Path):
    ct_image = nib.load(ct_path)
    pet_image = nib.load(pet_path)

    ct_iso = change_spacing(ct_image, new_spacing=1.5, interpolator="linear", default_pixel_value=-1000)
    pet_iso = change_spacing(pet_image, new_spacing=1.5, interpolator="linear")

    ct_resampled = resample_like(ct_iso, pet_iso, interpolator="linear", default_pixel_value=-1000)

    weight_kg, dose_bq = load_pet_metadata(metadata_json_path)
    suv_image = convert_pet_to_suv(pet_iso, weight_kg, dose_bq)

    norm_ct = normalize_ct_image(ct_resampled, clip_min=-200, clip_max=300)
    norm_suv = normalize_suv_image(suv_image, mode="scale", scale_max=20.0)

    norm_ct.header['scl_slope'] = 1.0
    norm_ct.header['scl_inter'] = 0.0
    norm_suv.header['scl_slope'] = 1.0
    norm_suv.header['scl_inter'] = 0.0

    save_image(norm_ct.get_fdata(), norm_ct.affine, norm_ct.header, output_dir / "CT_preprocessed.nii.gz")
    save_image(norm_suv.get_fdata(), norm_suv.affine, norm_suv.header, output_dir / "PET_preprocessed.nii.gz")


def preprocess_ct_pet_pair_from_dir(patient_dir: Path, context: str):
    logger.info(f"Launching preprocessing for context: {context} in patient folder: {patient_dir.name}")

    if context not in ["baseline", "normal"]:
        logger.error(f"Invalid context: {context}. Expected 'baseline' or 'normal'.")
        raise ValueError(f"Invalid context: {context}")

    ct_path = patient_dir / context / "CT.nii.gz"
    pet_path = patient_dir / context / "PET.nii.gz"
    metadata_path = patient_dir / "patient_info.json"
    output_dir = patient_dir / context

    for file_path in [ct_path, pet_path, metadata_path]:
        if not file_path.exists():
            logger.error(f"Missing file: {file_path}")
            raise FileNotFoundError(f"Expected file not found: {file_path}")

    preprocess_ct_pet_pair(ct_path, pet_path, metadata_path, output_dir)
    logger.info(f"Successfully preprocessing {context} for {patient_dir.name}")


def preprocess_all_ct_pet_pairs_for_patient(patient_dir: Path):
    logger.info(f"Running full CT/PET preprocessing for all contexts in: {patient_dir.name}")
    for context in ["baseline", "normal"]:
        preprocess_ct_pet_pair_from_dir(patient_dir, context)
    logger.info(f"All preprocessing completed for {patient_dir.name}.")


def preprocess_all_patients(root_processed_dir: Path):
    logger.info(f"Processing all patients in: {root_processed_dir}")
    for patient_dir in root_processed_dir.iterdir():
        if patient_dir.is_dir() and (patient_dir / "baseline").exists() and (patient_dir / "normal").exists():
            preprocess_all_ct_pet_pairs_for_patient(patient_dir)
    logger.info("Full preprocessing completed for all patients.")


# ==== Example ====
if __name__ == "__main__":
    preprocess_all_patients(Path("data/processed"))
