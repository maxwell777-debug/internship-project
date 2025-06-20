import dicom2nifti
import shutil
import re
import pydicom
import json
import time
from pathlib import Path
from loguru import logger


def classify_dicom(name: str) -> tuple[str, str]:
    name = name.lower()
    modality = "PET" if re.search(r"pet|tep|pt", name) else "CT" if "ct" in name else "unknown"
    context = "baseline" if re.search(r"baseline|baseli|basel", name) else "normal" if "norma" in name else "unknown"
    return modality, context


def organize_dicom(input_dir: Path):
    logger.info(f"Organizing DICOM files from: {input_dir}")
    for file in input_dir.iterdir():
        if file.is_file() and file.suffix.lower() == ".dcm":
            modality, context = classify_dicom(file.name)
            dest = input_dir / f"{modality}_{context}"
            dest.mkdir(parents=True, exist_ok=True)
            shutil.move(file, dest / file.name)
    logger.info(f"DICOM files successfully organized.")


def convert_dicom_dirs(base_dir: Path, output_dir: Path):
    start_time = time.time()
    conversion_count = 0
    for subdir in base_dir.iterdir():
        if subdir.is_dir() and any(subdir.glob("*.dcm")):
            modality, context = subdir.name.split("_")
            output_subdir = output_dir / context
            output_subdir.mkdir(parents=True, exist_ok=True)
            output_path = output_subdir / f"{modality}.nii.gz"
            try:
                dcm_to_nifti(subdir, output_path)
                logger.info(f"Converted DICOMs in {subdir.name} to NIfTI format.")
                conversion_count += 1
            except Exception as e:
                logger.error(f"Failed to convert DICOMs in {subdir.name}: {e}")
        else:
            logger.warning(f"Skipped (no DICOM files found): {subdir.name}")
    elapsed_time = time.time() - start_time
    logger.info(f"Completed {conversion_count} DICOM directory conversions in {elapsed_time:.2f}s")


def dcm_to_nifti(input_path: Path, output_path: Path):
    logger.info(f"Converting DICOM to NIfTI: {input_path} → {output_path}")
    try:
        dicom2nifti.dicom_series_to_nifti(str(input_path), str(output_path), reorient_nifti=True)
        logger.info(f"NIfTI image saved to: {output_path}.")
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise


def extract_patient_metadata(raw_data_dir: Path, output_dir: Path):
    logger.info(f"Extracting patient metadata from DICOM files in: {raw_data_dir.name}...")

    pet_dirs = [raw_data_dir / "PET_baseline", raw_data_dir / "PET_normal"]
    pet_dicom_files = [f for d in pet_dirs if d.exists() for f in d.glob("*.dcm")]

    if not pet_dicom_files:
        logger.warning("No PET DICOM files found for metadata extraction.")
        return

    ds = pydicom.dcmread(pet_dicom_files[0])

    try:
        patient_weight = float(ds.get("PatientWeight", 0))  # kg
        injected_dose = float(ds.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)  # Bq
        patient_info = {
            "PatientWeight": patient_weight,
            "InjectedDose": injected_dose
        }

    except (AttributeError, IndexError, ValueError) as e:
        logger.error(f"Failed to extract DICOM metadata: {e}")
        raise ValueError("Missing DICOM fields required for SUV calculation.")

    output_json = output_dir / "patient_info.json"
    with open(output_json, "w") as f:
        json.dump(patient_info, f, indent=4)

    logger.info(f"Saved patient_info.json to: {output_json}.")


def process_all_patients(dicom_root_dir: Path, processed_root_dir: Path = None):
    total_start_time = time.time()
    logger.info("Starting batch processing of all patients...")
    patient_count = 0
    
    if processed_root_dir is None :
        processed_root_dir = dicom_root_dir.parent / "processed"
        
    processed_root_dir.mkdir(parents=True, exist_ok=True)
    
    for patient_dicom_dir in dicom_root_dir.iterdir():
        if patient_dicom_dir.is_dir() and any(patient_dicom_dir.glob("*.dcm")):
            patient_start_time = time.time() 
            logger.info(f"Processing patient: {patient_dicom_dir.name}")
            
            patient_processed_dir = processed_root_dir / patient_dicom_dir.name
            patient_processed_dir.mkdir(parents=True, exist_ok=True)
            organize_dicom(patient_dicom_dir)
            convert_dicom_dirs(patient_dicom_dir, patient_processed_dir)
            extract_patient_metadata(patient_dicom_dir, patient_processed_dir)
            
            patient_elapsed = time.time() - patient_start_time
            logger.info(f"Completed processing patient {patient_dicom_dir.name} in {patient_elapsed:.2f}s")
            patient_count += 1
    
    total_elapsed = time.time() - total_start_time
    logger.info(f"Completed processing of {patient_count} patients in {total_elapsed:.2f}s\n")


# ==== Example usage ====
dicom_root_dir = Path("data/raw")
process_all_patients(dicom_root_dir)
