import subprocess
from pathlib import Path
import shutil
import sys
import json
import pydicom
import SimpleITK as sitk
import re

from loguru import logger
import nibabel as nib


logger.remove()
logger.add(sys.stderr, level="INFO")


class DICOMPipeline:

    def __init__(self, ct_resize=True):
        self.logger = logger.bind(name="DICOMPipeline")
        self.ct_resize = ct_resize

    def process_all_patients(self, raw_data_dir):
        self.logger.info(f"--- Processing all patients in directory: {raw_data_dir} ---")
        patients_dir = Path(raw_data_dir)

        if not patients_dir.exists() or not patients_dir.is_dir():
            self.logger.error(f"Invalid directory: {patients_dir}")
            return
        
        processed_data_dir = patients_dir.parent / "processed_data"
        processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Processed data will be saved in: {processed_data_dir}")

        for patient_dir in patients_dir.iterdir():
            if patient_dir.is_dir():
                self.process_patient(patient_dir, processed_data_dir)
            else:
                self.logger.warning(f"Skipping non-directory file: {patient_dir.name}")


    def process_patient(self, input_dir, output_dir):
        self.logger.info(f"--- Processing patient folder: {input_dir} ---")
        input_dir = Path(input_dir)

        patient_output_dir = output_dir / input_dir.name
        patient_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.convert_dicom_series(input_dir, patient_output_dir)
            self.delete_json_files(patient_output_dir)
            self.extract_patient_metadata(input_dir, patient_output_dir)
            required_outputs = [
                "PET_baseline.nii.gz",
                "CT_baseline.nii.gz",
                "PET_normal.nii.gz",
                "CT_normal.nii.gz",
                "patient_info.json"
            ]
            missing = [f for f in required_outputs if not (patient_output_dir / f).exists()]
            if missing:
                self.logger.error(f"Missing expected output files: {missing}")
                raise RuntimeError("Incomplete processing, skipping DICOM deletion.")

            # self.delete_subfolders(patient_output_dir)
            self.clean_invalid_nii_files(patient_output_dir)
            if self.ct_resize:
                for scan_type in ["baseline", "normal"]:
                    ct_path = patient_output_dir / f"CT_{scan_type}.nii.gz"
                    pet_path = patient_output_dir / f"PET_{scan_type}.nii.gz"

                    if ct_path.exists() and pet_path.exists():
                        self.resample_ct_to_pet(ct_path, pet_path)
                    else:
                        self.logger.warning(f"CT or PET {scan_type} files are missing for resampling.")

        except Exception as e:
            self.logger.error(f"Processing failed for {input_dir.name}: {e}")
            self.logger.warning("DICOM files were NOT deleted due to error.")
        
        finally:
            logger.success(f"Successfully finished processing {input_dir.name}")

    
    def convert_dicom_series(self, input_dir: Path, output_dir: Path):
        self.logger.info(f"Converting DICOM series in {input_dir.name} to NIfTI...")

        for file in input_dir.iterdir():
            if file.is_file() and file.suffix.lower() == ".dcm":
                modality, context = self.classify_dicom_filename(file.name)
                dir_path = output_dir / f"{modality}_{context}_dcm"
                dir_path.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, dir_path / file.name)

        for subdir in output_dir.iterdir():
            if subdir.is_dir():
                self.logger.info(f"Processing DICOM files in {subdir.name}...")
                self._run_dcm2niix(subdir, output_dir)



    def _run_dcm2niix(self, dicom_dir: Path, output_dir: Path):
        if dicom_dir.is_dir() and not any(dicom_dir.glob("*.dcm")):
            self.logger.warning(f"No DICOM files found in {dicom_dir.name}")
            return
        filename = dicom_dir.name.strip("_dcm")

        try:
            subprocess.run(
                [
                    "dcm2niix",           # DICOM to NIfTI converter
                    "-z", "y",            # Compress output (.nii.gz)
                    "-b", "y",            # Skip anonymized BIDS JSON
                    "-o", str(output_dir),# Output directory
                    "-f", filename,    # Output filename format
                    str(dicom_dir)        # Input DICOM folder
                ],
                check=True,
                capture_output=True,
                text=True
            )
            self.logger.info(f"Conversion successful for '{dicom_dir.name}'")

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Conversion failed for '{dicom_dir.name}': {e}")


    def delete_json_files(self, folder: Path):
        self.logger.info(f"Deleting JSON files in {folder.name}...")
        for file in folder.glob("*.json"):
            try:
                file.unlink()
            except Exception as e:
                self.logger.warning(f"Failed to delete {file.name}: {e}")

    def delete_subfolders(self, folder: Path):
        self.logger.info(f"Deleting subdirectories in {folder.name}...")
        for subdir in folder.iterdir():
            if subdir.is_dir():
                try:
                    shutil.rmtree(subdir)
                except Exception as e:
                    self.logger.warning(f"Failed to remove directory {subdir.name}: {e}")


    def classify_dicom_filename(self, filename: str):
        name = filename.lower()
    
        if re.search(r"pet|tep|pt", name):
            modality = "PET"
        elif "ct" in name:
            modality = "CT"
        else:
            modality = "unknown"

        if re.search(r"baseline|baseli|basel", name):
            context = "baseline"
        elif re.search(r"normal|norma", name):
            context = "normal"
        else:
            context = "unknown"

        return modality, context
    
    
    def extract_patient_metadata(self, input_dir: Path, output_dir: Path):
        self.logger.info(f"Extracting patient metadata from DICOM files for {input_dir.name}...")
        PET_dir = [output_dir / "PET_baseline_dcm", output_dir / "PET_normal_dcm"]
        PET_dicom_files = [f for d in PET_dir if d.exists() for f in d.glob("*.dcm")]

        if not PET_dicom_files:
            self.logger.warning("No PET DICOM files found for patient_info.json extraction")
            return

        ds = pydicom.dcmread(PET_dicom_files[0])
        try:
            patient_info = {
                "PatientWeight": float(ds.get("PatientWeight", 0)),
                "InjectedDose": float(ds.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)
            }

        except (AttributeError, IndexError, ValueError) as e:
            self.logger.error(f"Failed to extract DICOM metadata: {e}")
            raise ValueError("Required DICOM fields are missing for SUV calculation.")

        output_json = output_dir / "patient_info.json"

        with open(output_json, "w") as f:
            json.dump(patient_info, f, indent=4)

        self.logger.info(f"Saved patient_info.json to: {output_json}")


    def clean_invalid_nii_files(self, folder: Path):
        self.logger.info(f"Cleaning redundant NIfTI files in {folder.name}...")

        base_names = ["CT_baseline", "CT_normal", "PET_baseline", "PET_normal"]

        for base in base_names:
            nii_files = list(folder.glob(f"{base}*.nii*"))
            if len(nii_files) <= 1:
                continue
            best_file = None
            max_voxels = 0

            for file in nii_files:
                try:
                    img = nib.load(str(file))
                    shape = img.shape
                    num_voxels = 1
                    for dim in shape:
                        num_voxels *= dim

                    if num_voxels > max_voxels:
                        max_voxels = num_voxels
                        best_file = file

                except Exception as e:
                    self.logger.warning(f"Failed to read {file.name}: {e}")

            for file in nii_files:
                if file != best_file:
                    try:
                        file.unlink()
                        self.logger.info(f"Deleted redundant file: {file.name}")
                    except Exception as e:
                        self.logger.warning(f"Failed to delete {file.name}: {e}")
            clean_name = base + ".nii.gz"
            clean_path = folder / clean_name
            if best_file.name != clean_name:
                self.logger.info(f"Renaming {best_file.name} to {clean_name}")
                best_file.rename(clean_path)

    def resample_ct_to_pet(self, ct_path: Path, pet_path: Path):
        self.logger.info("Resampling CT image to match PET dimensions...")

        ct_img = sitk.ReadImage(str(ct_path), sitk.sitkFloat32)
        pet_img = sitk.ReadImage(str(pet_path), sitk.sitkFloat32)

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(pet_img)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)

        resampled_ct = resampler.Execute(ct_img)

        sitk.WriteImage(resampled_ct, str(ct_path))
        self.logger.success(f"Resampled CT saved to: {ct_path}")



if __name__ == "__main__":
    pipeline = DICOMPipeline(ct_resize=False)
    pipeline.process_all_patients("./raw_data/")
    

#Not working for Bronzite, Cornaline, Epidote, Fluorine, Galene, Idocrase