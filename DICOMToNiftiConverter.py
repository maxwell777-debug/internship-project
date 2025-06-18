import subprocess
from pathlib import Path
import shutil
import tempfile
import sys
import json
import pydicom
from loguru import logger
import SimpleITK as sitk
import re

logger.remove()
logger.add(sys.stderr, level="INFO")


class RawDataProcessor:
    def __init__(self, suv=True, ct_resize=True):
        self.logger = logger.bind(name="RawDataProcessor")
        self.suv = suv
        self.ct_resize = ct_resize

    def convert_and_cleanup(self, patient_dir: Path):
        """
        Converts DICOMs to NIfTI, extracts metadata, and deletes DICOMs
        only if everything succeeded.
        """
        patient_dir = Path(patient_dir)
        nii_dir = patient_dir
        self.logger.info(f"--- Processing patient folder: {patient_dir.name} ---")

        try:
            self.convert_dicom_series(dicom_dir=patient_dir, output_dir=nii_dir)
            self.extract_patient_metadata(dicom_dir=patient_dir, output_dir=nii_dir)

            # Check for required outputs before deleting
            required_outputs = [
                "baseline_PET.nii.gz",  # add more if needed
                "patient_info.json"
            ]
            missing = [f for f in required_outputs if not (nii_dir / f).exists()]
            if missing:
                self.logger.error(f"Missing expected output files: {missing}")
                raise RuntimeError("Incomplete processing, skipping DICOM deletion.")

            self._delete_dicoms(dicom_dir=patient_dir)
            self.logger.success(f"DICOM files removed for {patient_dir.name}")

        except Exception as e:
            self.logger.error(f"Processing failed for {patient_dir.name}: {e}")
            self.logger.warning("DICOM files were NOT deleted due to error.")


    def convert_dicom_series(self, dicom_dir: Path, output_dir: Path):
        patterns = [
            "*_normal*PET.*",
            "*_normal*CT.*",
            "*_basel*CT.*",
            "*_basel*PET.*"
        ]

        self.logger.info(f"Converting DICOM to NIfTI from: {dicom_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        for pattern in patterns:
            self._run_dcm2niix(dicom_dir, output_dir, pattern)

    def _run_dcm2niix(self, dicom_dir: Path, output_dir: Path, pattern: str):
        dicom_files = list(dicom_dir.glob(pattern))
        if not dicom_files:
            self.logger.warning(f"No DICOM files found for pattern: '{pattern}'")
            return

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            for dcm in dicom_files:
                shutil.copy(dcm, tmp_path / dcm.name)

            output_name = pattern.replace("*_", "").replace(".*", "")
            try:
                subprocess.run(
                    [
                        "dcm2niix",
                        "-z", "y", "-b", "y", "-ba", "y",
                        "-o", str(output_dir),
                        "-f", output_name,
                        str(tmp_path)
                    ],
                    check=True,
                    capture_output=True,
                    text=True
                )
                self.logger.info(f"Conversion successful: {output_name}")
                self._delete_dcm2niix_json(output_dir, output_name)
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Conversion failed for pattern '{pattern}'")
                self.logger.debug(f"dcm2niix stderr: {e.stderr}")

    def _delete_dcm2niix_json(self, output_dir: Path, base_name: str):
        json_file = output_dir / f"{base_name}.json"
        if json_file.exists():
            try:
                json_file.unlink()
                self.logger.debug(f"Deleted temporary metadata file: {json_file.name}")
            except Exception as e:
                self.logger.warning(f"Failed to delete {json_file.name}: {e}")

    def extract_patient_metadata(self, dicom_dir: Path, output_dir: Path):
        dicom_files = list(dicom_dir.glob("*_PET.*.dcm"))
        if not dicom_files:
            self.logger.warning("No PET DICOM files found for patient_info.json extraction")
            return

        ds = pydicom.dcmread(dicom_files[0])
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

        self.logger.success(f"Saved patient_info.json to: {output_json}")

    def _delete_dicoms(self, dicom_dir: Path):
        for dcm_file in dicom_dir.glob("*.dcm"):
            try:
                dcm_file.unlink()
            except Exception as e:
                self.logger.warning(f"Failed to delete DICOM file {dcm_file}: {e}")

    def resize_ct_to_pet(self, ct_path: Path, pet_path: Path, output_path: Path = None):
        self.logger.info("Resampling CT image to match PET dimensions...")

        ct_img = sitk.ReadImage(str(ct_path), sitk.sitkFloat32)
        pet_img = sitk.ReadImage(str(pet_path), sitk.sitkFloat32)

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(pet_img)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)

        resampled_ct = resampler.Execute(ct_img)
        output_path = output_path or ct_path

        sitk.WriteImage(resampled_ct, str(output_path))
        self.logger.success(f"Resampled CT saved to: {output_path}")



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



if __name__ == "__main__":
    processor = RawDataProcessor()
    root_dir = Path("./data")
    for subfolder in root_dir.iterdir():
        if subfolder.is_dir():
            print(f"-----------Processing folder: {subfolder.name} -----------")
            for file in subfolder.iterdir():
                if file.is_file():
                    print(processor.classify_dicom_filename(file.name))



    
