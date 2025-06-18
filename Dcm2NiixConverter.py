import subprocess
from pathlib import Path
import shutil
import re
from loguru import logger


class Dcm2NiixConverter:
    def __init__(self):
        self.logger = logger.bind(name="DICOMConverter")
    
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
