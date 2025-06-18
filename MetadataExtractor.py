from pathlib import Path
import json
import pydicom
from loguru import logger


class MetadataExtractor:
    def __init__(self):
        self.logger = logger.bind(name="MetadataExtractor")
    
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
