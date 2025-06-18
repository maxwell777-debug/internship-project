from pathlib import Path
import sys

from loguru import logger
from Dcm2NiixConverter import Dcm2NiixConverter
from MetadataExtractor import MetadataExtractor
from ImageProcessor import ImageProcessor
from FileManager import FileManager


logger.remove()
logger.add(sys.stderr, level="INFO")


class DICOMPipeline:
    def __init__(self, ct_resize=True, suv=True):
        self.logger = logger.bind(name="DICOMPipeline")
        self.ct_resize = ct_resize
        self.suv = suv
        
        self.converter = Dcm2NiixConverter()
        self.metadata_extractor = MetadataExtractor()
        self.image_processor = ImageProcessor()
        self.file_manager = FileManager()

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
            self.converter.convert_dicom_series(input_dir, patient_output_dir)
            self.file_manager.delete_json_files(patient_output_dir)
            self.metadata_extractor.extract_patient_metadata(input_dir, patient_output_dir)
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

            # self.file_manager.delete_subfolders(patient_output_dir)
            self.file_manager.clean_invalid_nii_files(patient_output_dir)
            
            if self.ct_resize:
                for scan_type in ["baseline", "normal"]:
                    ct_path = patient_output_dir / f"CT_{scan_type}.nii.gz"
                    pet_path = patient_output_dir / f"PET_{scan_type}.nii.gz"

                    if ct_path.exists() and pet_path.exists():
                        self.image_processor.resample_ct_to_pet(ct_path, pet_path)
                        logger.info(f"Resampled CT for {scan_type} saved to: {ct_path}")
                    else:
                        self.logger.warning(f"CT or PET {scan_type} files are missing for resampling.")
            
            if self.suv:
                for scan_type in ["baseline", "normal"]:
                    pet_path = patient_output_dir / f"PET_{scan_type}.nii.gz"
                    json_path = patient_output_dir / "patient_info.json"
                    if pet_path.exists():
                        self.image_processor.calculate_suv(pet_path, json_path, patient_output_dir)
                    else:
                        self.logger.warning(f"PET {scan_type} file is missing for SUV calculation.")

        except Exception as e:
            self.logger.error(f"Processing failed for {input_dir.name}: {e}")
            self.logger.warning("DICOM files were NOT deleted due to error.")
        
        finally:
            logger.success(f"Successfully finished processing {input_dir.name}")


if __name__ == "__main__":
    pipeline = DICOMPipeline(ct_resize=False)
    pipeline.process_all_patients("./raw_data/")


#Not working for Bronzite, Cornaline, Epidote, Fluorine, Galene, Idocrase