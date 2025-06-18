import os
import json
import nibabel as nib
from loguru import logger

class SUVCalculator:
    def __init__(self, json_file, nii_file):
        self.json_file = json_file
        self.nii_file = nii_file
        self.image = nib.load(nii_file)
        self.load_metadata()


    def load_metadata(self):
        with open(self.json_file, 'r') as f:
            metadata = json.load(f)
        if 'InjectedDose' not in metadata or 'PatientWeight' not in metadata:
            logger.error("JSON file must contain 'InjectedDose' and 'PatientWeight' fields.")
            raise ValueError("Missing required fields in JSON metadata.")
        self.injected_dose = metadata['InjectedDose']
        self.patient_weight = metadata['PatientWeight']
        logger.info(f"Loaded metadata: InjectedDose={self.injected_dose}, PatientWeight={self.patient_weight}")


    def compute(self):
        suv_value = self.injected_dose / (self.patient_weight * 1000)
        data = self.image.get_fdata()
        suv_image = data * suv_value
        return suv_image
    
    
    def save_suv_image(self, output_dir):
        suv_image = self.compute()
        suv_nifti = nib.Nifti1Image(suv_image, self.image.affine, self.image.header)
        base_name = os.path.basename(self.nii_file).replace('.nii.gz', '').replace('.nii', '')
        output_file = os.path.join(output_dir, f"{base_name}_SUV.nii.gz")        
        nib.save(suv_nifti, output_file)
        logger.success(f"SUV image saved to {output_file}")


calculator = SUVCalculator(
    json_file="Patient0/nii/patient_info.json",
    nii_file="Patient0/nii/baseline_PET.nii.gz"
)

calculator.save_suv_image("Patient0/nii")


