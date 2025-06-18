import os
import json
import SimpleITK as sitk
import nibabel as nib
from pathlib import Path
from loguru import logger


class ImageProcessor:
    def __init__(self):
        self.logger = logger.bind(name="ImageProcessor")
    
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
    
    def calculate_suv(self, pet_path: Path, json_path: Path, output_dir: Path):
        """Calculate and save SUV image from PET image and patient metadata"""
        self.logger.info(f"Calculating SUV for PET image: {pet_path}")
        
        # Load PET image and metadata
        pet_image = nib.load(pet_path)
        metadata = self._load_metadata(json_path)
        
        # Calculate SUV value
        suv_value = metadata["InjectedDose"] / (metadata["PatientWeight"] * 1000)
        pet_data = pet_image.get_fdata()
        suv_image = pet_data * suv_value
        
        # Create and save SUV image
        suv_nifti = nib.Nifti1Image(suv_image, pet_image.affine, pet_image.header)
        base_name = os.path.basename(str(pet_path)).replace('.nii.gz', '').replace('.nii', '')
        output_file = os.path.join(output_dir, f"{base_name}_SUV.nii.gz")
        
        nib.save(suv_nifti, output_file)
        self.logger.success(f"SUV image saved to {output_file}")
    
    def _load_metadata(self, json_file: Path):
        """Load patient metadata from JSON file"""
        with open(json_file, 'r') as f:
            metadata = json.load(f)
            
        if 'InjectedDose' not in metadata or 'PatientWeight' not in metadata:
            self.logger.error("JSON file must contain 'InjectedDose' and 'PatientWeight' fields.")
            raise ValueError("Missing required fields in JSON metadata.")
            
        self.logger.info(f"Loaded metadata: InjectedDose={metadata['InjectedDose']}, PatientWeight={metadata['PatientWeight']}")
        return metadata
        suv_image = data * suv_value
        return suv_image
    
    def save_suv_image(self, output_dir):
        suv_image = self.compute()
        suv_nifti = nib.Nifti1Image(suv_image, self.image.affine, self.image.header)
        base_name = os.path.basename(self.nii_file).replace('.nii.gz', '').replace('.nii', '')
        output_file = os.path.join(output_dir, f"{base_name}_SUV.nii.gz")        
        nib.save(suv_nifti, output_file)
        logger.success(f"SUV image saved to {output_file}")
