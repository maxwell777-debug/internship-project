import nibabel as nib
import numpy as np
from pathlib import Path
from loguru import logger
import os

class PhysiologicalMaskGenerator:

    def __init__(self, ct_file, mask_dir):
        self.ct_file = Path(ct_file)
        self.mask_dir = Path(mask_dir)
        self.ct_image = nib.load(self.ct_file)
        self.organs = {
            "brain": 10.0,
            "liver": 3.5,
            "kidney_left": 5.0,
            "kidney_right": 5.0,
            "urinary_bladder": 20.0,
            "spleen": 4.0,
            "heart": 7.5,
            "stomach": 4.0,
            "small_bowel": 3.5,
            "colon": 3.5,
        }


        
    def generate_masks(self):
        mask = np.zeros(self.ct_image.shape, dtype=np.uint8)
        for organ in self.organs.keys():
            organ_path = self.mask_dir / f"{organ}.nii.gz"
            if not organ_path.exists():
                logger.warning(f"Mask file for {organ} does not exist: {organ_path}")
                continue
            organ_img = nib.load(organ_path)
            organ_mask = (organ_img.get_fdata() > 0).astype(np.uint8)
            mask = np.logical_or(mask, organ_mask).astype(np.uint8)
        logger.info(f"Generated mask for CT image: {self.ct_file}")
        return mask
    

    def save_mask(self, output_dir):
        logger.info(f"Generating physiological mask from CT image: {self.ct_file}...")
        mask = self.generate_masks()
        mask_image = nib.Nifti1Image(mask.astype(np.uint8), self.ct_image.affine, self.ct_image.header)
        output_path = Path(output_dir) / "physiological_mask.nii.gz"
        nib.save(mask_image, output_path)
        logger.success(f"Saved mask to {output_path}")


    def suppress_physiological_uptake(self, tep_file, output_dir):
        logger.info(f"Applying mask on TEP image: {tep_file}...")
        tep_image = nib.load(tep_file)
        tep_data = tep_image.get_fdata()
        filtered_voxels = tep_data[(tep_data > 0) & (tep_data <= 20)]
        mean_noise = np.mean(filtered_voxels)
        logger.debug(f"Mean noise level calculated: {filtered_voxels}")
        for organ in self.organs.keys():
            organ_path = self.mask_dir / f"{organ}.nii.gz"
            if not organ_path.exists():
                logger.warning(f"Mask file for {organ} does not exist: {organ_path}")
                continue
            organ_img = nib.load(organ_path)
            organ_mask = (organ_img.get_fdata() > 0).astype(np.uint8)
            tep_data[organ_mask == 1] /= self.organs[organ] * mean_noise
        masked_image = nib.Nifti1Image(tep_data, tep_image.affine, tep_image.header)
        output_path = Path(output_dir) / "masked_tep.nii.gz"
        nib.save(masked_image, output_path)
        logger.success(f"Saved masked TEP image to {output_path}")


mask_generator = PhysiologicalMaskGenerator(
    ct_file="Patient0/nii/baseline_CTa.nii.gz",
    mask_dir="Patient0/segmentation_output"
)

# mask_generator.save_mask(output_dir="Patient0/nii")

# mask_generator.suppress_physiological_uptake(
#     tep_file="Patient0/nii/baseline_PET_SUV.nii.gz",
#     output_dir="Patient0/nii"
# )

img = nib.load("Patient0/nii/baseline_CTa.nii.gz")
hdr = img.header

print("dtype:", hdr.get_data_dtype())
print("scl_slope:", hdr['scl_slope'])
print("scl_inter:", hdr['scl_inter'])
print("intent_name:", hdr.get('intent_name'))
print("description:", hdr['descrip'])