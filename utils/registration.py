import ants
import time
import nibabel as nib
from pathlib import Path
from loguru import logger
from image_conversion import nib_to_ants, ants_to_nib


def register_image_to_reference(image_source: nib.Nifti1Image, target_image: nib.Nifti1Image, transform_type: str = "Rigid"):
    moving = nib_to_ants(image_source)
    fixed = nib_to_ants(target_image)

    logger.info(f"Registering post to pre using {transform_type} transform...")
    start_time = time.perf_counter()

    registration = ants.registration(fixed=fixed, moving=moving, type_of_transform=transform_type)

    duration = time.perf_counter() - start_time
    logger.info(f"Registration completed in {duration:.2f} seconds")

    aligned_image = registration["warpedmovout"]
    aligned_image_nib = ants_to_nib(aligned_image)
    return aligned_image_nib


# ====== Example usage ======
if __name__ == "__main__":
    dir_path = Path("data/preprocessed/")
    normal_path = dir_path / "PET_normal_preprocessed.nii.gz"
    baseline_path = dir_path / "PET_baseline_preprocessed.nii.gz"
    output_path = dir_path / "PET_normal_aligned.nii.gz"

    normal_image = nib.load(normal_path)
    baseline_image = nib.load(baseline_path)
    normal_aligned = register_image_to_reference(normal_image, baseline_image, transform_type="Rigid")
    nib.save(normal_aligned, output_path)
