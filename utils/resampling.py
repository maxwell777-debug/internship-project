from pathlib import Path
import SimpleITK as sitk
from loguru import logger
import nibabel as nib
from image_conversion import sitk_to_nib, nib_to_sitk


def change_spacing(image: nib.Nifti1Image, new_spacing: float = 1.0, interpolator: str = "linear", default_pixel_value: float = 0.0) -> nib.Nifti1Image:
    logger.info(f"Resampling image to spacing {new_spacing} mm using {interpolator} interpolation")

    sitk_image = nib_to_sitk(image)

    interp_map = {
        "linear": sitk.sitkLinear,
        "nearest": sitk.sitkNearestNeighbor,
        "bspline": sitk.sitkBSpline
    }

    if interpolator not in interp_map:
        raise ValueError(f"Unknown interpolator '{interpolator}'. Must be one of {list(interp_map)}.")

    interp = interp_map[interpolator]

    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()
    target_spacing = [new_spacing] * 3

    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(sitk_image.GetOrigin())
    resampler.SetOutputDirection(sitk_image.GetDirection())
    resampler.SetDefaultPixelValue(default_pixel_value)
    resampler.SetInterpolator(interp)

    resampled_sitk = resampler.Execute(sitk_image)
    return sitk_to_nib(resampled_sitk)


def resample_like(source_image: nib.Nifti1Image, target_image: nib.Nifti1Image, interpolator: str = "linear", default_pixel_value: float = 0.0) -> nib.Nifti1Image:
    logger.info("Resampling image to match reference shape, spacing, direction, and origin...")

    src = nib_to_sitk(source_image)
    tgt = nib_to_sitk(target_image)

    interp_map = {
        "linear": sitk.sitkLinear,
        "nearest": sitk.sitkNearestNeighbor,
        "bspline": sitk.sitkBSpline
    }

    if interpolator not in interp_map:
        raise ValueError(f"Unknown interpolator: {interpolator}")
    interp = interp_map[interpolator]

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(tgt)
    resampler.SetInterpolator(interp)
    resampler.SetDefaultPixelValue(default_pixel_value)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))

    resampled_sitk = resampler.Execute(src)
    return sitk_to_nib(resampled_sitk)


# ==== EXEMPLE D'UTILISATION ====
if __name__ == "__main__":
    source_img_path = Path("processed_data/Agathe/CT_baseline.nii.gz")
    target_img_path = Path("processed_data/Agathe/PET_baseline.nii.gz")
    output_img_path = Path("processed_data/Agathe/CT_spacing_1mm.nii.gz")

    source_img = nib.load(str(source_img_path))
    target_img = nib.load(str(target_img_path))

    resampled = change_spacing(source_img, new_spacing=1.0, interpolator="linear")
    nib.save(resampled, str(output_img_path))
