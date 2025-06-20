import numpy as np
import nibabel as nib
import SimpleITK as sitk


def nib_to_sitk(nib_image: nib.Nifti1Image) -> sitk.Image:
    array = nib_image.get_fdata().astype(np.float32)
    array = np.transpose(array, (2, 1, 0))  # Nibabel: [x,y,z] → SITK: [z,y,x]

    sitk_image = sitk.GetImageFromArray(array)

    affine = nib_image.affine
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    origin = affine[:3, 3]

    direction = affine[:3, :3] / spacing
    direction = direction.flatten()

    sitk_image.SetSpacing(tuple(spacing))
    sitk_image.SetOrigin(tuple(origin))
    sitk_image.SetDirection(tuple(direction))

    return sitk_image


def sitk_to_nib(sitk_image: sitk.Image) -> nib.Nifti1Image:
    array = sitk.GetArrayFromImage(sitk_image)
    array = np.transpose(array, (2, 1, 0))

    spacing = np.array(sitk_image.GetSpacing())
    origin = np.array(sitk_image.GetOrigin())
    direction = np.array(sitk_image.GetDirection()).reshape(3, 3)

    affine = np.eye(4)
    affine[:3, :3] = direction * spacing
    affine[:3, 3] = origin

    return nib.Nifti1Image(array.astype(np.float32), affine)
