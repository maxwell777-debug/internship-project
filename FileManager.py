from pathlib import Path
import shutil
import nibabel as nib
from loguru import logger


class FileManager:
    def __init__(self):
        self.logger = logger.bind(name="FileManager")
    
    def delete_json_files(self, folder: Path):
        self.logger.info(f"Deleting JSON files in {folder.name}...")
        for file in folder.glob("*.json"):
            try:
                file.unlink()
            except Exception as e:
                self.logger.warning(f"Failed to delete {file.name}: {e}")

    def delete_subfolders(self, folder: Path):
        self.logger.info(f"Deleting subdirectories in {folder.name}...")
        for subdir in folder.iterdir():
            if subdir.is_dir():
                try:
                    shutil.rmtree(subdir)
                except Exception as e:
                    self.logger.warning(f"Failed to remove directory {subdir.name}: {e}")
    
    def clean_invalid_nii_files(self, folder: Path):
        self.logger.info(f"Cleaning redundant NIfTI files in {folder.name}...")

        base_names = ["CT_baseline", "CT_normal", "PET_baseline", "PET_normal"]

        for base in base_names:
            nii_files = list(folder.glob(f"{base}*.nii*"))
            if len(nii_files) <= 1:
                continue
            best_file = None
            max_voxels = 0

            for file in nii_files:
                try:
                    img = nib.load(str(file))
                    shape = img.shape
                    num_voxels = 1
                    for dim in shape:
                        num_voxels *= dim

                    if num_voxels > max_voxels:
                        max_voxels = num_voxels
                        best_file = file

                except Exception as e:
                    self.logger.warning(f"Failed to read {file.name}: {e}")

            for file in nii_files:
                if file != best_file:
                    try:
                        file.unlink()
                        self.logger.info(f"Deleted redundant file: {file.name}")
                    except Exception as e:
                        self.logger.warning(f"Failed to delete {file.name}: {e}")
            clean_name = base + ".nii.gz"
            clean_path = folder / clean_name
            if best_file.name != clean_name:
                self.logger.info(f"Renaming {best_file.name} to {clean_name}")
                best_file.rename(clean_path)
