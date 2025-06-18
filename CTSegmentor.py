import subprocess
from pathlib import Path
from loguru import logger


class CTSegmentator:
    
    DEFAULT_ROI_SUBSET = [
        "brain",
        "heart",
        "liver",
        "spleen",
        "kidney_left",
        "kidney_right",
        "urinary_bladder",
        "stomach",
        "small_bowel",
        "colon",
        "pancreas",
        "iliopsoas_left",
        "iliopsoas_right",
        "gluteus_maximus_left",
        "gluteus_maximus_right",
        "gluteus_medius_left",
        "gluteus_medius_right",
        "gluteus_minimus_left",
        "gluteus_minimus_right"
    ]

    def __init__(self, image_path, output_dir="segmentation_output", fast=True, roi_subset=None):
        self.image_path = Path(image_path)
        self.output_dir = Path(output_dir)
        self.fast = fast
        self.roi_subset = roi_subset or self.DEFAULT_ROI_SUBSET
        self.logger = logger.bind(component="CTSegmentator")
        if not self.image_path.exists():
            self.logger.error(f"File not found: {self.image_path}")
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        command = [
            "TotalSegmentator",
            "-i", str(self.image_path),
            "-o", str(self.output_dir)
        ]

        if self.fast:
            command.append("--fast")

        if self.roi_subset:
            command += ["--roi_subset"] + self.roi_subset

        self.logger.info(f"Launching TotalSegmentator with ROI subset of {len(self.roi_subset)} regions.")

        try:
            subprocess.run(command, check=True)
            self.logger.success(f"Segmentation completed. Results in: {self.output_dir}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Segmentation failed: {e}")
            if e.stderr:
                self.logger.debug(f"Error: {e.stderr}")