import subprocess
from pathlib import Path
from loguru import logger


DEFAULT_ROI_SUBSET = [
    "brain", "heart", "liver", "spleen", "kidney_left", "kidney_right", "urinary_bladder",
    "stomach", "small_bowel", "colon", "pancreas",
    "iliopsoas_left", "iliopsoas_right",
    "gluteus_maximus_left", "gluteus_maximus_right",
    "gluteus_medius_left", "gluteus_medius_right",
    "gluteus_minimus_left", "gluteus_minimus_right"
]


def run_ct_segmentation(image_path: Path, output_dir: Path = None, fast: bool = True, roi_subset: list[str] = None):
    image_path = Path(image_path)
    if not image_path.exists():
        logger.error(f"File not found: {image_path}")
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    if output_dir is None:
        output_dir = "segmentation_output"
        output_path = image_path.parent / output_dir

    output_path.mkdir(parents=True, exist_ok=True)

    selected_rois = roi_subset or DEFAULT_ROI_SUBSET

    command = [
        "TotalSegmentator",
        "-i", str(image_path),
        "-o", str(output_path)
    ]

    if fast:
        command.append("--fast")

    if selected_rois:
        command += ["--roi_subset"] + selected_rois

    logger.info(f"Launching TotalSegmentator with ROI subset of {len(selected_rois)} regions...")

    try:
        subprocess.run(command, check=True)
        logger.success(f"Segmentation completed. Results in: {output_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Segmentation failed: {e}")
        if e.stderr:
            logger.debug(f"Error details: {e.stderr}")


# ==== Exemple d'utilisation ====
run_ct_segmentation(image_path=Path("processed_data/Agathe/CT_baseline_resampled.nii.gz"))   