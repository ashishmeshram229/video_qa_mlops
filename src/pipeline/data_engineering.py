import os
import tarfile
import cv2
from pathlib import Path
from pydantic import ValidationError

from src.utils.logger import get_logger
from src.config.core import config
from src.pipeline.schema import YoloAnnotationSchema

logger = get_logger(__name__)

class MVTecDataPipeline:
    def __init__(self):
        self.raw_archive = config.DATA_DIR / "raw" / "bottle.tar.xz"
        self.extract_path = config.DATA_DIR / "raw" / "mvtec_bottle"
        self.interim_img_dir = config.DATA_DIR / "interim" / "images"
        self.interim_lbl_dir = config.DATA_DIR / "interim" / "labels"
        
        self.valid_files = 0
        self.bbox_areas = []

    def extract_data(self):
        """Extracts the MVTec archive if not already extracted."""
        if not self.raw_archive.exists():
            raise FileNotFoundError(f"Missing raw data at {self.raw_archive}")
            
        if not self.extract_path.exists():
            logger.info("Extracting bottle.tar.xz...")
            with tarfile.open(self.raw_archive, "r:xz") as tar:
                tar.extractall(path=self.extract_path)
            logger.info("Extraction complete.")
        else:
            logger.info("Data already extracted. Skipping.")

    def transform_and_validate(self):
        """Converts masks to YOLO format and strictly validates them."""
        logger.info("Starting transformation and schema validation...")
        self.interim_img_dir.mkdir(parents=True, exist_ok=True)
        self.interim_lbl_dir.mkdir(parents=True, exist_ok=True)

        ground_truth_dir = self.extract_path / "bottle" / "ground_truth"
        test_images_dir = self.extract_path / "bottle" / "test"
        
        if not ground_truth_dir.exists():
            raise FileNotFoundError("Ground truth directory not found.")

        for defect_category in ground_truth_dir.iterdir():
            if not defect_category.is_dir(): continue
                
            for mask_file in defect_category.glob("*.png"):
                base_name = mask_file.stem.replace("_mask", "")
                image_file = test_images_dir / defect_category.name / f"{base_name}.png"
                
                if not image_file.exists(): continue

                # Extract bounding boxes using OpenCV
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                img_height, img_width = mask.shape
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                yolo_labels = []
                is_valid = True
                
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if w < 5 or h < 5: continue # Ignore noise
                        
                    norm_x = (x + w / 2) / img_width
                    norm_y = (y + h / 2) / img_height
                    norm_w = w / img_width
                    norm_h = h / img_height
                    
                    try:
                        # STRICT VALIDATION: Ensure coordinates are valid via Pydantic
                        box = YoloAnnotationSchema(
                            class_id=1, x_center=norm_x, y_center=norm_y, width=norm_w, height=norm_h
                        )
                        yolo_labels.append(f"{box.class_id} {box.x_center:.6f} {box.y_center:.6f} {box.width:.6f} {box.height:.6f}")
                        self.bbox_areas.append(box.width * box.height)
                    except ValidationError as e:
                        logger.warning(f"Validation failed for {mask_file.name}: {e}")
                        is_valid = False

                if yolo_labels and is_valid:
                    new_img_path = self.interim_img_dir / f"{defect_category.name}_{base_name}.png"
                    new_lbl_path = self.interim_lbl_dir / f"{defect_category.name}_{base_name}.txt"
                    
                    cv2.imwrite(str(new_img_path), cv2.imread(str(image_file)))
                    with open(new_lbl_path, "w") as f:
                        f.write("\n".join(yolo_labels))
                    self.valid_files += 1

    def calculate_baselines(self):
        """Calculates statistical baselines for future drift detection."""
        if self.bbox_areas:
            avg_area = sum(self.bbox_areas) / len(self.bbox_areas)
            logger.info(f"PIPELINE SUCCESS: Processed {self.valid_files} valid images.")
            logger.info(f"DRIFT BASELINE: Average bounding box area is {avg_area:.6f}")

    def run(self):
        self.extract_data()
        self.transform_and_validate()
        self.calculate_baselines()

if __name__ == "__main__":
    pipeline = MVTecDataPipeline()
    pipeline.run()