# src/models/inference.py
import io
from PIL import Image
from ultralytics import YOLO
from src.config.core import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class InferenceEngine:
    def __init__(self):
        # Pointing to the Colab model you forced into DVC
        self.model_path = config.MODEL_DIR / "yolo_defect_run" / "weights" / "best.pt"
        logger.info(f"Loading segmentation model from {self.model_path}")
        
        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError("Model weight file not found. Did you run dvc pull?")

    def predict(self, image_bytes: bytes) -> list:
        """Runs segmentation inference and extracts boxes and masks."""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Explicitly call the segment task
        results = self.model.predict(image, task='segment', conf=0.25)
        
        formatted_results = []
        for result in results:
            boxes = result.boxes
            masks = result.masks

            if boxes is None:
                continue

            for i in range(len(boxes)):
                box = boxes.xyxy[i].tolist()
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                class_name = self.model.names[cls_id]

                # Extract Segmentation Mask (if it exists)
                mask_points = None
                if masks is not None and len(masks.xy) > i:
                    # masks.xy is a list of numpy arrays representing the polygon
                    mask_points = masks.xy[i].tolist()

                formatted_results.append({
                    "class_name": class_name,
                    "confidence": conf,
                    "box": box,
                    "mask": mask_points
                })

        return formatted_results