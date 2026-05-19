import io
import mlflow
from PIL import Image
from ultralytics import YOLO
from src.config.core import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class InferenceEngine:
    def __init__(self):
        logger.info("Connecting to MLflow Model Registry to fetch model...")
        
        # Connect to tracking server (Use host.docker.internal if running inside Docker to reach Mac localhost)
        mlflow.set_tracking_uri("http://host.docker.internal:5000")
        model_name = "YOLOv8m_Defect_Segmentation"
        
        try:
            # Dynamically pull the model currently in Staging
            logger.info(f"Attempting to download '{model_name}' from Staging...")
            model_uri = f"models:/{model_name}/Staging"
            local_model_path = mlflow.artifacts.download_artifacts(model_uri)
            
            # --- THE 1-LINE FIX IS HERE ---
            # Tell YOLO to look inside the downloaded MLflow folder for the .pt file
            self.model = YOLO(f"{local_model_path}/best.pt")
            
            logger.info("Successfully loaded MLflow Registered Model into memory.")
        except Exception as e:
            logger.warning(f"Failed to fetch model from MLflow Registry: {e}")
            logger.info("Falling back to local DVC best.pt...")
            
            # Fallback to local DVC tracked file
            local_fallback_path = "artifacts/models/yolo_seg_run/weights/best.pt"
            self.model = YOLO(local_fallback_path)

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

                # Extract Segmentation Mask
                mask_points = None
                if masks is not None and len(masks.xy) > i:
                    mask_points = masks.xy[i].tolist()

                formatted_results.append({
                    "class_name": class_name,
                    "confidence": conf,
                    "box": box,
                    "mask": mask_points
                })

        return formatted_results