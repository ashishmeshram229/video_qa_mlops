import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import mlflow
from mlflow.tracking.client import MlflowClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ModelTrainer:
    def __init__(self):
        self.params_path = Path("params.yaml")
        self.data_yaml_path = Path("data/interim/data.yaml") # Ensure your data.yaml is here!
        self.model_output_dir = Path("artifacts/models")
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.params_path, "r") as f:
            self.params = yaml.safe_load(f)["train"]

    def train_and_register(self):
        logger.info(f"Starting YOLOv8 Segmentation training...")
        
        # Connect to local MLflow tracking server
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("YOLOv8_Segmentation_Pipeline")

        with mlflow.start_run() as run:
            # 1. Log Hyperparameters to MLflow
            mlflow.log_params(self.params)

            # 2. Train Model (Uses MPS for Apple Silicon. Change to device=0 if on Colab)
            model = YOLO(self.params['model_name'])
            results = model.train(
                data=str(self.data_yaml_path),
                epochs=self.params['epochs'],
                patience=self.params['patience'],
                batch=self.params['batch'],
                imgsz=self.params['imgsz'],
                project=str(self.model_output_dir),
                name="yolo_seg_run",
                device="mps", 
                exist_ok=True
            )

            # 3. Evaluate and Register
            self._evaluate_and_register(run.info.run_id, results)

    def _evaluate_and_register(self, run_id, results):
        client = MlflowClient()
        model_name = "YOLOv8m_Defect_Segmentation"
        target_map = 0.85 # The threshold the model must beat to go to production
        
        # Extract mAP score from YOLO results safely
        current_map = results.seg.map if hasattr(results, 'seg') else getattr(results.box, 'map', 0.0)

        logger.info(f"Current mAP: {current_map:.4f}, Target: {target_map}")
        
        # Log metrics and artifacts (graphs, confusion matrices) to MLflow
        mlflow.log_metric("mAP50-95_seg", current_map)
        mlflow.log_artifacts(str(self.model_output_dir / "yolo_seg_run"), artifact_path="yolo_artifacts")

        # The CI/CD Gate
        if current_map >= target_map:
            logger.info("Threshold met! Registering to Staging...")
            model_uri = f"runs:/{run_id}/yolo_artifacts/weights/best.pt"
            reg_model = mlflow.register_model(model_uri=model_uri, name=model_name)
            
            client.transition_model_version_stage(
                name=model_name, version=reg_model.version, stage="Staging"
            )
            logger.info(f"Successfully deployed Model v{reg_model.version} to Staging.")
        else:
            logger.warning(f"Model mAP ({current_map:.4f}) below threshold. Not registering.")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_and_register()