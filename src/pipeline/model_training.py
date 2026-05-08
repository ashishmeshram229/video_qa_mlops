import os
import shutil
import random
import yaml
from pathlib import Path
from ultralytics import YOLO
import mlflow

from src.utils.logger import get_logger
from src.config.core import config

logger = get_logger(__name__)

class YoloTrainingPipeline:
    def __init__(self):
        # I/O Paths
        self.interim_img_dir = config.DATA_DIR / "interim" / "images"
        self.interim_lbl_dir = config.DATA_DIR / "interim" / "labels"
        
        self.processed_dir = config.DATA_DIR / "processed"
        self.train_img_dir = self.processed_dir / "images" / "train"
        self.val_img_dir = self.processed_dir / "images" / "val"
        self.train_lbl_dir = self.processed_dir / "labels" / "train"
        self.val_lbl_dir = self.processed_dir / "labels" / "val"
        
        self.yaml_path = self.processed_dir / "data.yaml"
        self.model_output_dir = config.MODEL_DIR

        # LOAD PARAMS FROM YAML FOR DVC REPRODUCIBILITY
        with open("params.yaml", "r") as f:
            self.params = yaml.safe_load(f)["train"]

    def setup_and_split_data(self):
        """Creates train/val splits required by YOLO using dynamic parameters."""
        split_ratio = self.params["split_ratio"]
        logger.info(f"Setting up processed data directories and splitting data with ratio {split_ratio}...")
        
        if self.processed_dir.exists():
            shutil.rmtree(self.processed_dir)
            
        for d in [self.train_img_dir, self.val_img_dir, self.train_lbl_dir, self.val_lbl_dir, self.model_output_dir]:
            d.mkdir(parents=True, exist_ok=True)

        images = list(self.interim_img_dir.glob("*.png")) + list(self.interim_img_dir.glob("*.jpg"))
        if not images:
            raise FileNotFoundError("No images found in the interim directory!")

        random.seed(42)
        random.shuffle(images)
        
        train_size = int(len(images) * split_ratio)
        
        def copy_files(img_list, dest_img, dest_lbl):
            for img in img_list:
                lbl = self.interim_lbl_dir / f"{img.stem}.txt"
                if lbl.exists():
                    shutil.copy(img, dest_img / img.name)
                    shutil.copy(lbl, dest_lbl / lbl.name)

        copy_files(images[:train_size], self.train_img_dir, self.train_lbl_dir)
        copy_files(images[train_size:], self.val_img_dir, self.val_lbl_dir)
        logger.info(f"Split complete: {train_size} training, {len(images) - train_size} validation samples.")

    def generate_yaml(self):
        """Generates the data.yaml configuration for YOLO."""
        yaml_content = {
            "path": str(self.processed_dir.absolute()),
            "train": "images/train",
            "val": "images/val",
            "names": {0: "normal", 1: "defect"}
        }
        with open(self.yaml_path, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

    def train(self):
        """Trains the YOLOv8 model using Apple Silicon (MPS), MLflow, and dynamic params."""
        logger.info(f"Initializing YOLO training with {self.params['model_name']}...")
        
        # Link MLflow
        os.environ["MLFLOW_TRACKING_URI"] = config.MLFLOW_TRACKING_URI
        os.environ["MLFLOW_EXPERIMENT_NAME"] = config.MLFLOW_EXPERIMENT_NAME

        # Load dynamic model name (e.g., yolov8m.pt)
        model = YOLO(self.params["model_name"]) 
        
        # Train model using 'mps' device and dynamic hyper-parameters
        model.train(
            data=str(self.yaml_path),
            epochs=self.params["epochs"],
            patience=self.params["patience"],
            batch=self.params["batch"],
            imgsz=self.params["imgsz"],
            device='mps', 
            project=str(self.model_output_dir),
            name="yolo_defect_run",
            exist_ok=True,
            plots=True # Ensures confusion matrices and F1 curves are generated
        )
        logger.info("Training complete. Artifacts logged to local directory and MLflow.")

    def run(self):
        self.setup_and_split_data()
        self.generate_yaml()
        self.train()

if __name__ == "__main__":
    pipeline = YoloTrainingPipeline()
    pipeline.run()