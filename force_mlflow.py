import os
import shutil
import yaml
import mlflow
from mlflow.tracking.client import MlflowClient

# Connect to your local MLflow server
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("YOLOv8_Segmentation_Pipeline")

# Paths
model_path = "artifacts/models/best.pt"
model_name = "YOLOv8m_Defect_Segmentation"
params_path = "params.yaml"
colab_results_dir = "artifacts/colab_results"

print("🚀 Starting Complete MLflow Packaging...")

# 1. Create a temporary staging directory
staging_dir = "mlflow_staging"
os.makedirs(staging_dir, exist_ok=True)

# 2. Copy the .pt file into it
shutil.copy(model_path, os.path.join(staging_dir, "best.pt"))

# 3. Create the required 'MLmodel' file
with open(os.path.join(staging_dir, "MLmodel"), "w") as f:
    f.write('artifact_path: yolo_model\nflavors:\n  python_function:\n    loader_module: mlflow.pyfunc\n')

with mlflow.start_run() as run:
    
    # ==========================================
    # 1. LOG PARAMETERS (Dynamically from params.yaml)
    # ==========================================
    print("⚙️ Logging parameters from params.yaml...")
    if os.path.exists(params_path):
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)["train"]
            mlflow.log_params(params)
    else:
        print("⚠️ params.yaml not found. Skipping parameter logging.")

    # ==========================================
    # 2. LOG METRICS (Update these to match your Colab run)
    # ==========================================
    print("📊 Logging performance metrics...")
    mlflow.log_metric("mAP50_box", 0.925)       
    mlflow.log_metric("mAP50-95_box", 0.881)    
    mlflow.log_metric("mAP50_seg", 0.910)       
    mlflow.log_metric("mAP50-95_seg", 0.865)    

    # ==========================================
    # 3. LOG ARTIFACTS (Graphs & Predictions)
    # ==========================================
    if os.path.exists(colab_results_dir) and len(os.listdir(colab_results_dir)) > 0:
        print("📈 Logging Colab graphs and prediction images...")
        mlflow.log_artifacts(colab_results_dir, artifact_path="training_visuals")
    else:
        print("⚠️ No images found in artifacts/colab_results/. Skipping visuals logging.")

    # ==========================================
    # 4. PACKAGE AND REGISTER THE MODEL
    # ==========================================
    print("📦 Uploading model file...")
    mlflow.log_artifacts(staging_dir, artifact_path="yolo_model")
    
    print("🏷️ Registering model to Model Registry...")
    model_uri = f"runs:/{run.info.run_id}/yolo_model"
    reg_model = mlflow.register_model(model_uri=model_uri, name=model_name)
    
    print("🚀 Promoting model to Staging...")
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=reg_model.version,
        stage="Staging"
    )

# Clean up the temp folder
shutil.rmtree(staging_dir)
print(f"✅ Success! Model version {reg_model.version} is completely logged and in Staging.")