from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path

# Resolve base directory path dynamically
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    """
    Core settings for the ML application. Reads dynamically from the .env file.
    Validates all inputs to prevent runtime crashes.
    """
    # Project Identity
    PROJECT_NAME: str = Field(default="MVTec_Quality_Assurance_YOLO")
    ENVIRONMENT: str = Field(default="development")
    
    # Paths
    DATA_DIR: Path = Field(default=BASE_DIR / "data")
    MODEL_DIR: Path = Field(default=BASE_DIR / "artifacts" / "models")
    LOG_DIR: Path = Field(default=BASE_DIR / "logs")
    
    # MLOps & Experiment Tracking
    MLFLOW_TRACKING_URI: str = Field(default="http://localhost:5000")
    MLFLOW_EXPERIMENT_NAME: str = Field(default="mvtec_ad_bottle_inspection")
    
    # Inference Configuration
    INFERENCE_DEVICE: str = Field(default="cpu") # We will override this with 'mps' for your M4 later
    CONFIDENCE_THRESHOLD: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Monitoring & Alerting
    ALERT_ERROR_RATE_THRESHOLD: float = Field(default=0.05, ge=0.0, le=1.0)
    
    # API Configuration
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

config = Settings()