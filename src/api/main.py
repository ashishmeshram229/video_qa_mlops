# src/api/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from prometheus_fastapi_instrumentator import Instrumentator ### NEW

from src.models.inference import InferenceEngine

app = FastAPI(title="MVTec Defect Segmentation API", version="2.0")
engine = InferenceEngine()
# Expose /metrics endpoint for Prometheus
Instrumentator().instrument(app).expose(app) ### NEW
# --- Pydantic Schemas for Segmentation ---
class Detection(BaseModel):
    class_name: str
    confidence: float
    box: List[float]
    mask: Optional[List[List[float]]] = None  # Polygon coordinates: [[x1,y1], [x2,y2], ...]

class DetectionResponse(BaseModel):
    detections: List[Detection]

# --- Endpoints ---
@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "yolov8m-seg"}

@app.post("/predict", response_model=DetectionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    try:
        image_bytes = await file.read()
        detections = engine.predict(image_bytes)
        return {"detections": detections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)