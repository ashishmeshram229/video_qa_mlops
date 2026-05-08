from pydantic import BaseModel, Field
from typing import List

class BoundingBox(BaseModel):
    """Pydantic schema for a single detected object."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_id: int
    class_name: str

class DetectionResponse(BaseModel):
    """Strict schema for the JSON response returned to the UI."""
    status: str
    filename: str
    detections: List[BoundingBox]
    inference_time_ms: float