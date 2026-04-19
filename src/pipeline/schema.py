from pydantic import BaseModel, Field, field_validator

class YoloAnnotationSchema(BaseModel):
    """
    Pydantic schema to validate YOLO format bounding box annotations.
    Strictly enforces that all bounding box coordinates are normalized (0.0 to 1.0).
    """
    class_id: int = Field(..., ge=0, description="Integer representing the defect class.")
    x_center: float = Field(..., ge=0.0, le=1.0)
    y_center: float = Field(..., ge=0.0, le=1.0)
    width: float = Field(..., gt=0.0, le=1.0)
    height: float = Field(..., gt=0.0, le=1.0)

    @field_validator('width', 'height')
    def check_dimensions(cls, value):
        """Ensures bounding boxes aren't microscopic artifacts."""
        if value <= 0.001:
            raise ValueError(f"Bounding box dimension ({value}) is too small; likely an artifact.")
        return value