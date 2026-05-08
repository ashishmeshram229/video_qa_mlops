# src/app/ui.py
import streamlit as st
import requests
import io
from PIL import Image, ImageDraw

API_URL = "http://fastapi-backend:8000" # Update to "http://localhost:8000" if running outside Docker

st.set_page_config(page_title="Industrial Defect Segmentation", layout="wide")

def draw_segmentation(image: Image.Image, detections: list) -> Image.Image:
    """Draws semi-transparent polygon masks and bounding boxes."""
    # Create an RGBA overlay for transparent masks
    overlay = image.convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")

    for det in detections:
        # Red for defect, Green for normal
        fill_color = (255, 0, 0, 90) if det['class_name'] == 'defect' else (0, 255, 0, 90)
        outline_color = (255, 0, 0, 255) if det['class_name'] == 'defect' else (0, 255, 0, 255)

        # 1. Draw the Polygon Mask
        if det.get('mask'):
            # Flatten the nested list [[x,y], [x,y]] -> [x, y, x, y] for PIL
            poly_points = [coord for point in det['mask'] for coord in point]
            if len(poly_points) >= 6: # Requires at least 3 points (triangle)
                draw.polygon(poly_points, fill=fill_color, outline=outline_color)

        # 2. Draw the Bounding Box & Text
        box = det['box']
        draw.rectangle((box[0], box[1], box[2], box[3]), outline=outline_color, width=2)
        draw.text((box[0], max(0, box[1] - 15)), f"{det['class_name']} ({det['confidence']:.2f})", fill=outline_color)

    # Composite the overlay onto the original image
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

st.title("🔍 MVTec Defect Segmentation Panel")
st.markdown("Upload a bottle image to run pixel-perfect anomaly detection.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display original
    original_image = Image.open(uploaded_file)
    st.image(original_image, caption="Uploaded Image", use_container_width=True)

    if st.button("Run Segmentation Inference"):
        with st.spinner("Analyzing pixels..."):
            # Send to FastAPI
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            response = requests.post(f"{API_URL}/predict", files=files)

            if response.status_code == 200:
                detections = response.json().get("detections", [])
                
                if not detections:
                    st.success("✅ Bottle passed inspection. No defects detected.")
                else:
                    st.error(f"⚠️ {len(detections)} anomaly mask(s) detected!")
                    
                    # Draw masks and show result
                    result_image = draw_segmentation(original_image, detections)
                    st.image(result_image, caption="Segmentation Map", use_container_width=True)
            else:
                st.error(f"Backend Error: {response.text}")