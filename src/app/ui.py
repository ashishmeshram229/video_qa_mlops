import streamlit as st
import requests
import zipfile

# If running via Docker Compose, use 'fastapi-backend'. 
# If running locally on Mac, change this to 'localhost'.
API_URL = "http://fastapi-backend:8000/predict" 

st.title("🏭 MVTec Quality Assurance Monitor")
st.write("Upload a product image or a ZIP file containing multiple images to detect defects.")

# File uploader accepts both single images and ZIP archives
uploaded_file = st.file_uploader("Upload Image or ZIP", type=["jpg", "jpeg", "png", "zip"])

if uploaded_file is not None:
    
    if uploaded_file.name.lower().endswith('.zip'):
        st.info("📦 ZIP file detected. Extracting and processing batch...")
        
        try:
            with zipfile.ZipFile(uploaded_file, 'r') as z:
                # Filter out folders and hidden Mac files
                image_names = [
                    n for n in z.namelist() 
                    if n.lower().endswith(('.png', '.jpg', '.jpeg')) and '__MACOSX' not in n
                ]
                
                if not image_names:
                    st.error("❌ No valid images found in the ZIP file!")
                else:
                    st.success(f"✅ Found {len(image_names)} images. Sending to Inference Engine...")
                    
                    # Create a 3-column grid for clean UI
                    cols = st.columns(3)
                    
                    for idx, img_name in enumerate(image_names):
                        with z.open(img_name) as img_file:
                            img_bytes = img_file.read()
                            
                            try:
                                # Send to FastAPI
                                files = {"file": (img_name, img_bytes, "image/jpeg")}
                                response = requests.post(API_URL, files=files, timeout=15)
                                
                                # Route to the correct column (0, 1, or 2)
                                col = cols[idx % 3]
                                
                                if response.status_code == 200:
                                    results = response.json()
                                    col.image(img_bytes, caption=f"Processed: {img_name}")
                                    col.json(results) # Display the bounding boxes/status
                                else:
                                    col.error(f"❌ Failed to process {img_name}. API Status: {response.status_code}")
                                    
                            except requests.exceptions.ConnectionError:
                                st.error("🚨 CRITICAL ERROR: Could not connect to FastAPI. Is the backend running?")
                                break # Stop the loop if the server is down
                                
        except zipfile.BadZipFile:
            st.error("❌ The uploaded file is not a valid ZIP archive or is corrupted.")

  
    else:
        st.image(uploaded_file, caption="Uploaded Image", width=400)
        
        if st.button("Run Inspection"):
            with st.spinner("Detecting defects..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "image/jpeg")}
                    response = requests.post(API_URL, files=files, timeout=10)
                    
                    if response.status_code == 200:
                        st.success("✅ Inspection Complete!")
                        st.json(response.json())
                    else:
                        st.error(f"❌ API Error: Status {response.status_code}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("🚨 CRITICAL ERROR: Could not connect to FastAPI. Is the backend container running?")