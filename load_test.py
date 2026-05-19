import requests
import time
from concurrent.futures import ThreadPoolExecutor

# Your FastAPI endpoint
API_URL = "http://127.0.0.1:8000/predict"

# ⚠️ IMPORTANT: Change this to the actual path of an image on your computer!
REAL_IMAGE_PATH = "/Users/ashishmeshram/Documents/IITM - ACADEMICS/SEM2/COURSES/MLops/PROJECT/video_qa_mlops/data/interim/images/broken_large_000.png" # Example path, update this!

def send_request(request_id):
    """Sends a real image to the YOLO inference API."""
    try:
        # Open the real image file
        with open(REAL_IMAGE_PATH, "rb") as image_file:
            files = {"file": ("test_image.png", image_file, "image/png")}
            
            # Send the POST request to FastAPI
            res = requests.post(API_URL, files=files, timeout=10) # Increased timeout just in case
            print(f"📷 Camera {request_id} processed | Status: {res.status_code}")
            
            if res.status_code != 200:
                print(f"   Error Details: {res.text}")
                
    except FileNotFoundError:
        print(f"❌ ERROR: Could not find the image at {REAL_IMAGE_PATH}. Please update the path.")
    except Exception as e:
        print(f"❌ Camera {request_id} failed: {e}")

print("🚀 Initiating Factory Floor Load Test...")
print("Switch over to your Grafana Dashboard NOW to watch the spike!")

# Let's do 20 concurrent requests first to make sure it works without crashing
with ThreadPoolExecutor(max_workers=5) as executor:
    for i in range(1, 21): 
        executor.submit(send_request, i)
        time.sleep(0.2) 

print("✅ Load Test Complete!")