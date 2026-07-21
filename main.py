from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = FastAPI()

# Allow the HTML file to communicate with this Python server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# change to model file
model = YOLO(r"runs\detect\train3\weights\best.pt")

@app.post("/detect")
async def detect_ppe(file: UploadFile = File(...), confidence: float = Form(...)):
    # 1. Read the uploaded file into OpenCV
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. Run inference using your YOLO model
    results = model(img, conf=confidence)
    
    # 3. Get the image with bounding boxes drawn on it
    annotated_img = results[0].plot()

    # 4. Convert the annotated image to Base64 so HTML can display it
    _, buffer = cv2.imencode('.jpg', annotated_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    image_src = f"data:image/jpeg;base64,{img_base64}"

    # 5. Extract the list of detections to display in the UI sidebar
    detections = []
    for box in results[0].boxes:
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id]
        conf = float(box.conf[0].item())
        
        detections.append({
            "class": class_name,
            "conf": conf
        })

    # Return everything to the frontend
    return {
        "imageSrc": image_src,
        "detections": detections
    }