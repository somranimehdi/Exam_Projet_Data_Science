from fastapi import FastAPI, HTTPException,UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch

from Backend.Model.inference import load_model,inference
from Backend.Model.TrafficSignClassifier import TrafficSignClassifier

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = "C:\\Users\\Mehdi\\Desktop\\TrafficSignsProject-main\\TrafficSignsProject-main\\Backend\\Model\\traffic_sign_classifier.pth"
print(model_path)
num_classes = 15
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(TrafficSignClassifier, model_path, num_classes, device)

@app.get("/")

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    if not file.filename.endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .jpg, .jpeg, or .png file.")
    try:
        prediction,confidence = inference(model, file, device)
        print(confidence)
        return {"label": prediction, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the file: {str(e)}")


