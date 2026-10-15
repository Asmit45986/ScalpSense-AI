import os
import torch

import urllib.request

import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from src.config import DEVICE, SAVED_MODELS_DIR, CLASSES
from src.model import get_scalp_model
from src.gradcam import GradCAM, generate_heatmap_overlay
from backend.database import SessionLocal, init_db, ScanRecord
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Scalp Disorder AI Screening API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "best_scalp_model.pth")

model = None
grad_cam = None


# (baki imports wese hi rahenge)

@app.on_event("startup")
def startup_event():
    global model, grad_cam
    init_db()
    
    # Agar model file local folder me nahi milti (jaise Render par), toh download karein
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model checkpoint not found at {MODEL_PATH}! Downloading from GitHub Release...")
        os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
        
        # Yahan apna GitHub release ka direct download link paste kar diya hai:
        MODEL_DOWNLOAD_URL = "https://github.com/Asmit45986/ScalpSense-AI/releases/download/v1.0/best_scalp_model.pth"
        
        try:
            urllib.request.urlretrieve(MODEL_DOWNLOAD_URL, MODEL_PATH)
            print("Model successfully downloaded from GitHub!")
        except Exception as e:
            print(f"Failed to download model: {e}")

    # Load Model
    model = get_scalp_model(num_classes=len(CLASSES))
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Loaded trained model checkpoint from {MODEL_PATH}")
    else:
        print(f"Critical Error: Model checkpoint still missing!")
        
    model.to(DEVICE)
    model.eval()
    
    target_layer = model.features[-1]
    grad_cam = GradCAM(model, target_layer)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

os.makedirs("outputs/uploads", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

@app.post("/predict/")
async def predict_scalp_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=400, detail="Model checkpoint not found.")
    
    file_path = os.path.join("outputs/uploads", file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
        
    raw_img = Image.open(file_path).convert("RGB")
    input_tensor = transform(raw_img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, pred_idx = torch.max(probabilities, dim=0)
        
    predicted_label = CLASSES[pred_idx.item()]
    conf_score = round(confidence.item(), 4)
    
    heatmap, _ = grad_cam(input_tensor, class_idx=pred_idx.item())
    superimposed_img = generate_heatmap_overlay(file_path, heatmap)
    
    heatmap_filename = f"heatmap_{file.filename}"
    heatmap_path = os.path.join("outputs/uploads", heatmap_filename)
    import cv2
    cv2.imwrite(heatmap_path, superimposed_img)
    
    db_record = ScanRecord(
        filename=file.filename,
        predicted_class=predicted_label,
        confidence=round(conf_score * 100, 2)
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    
    return {
        "status": "success",
        "prediction": predicted_label,
        "confidence": conf_score,
        "image_url": f"/outputs/uploads/{file.filename}",
        "heatmap_url": f"/outputs/uploads/{heatmap_filename}"
    }

@app.get("/history/")
def get_scan_history(db: Session = Depends(get_db)):
    records = db.query(ScanRecord).order_by(ScanRecord.timestamp.desc()).all()
    return records