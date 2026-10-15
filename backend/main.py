import os
import torch
from fastapi.middleware.cors import CORSMiddleware
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



@app.on_event("startup")
def startup_event():
    global model, grad_cam
    init_db()
    
    print(f"Checking model path at: {MODEL_PATH}")
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found. Attempting download from GitHub Release...")
        MODEL_DOWNLOAD_URL = "https://github.com/Asmit45986/ScalpSense-AI/releases/download/v1.0/best_scalp_model.pth"
        
        try:
            req = urllib.request.Request(
                MODEL_DOWNLOAD_URL, 
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req) as response, open(MODEL_PATH, 'wb') as out_file:
                out_file.write(response.read())
            print(f"Successfully downloaded model to {MODEL_PATH}! Size: {os.path.getsize(MODEL_PATH)} bytes")
        except Exception as e:
            print(f"CRITICAL ERROR downloading model: {str(e)}")

    # Load Model
    try:
        model = get_scalp_model(num_classes=len(CLASSES))
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            print(f"Loaded trained model checkpoint successfully from {MODEL_PATH}")
        else:
            print(f"Critical Error: Model file still does not exist at {MODEL_PATH}!")
    except Exception as e:
        print(f"Error loading model state dict: {str(e)}")
        
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Aap chahein toh specific Vercel URL bhi de sakte hain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
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