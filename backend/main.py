import os
import torch
import gdown
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

# Initialize app ONCE with full settings
app = FastAPI(title="Scalp Disorder AI Screening API", version="1.0")

# Enable CORS right here
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any origin (localhost:3000, etc.)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Initialize SQLite database table on startup
@app.on_event("startup")
def startup_event():
    init_db()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Load Model into Memory
# Load Model into Memory
# Load Model into Memory


# Load Model into Memory
MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "best_scalp_model.pth")

def download_model_if_not_exists():
    print(f"Checking model path at: {MODEL_PATH}")
    
    # Check karein ki file exist karti hai ya nahi, ya phir file ka size bohot chota (corrupt) toh nahi
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
        print("Model not found locally. Downloading from Google Drive...")
        
        file_id = "11X9HYJx97sspyaHwYkbkP3LXksipas6H"
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        try:
            gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
            if os.path.exists(MODEL_PATH):
                print(f"Download finished! File size: {os.path.getsize(MODEL_PATH)} bytes")
            else:
                print("Download failed: File still missing after gdown execution.")
        except Exception as e:
            print(f"Exception occurred during model download: {e}")
    else:
        print(f"Model already exists locally. Size: {os.path.getsize(MODEL_PATH)} bytes")

# Call it on startup
download_model_if_not_exists()
model = get_scalp_model(num_classes=len(CLASSES))

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"Loaded trained model checkpoint from {MODEL_PATH}")
else:
    print("Warning: Model checkpoint not found! Train the model first using src.train.")

model.to(DEVICE)
model.eval()

# Identify target convolutional layer for Grad-CAM in MobileNetV2
target_layer = model.features[-1]
grad_cam = GradCAM(model, target_layer)

# Image preprocessing pipeline for inference
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Mount static folder to serve uploaded images and heatmaps
os.makedirs("outputs/uploads", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# (Your @app.post("/predict/") and @app.get("/history/") routes follow below...)

@app.post("/predict/")
async def predict_scalp_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=400, detail="Model checkpoint not found. Please run training first.")
    
    # Save uploaded file temporarily
    file_path = os.path.join("outputs/uploads", file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
        
    # Open and transform image
    raw_img = Image.open(file_path).convert("RGB")
    input_tensor = transform(raw_img).unsqueeze(0).to(DEVICE)
    
    # Run prediction & Grad-CAM
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, pred_idx = torch.max(probabilities, dim=0)
        
    predicted_label = CLASSES[pred_idx.item()]
    conf_score = round(confidence.item(), 4) # Returned as decimal (e.g., 0.9250) for frontend calculation
    
    # Generate Grad-CAM heatmap
    heatmap, _ = grad_cam(input_tensor, class_idx=pred_idx.item())
    superimposed_img = generate_heatmap_overlay(file_path, heatmap)
    
    heatmap_filename = f"heatmap_{file.filename}"
    heatmap_path = os.path.join("outputs/uploads", heatmap_filename)
    import cv2
    cv2.imwrite(heatmap_path, superimposed_img)
    
    # Save record to database
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