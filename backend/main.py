import os
import gc
import uuid
import logging
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import torch
import torchvision.transforms as transforms

from PIL import Image, UnidentifiedImageError
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Depends,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from src.config import DEVICE, SAVED_MODELS_DIR, CLASSES
from src.model import get_scalp_model
from src.gradcam import GradCAM, generate_heatmap_overlay
from backend.database import SessionLocal, init_db, ScanRecord


# =========================================================
# CONFIGURATION
# =========================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(
    SAVED_MODELS_DIR,
    "best_scalp_model.pth",
)

MODEL_DOWNLOAD_URL = (
    "https://github.com/Asmit45986/ScalpSense-AI/"
    "releases/download/v1.0/best_scalp_model.pth"
)

OUTPUT_DIR = Path("outputs")
UPLOAD_DIR = OUTPUT_DIR / "uploads"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

model = None
grad_cam = None

# Set this to your actual frontend URL in production.
# Example: ["https://your-app.vercel.app"]
FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "*",
    ).split(",")
    if origin.strip()
]


# =========================================================
# IMAGE TRANSFORM
# =========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# =========================================================
# MODEL DOWNLOAD
# =========================================================

def download_model():
    """Download the trained checkpoint if it is missing."""

    if os.path.exists(MODEL_PATH):
        logger.info("Model checkpoint already exists: %s", MODEL_PATH)
        return

    logger.info("Downloading trained model checkpoint...")

    temp_path = MODEL_PATH + ".part"

    try:
        request = urllib.request.Request(
            MODEL_DOWNLOAD_URL,
            headers={"User-Agent": "ScalpSense-AI/1.0"},
        )

        with urllib.request.urlopen(request, timeout=120) as response:
            with open(temp_path, "wb") as output_file:
                while True:
                    chunk = response.read(1024 * 1024)

                    if not chunk:
                        break

                    output_file.write(chunk)

        if os.path.getsize(temp_path) == 0:
            raise RuntimeError("Downloaded model checkpoint is empty.")

        os.replace(temp_path, MODEL_PATH)

        logger.info(
            "Model downloaded successfully. Size: %s bytes",
            os.path.getsize(MODEL_PATH),
        )

    except Exception as exc:
        if os.path.exists(temp_path):
            os.remove(temp_path)

        raise RuntimeError(
            f"Unable to download model checkpoint: {exc}"
        ) from exc


# =========================================================
# MODEL INITIALIZATION
# =========================================================

def initialize_model():
    """Load the trained model and initialize Grad-CAM."""

    global model, grad_cam

    download_model()

    logger.info("Initializing model on device: %s", DEVICE)

    loaded_model = get_scalp_model(
        num_classes=len(CLASSES)
    )

    try:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True,
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
        )

    # Support checkpoints saved as either a state_dict
    # or a dictionary containing model_state_dict.
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

    loaded_model.load_state_dict(checkpoint)
    loaded_model.to(DEVICE)
    loaded_model.eval()

    # Confirm that this layer matches your model architecture.
    if not hasattr(loaded_model, "features"):
        raise RuntimeError(
            "Model has no 'features' attribute. "
            "Update the Grad-CAM target layer for your architecture."
        )

    target_layer = loaded_model.features[-1]
    loaded_grad_cam = GradCAM(
        loaded_model,
        target_layer,
    )

    model = loaded_model
    grad_cam = loaded_grad_cam

    logger.info("Model and Grad-CAM initialized successfully.")


# =========================================================
# APPLICATION LIFESPAN
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and model when the app starts."""

    global model, grad_cam

    init_db()

    try:
        initialize_model()
    except Exception:
        logger.exception("Model initialization failed.")
        raise

    yield

    # Release references when the application shuts down.
    grad_cam = None
    model = None

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Scalp Disorder AI Screening API",
    description=(
        "AI-assisted scalp image screening with "
        "prediction, Grad-CAM visualization, and scan history."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Wildcard origins cannot be combined safely with credentials.
# Use explicit frontend origins if cookies or credentials are needed.
allow_credentials = FRONTEND_ORIGINS != ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded images and generated heatmaps.
app.mount(
    "/outputs",
    StaticFiles(directory=str(OUTPUT_DIR)),
    name="outputs",
)


# =========================================================
# DATABASE DEPENDENCY
# =========================================================

def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/")
def root():
    return {
        "status": "success",
        "message": "ScalpSense AI API is running.",
        "docs": "/docs",
    }


@app.get("/health/")
def health_check():
    return {
        "status": "healthy" if model is not None else "not_ready",
        "model_loaded": model is not None,
        "gradcam_loaded": grad_cam is not None,
        "device": str(DEVICE),
    }


# =========================================================
# IMAGE PREDICTION
# =========================================================

@app.post("/predict/")
async def predict_scalp_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Predict a scalp image class and generate a Grad-CAM overlay.
    """

    if model is None or grad_cam is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not initialized.",
        )

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )

    allowed_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    original_name = Path(file.filename).name
    extension = Path(original_name).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload JPG, JPEG, PNG, or WEBP.",
        )

    # Use a unique filename to prevent path traversal and overwrites.
    unique_id = uuid.uuid4().hex
    image_filename = f"{unique_id}{extension}"
    heatmap_filename = f"heatmap_{unique_id}.jpg"

    image_path = UPLOAD_DIR / image_filename
    heatmap_path = UPLOAD_DIR / heatmap_filename

    image_bytes = await file.read(MAX_FILE_SIZE + 1)

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum allowed size is 10 MB.",
        )

    input_tensor = None
    outputs = None
    probabilities = None
    heatmap = None

    try:
        # Validate that the uploaded file is a readable image.
        try:
            import io

            with Image.open(io.BytesIO(image_bytes)) as img:
                img.verify()

            with Image.open(io.BytesIO(image_bytes)) as img:
                raw_img = img.convert("RGB")

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
        ) as exc:
            raise HTTPException(
                status_code=400,
                detail="Invalid or corrupted image file.",
            ) from exc

        # Persist the original image.
        with open(image_path, "wb") as buffer:
            buffer.write(image_bytes)

        input_tensor = transform(raw_img).unsqueeze(0).to(DEVICE)

        # Prediction does not need gradients.
        with torch.inference_mode():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence_tensor, pred_idx_tensor = torch.max(
                probabilities,
                dim=0,
            )

            pred_idx = int(pred_idx_tensor.item())
            confidence = float(confidence_tensor.item())

        if pred_idx >= len(CLASSES):
            raise RuntimeError(
                "Predicted class index exceeds the configured class list."
            )

        predicted_label = CLASSES[pred_idx]
        confidence_percent = round(confidence * 100, 2)

        # Grad-CAM needs gradients, so run outside inference_mode.
        with torch.enable_grad():
            heatmap, _ = grad_cam(
                input_tensor,
                class_idx=pred_idx,
            )

        # Generate the visualization from the saved original image.
        superimposed_img = generate_heatmap_overlay(
            str(image_path),
            heatmap,
        )

        if superimposed_img is None:
            raise RuntimeError("Grad-CAM overlay generation failed.")

        if not cv2.imwrite(
            str(heatmap_path),
            superimposed_img,
        ):
            raise RuntimeError("Unable to save Grad-CAM heatmap.")

        # Save scan metadata.
        db_record = ScanRecord(
            filename=image_filename,
            predicted_class=predicted_label,
            confidence=confidence_percent,
        )

        db.add(db_record)
        db.commit()
        db.refresh(db_record)

        return {
            "status": "success",
            "prediction": predicted_label,
            "confidence": confidence,
            "confidence_percent": confidence_percent,
            "image_url": f"/outputs/uploads/{image_filename}",
            "heatmap_url": f"/outputs/uploads/{heatmap_filename}",
            "scan_id": getattr(db_record, "id", None),
            "message": (
                "AI screening result generated successfully. "
                "This result is not a medical diagnosis."
            ),
        }

    except HTTPException:
        raise

    except Exception as exc:
        db.rollback()

        # Avoid exposing internal server errors to API clients.
        logger.exception("Prediction failed.")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the server logs.",
        ) from exc

    finally:
        if input_tensor is not None:
            del input_tensor

        if outputs is not None:
            del outputs

        if probabilities is not None:
            del probabilities

        if heatmap is not None:
            del heatmap

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        await file.close()


# =========================================================
# SCAN HISTORY
# =========================================================

@app.get("/history/")
def get_scan_history(
    db: Session = Depends(get_db),
):
    """Return recent scans, newest first."""

    try:
        records = (
            db.query(ScanRecord)
            .order_by(ScanRecord.timestamp.desc())
            .limit(100)
            .all()
        )

        return [
            {
                "id": getattr(record, "id", None),
                "filename": record.filename,
                "predicted_class": record.predicted_class,
                "confidence": record.confidence,
                "timestamp": (
                    record.timestamp.isoformat()
                    if record.timestamp is not None
                    else None
                ),
                "image_url": (
                    f"/outputs/uploads/{record.filename}"
                ),
            }
            for record in records
        ]

    except Exception as exc:
        logger.exception("Failed to fetch scan history.")

        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve scan history.",
        ) from exc