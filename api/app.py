"""
FastAPI application for U-Net segmentation - Enhanced Version
Supports: file upload, URL, local path, and batch processing
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel
from typing import List
import torch
import cv2
import numpy as np
import yaml
import requests
import base64
from pathlib import Path

import sys
sys.path.append('.')

from src.models.unet import SegmentationModel
from src.data.augmentation import get_valid_aug


app = FastAPI(title="U-Net Segmentation API - Enhanced")

model = None
device = None
config = None


class ImageURL(BaseModel):
    url: str


class ImagePath(BaseModel):
    path: str


class BatchImagePaths(BaseModel):
    paths: List[str]


def process_image(image_array):
    """Process a single image and return mask"""
    original_height, original_width = image_array.shape[:2]
    
    # Apply augmentation
    augmentation = get_valid_aug(config['data']['image_size'])
    augmented = augmentation(image=image_array)
    image_resized = augmented['image']
    
    # Prepare for model
    image_tensor = image_resized.astype(np.float32) / 255.0
    image_tensor = np.transpose(image_tensor, (2, 0, 1))
    image_tensor = torch.tensor(image_tensor, dtype=torch.float32).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    # Predict
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)
        pred_mask = (probs > 0.5).float()
    
    # Convert to numpy
    pred_mask = pred_mask.squeeze().cpu().numpy()
    pred_mask = cv2.resize(pred_mask, (original_width, original_height))
    pred_mask = (pred_mask * 255).astype(np.uint8)
    
    return pred_mask


@app.on_event("startup")
async def load_model():
    """Load model when API starts"""
    global model, device, config
    
    # Load config
    try:
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("Config loaded successfully")
    except FileNotFoundError:
        print("ERROR: configs/config.yaml not found")
        print("Make sure to run API from project root directory")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SegmentationModel(
        encoder=config['model']['encoder'],
        weights=config['model']['encoder_weights']
    )
    
    # Check for trained model
    checkpoint_path = 'checkpoints/best_model.pth'
    
    if not Path(checkpoint_path).exists():
        print("WARNING: No trained model found at checkpoints/best_model.pth")
        print("Using untrained model for API testing only")
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        print("Trained model loaded successfully!")
    
    model = model.to(device)
    model.eval()
    print("API ready!")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "message": "U-Net Segmentation API",
        "endpoints": {
            "/predict/upload": "Upload single image file",
            "/predict/url": "Provide image URL",
            "/predict/path": "Provide local file path",
            "/predict/batch": "Process multiple images from paths"
        }
    }


@app.post("/predict/upload")
async def predict_upload(file: UploadFile = File(...)):
    """Predict from uploaded file"""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        pred_mask = process_image(image)
        
        _, img_encoded = cv2.imencode('.png', pred_mask)
        return Response(content=img_encoded.tobytes(), media_type="image/png")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/url")
async def predict_url(image_data: ImageURL):
    """Predict from image URL"""
    try:
        response = requests.get(image_data.url)
        nparr = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        pred_mask = process_image(image)
        
        _, img_encoded = cv2.imencode('.png', pred_mask)
        return Response(content=img_encoded.tobytes(), media_type="image/png")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/path")
async def predict_path(image_data: ImagePath):
    """Predict from local file path"""
    try:
        if not Path(image_data.path).exists():
            raise HTTPException(status_code=404, detail="Image file not found")
        
        image = cv2.imread(image_data.path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        pred_mask = process_image(image)
        
        _, img_encoded = cv2.imencode('.png', pred_mask)
        return Response(content=img_encoded.tobytes(), media_type="image/png")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(batch_data: BatchImagePaths):
    """Predict from multiple local file paths"""
    try:
        results = []
        
        for image_path in batch_data.paths:
            if not Path(image_path).exists():
                results.append({
                    "path": image_path,
                    "status": "error",
                    "message": "File not found"
                })
                continue
            
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pred_mask = process_image(image)
            
            # Convert to base64 for JSON response
            _, img_encoded = cv2.imencode('.png', pred_mask)
            mask_base64 = base64.b64encode(img_encoded).decode('utf-8')
            
            results.append({
                "path": image_path,
                "status": "success",
                "mask_base64": mask_base64
            })
        
        return JSONResponse(content={"results": results})
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }
