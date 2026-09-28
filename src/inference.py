"""
Inference & Web Deployment - FastAPI for Industrial Defect Segmentation
Production API for crack detection
"""

import torch
import numpy as np
from PIL import Image
import io
import cv2
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models import get_model

class SegmentationInference:
    def __init__(self, model_name='unet', encoder='resnet18', model_path=None, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model = get_model(model_name=model_name, num_classes=2, encoder=encoder)
        
        if model_path is None:
            models_dir = Path(__file__).parent.parent / "models"
            candidates = list(models_dir.glob(f"best_{model_name}*.pth"))
            if candidates:
                model_path = candidates[0]
        
        if model_path and Path(model_path).exists():
            print(f"Loading model from {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print(f"Model not found, using pretrained encoder only")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    
    def predict(self, image):
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        transformed = self.transform(image=image_np)
        img_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(img_tensor)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
        return pred
    
    def predict_with_overlay(self, image, alpha=0.5):
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        mask = self.predict(image_np)
        mask_resized = cv2.resize(mask.astype(np.uint8), (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        colored_mask = np.zeros_like(image_np)
        colored_mask[mask_resized == 1] = [255, 0, 0]  # Red for crack
        
        overlay = cv2.addWeighted(image_np, 1-alpha, colored_mask, alpha, 0)
        
        return image_np, mask_resized, overlay

# FastAPI App
try:
    from fastapi import FastAPI, File, UploadFile
    from fastapi.responses import StreamingResponse
    import uvicorn
    
    app = FastAPI(title="Industrial Defect Segmentation API", description="U-Net for crack detection", version="1.0")
    
    inference_model = None
    
    @app.on_event("startup")
    def load_model():
        global inference_model
        inference_model = SegmentationInference(model_name='unet', encoder='resnet18')
    
    @app.get("/")
    def root():
        return {"message": "Industrial Defect Segmentation API - U-Net DeepCrack", "models": ["unet", "deeplabv3plus", "fpn"], "status": "ready"}
    
    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        image = Image.open(io.BytesIO(await file.read())).convert('RGB')
        mask = inference_model.predict(image)
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        img_byte_arr = io.BytesIO()
        mask_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        return StreamingResponse(img_byte_arr, media_type="image/png")
    
    @app.post("/predict_overlay")
    async def predict_overlay(file: UploadFile = File(...)):
        image = Image.open(io.BytesIO(await file.read())).convert('RGB')
        original, mask, overlay = inference_model.predict_with_overlay(image)
        overlay_img = Image.fromarray(overlay.astype(np.uint8))
        img_byte_arr = io.BytesIO()
        overlay_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        return StreamingResponse(img_byte_arr, media_type="image/png")
    
    if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=8000)

except ImportError:
    print("FastAPI not installed")
    app = None

if __name__ == "__main__":
    from data_loader import get_dataloaders
    train_loader, val_loader = get_dataloaders(batch_size=2, img_size=256, root='../data')
    inference = SegmentationInference(model_name='unet', encoder='resnet18')
    for images, masks in val_loader:
        img_tensor = images[0]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_denorm = img_tensor * std + mean
        img_denorm = torch.clamp(img_denorm, 0, 1)
        img_np = (img_denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pred_mask = inference.predict(img_np)
        print(f"Predicted mask shape: {pred_mask.shape}, unique: {np.unique(pred_mask)}")
        break
