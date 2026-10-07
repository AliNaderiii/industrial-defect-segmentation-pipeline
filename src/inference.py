"""Safe FastAPI inference for a locally trained DeepCrack checkpoint."""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from .evaluate import load_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best.pt"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))


class CrackInference:
    """Perform original-size crack-mask inference from a valid local checkpoint."""

    def __init__(self, checkpoint_path: str | Path = DEFAULT_CHECKPOINT) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"No trained checkpoint at {self.checkpoint_path}. "
                "Train the model before requesting predictions."
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, payload = load_checkpoint(self.checkpoint_path, self.device)
        self.image_size = int(payload["dataset"]["image_size"])
        self.model_name = payload["model"]["name"]
        self.encoder = payload["model"]["encoder"]

    def predict(self, image: Image.Image | np.ndarray) -> np.ndarray:
        if isinstance(image, Image.Image):
            rgb = np.asarray(image.convert("RGB"))
        else:
            rgb = np.asarray(image)
            if rgb.ndim != 3 or rgb.shape[2] not in {3, 4}:
                raise ValueError("Expected an RGB or RGBA image")
            rgb = rgb[:, :, :3]
        height, width = rgb.shape[:2]
        resized = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            labels = self.model(tensor).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        return cv2.resize(labels, (width, height), interpolation=cv2.INTER_NEAREST)

    def predict_overlay(self, image: Image.Image | np.ndarray, alpha: float = 0.45) -> np.ndarray:
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        rgb = np.asarray(image.convert("RGB")) if isinstance(image, Image.Image) else np.asarray(image)[:, :, :3]
        mask = self.predict(rgb)
        red = np.zeros_like(rgb)
        red[mask == 1] = (255, 0, 0)
        return cv2.addWeighted(rgb, 1 - alpha, red, alpha, 0)


try:
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import StreamingResponse

    app = FastAPI(
        title="DeepCrack Segmentation Reference API",
        description="Serves a local, trained binary crack-segmentation checkpoint only.",
        version="2.0.0",
    )
    _engine: CrackInference | None = None
    _load_error: str | None = None

    def get_engine() -> CrackInference:
        global _engine, _load_error
        if _engine is None and _load_error is None:
            try:
                _engine = CrackInference(Path(os.getenv("CHECKPOINT_PATH", str(DEFAULT_CHECKPOINT))))
            except (FileNotFoundError, RuntimeError, ValueError) as error:
                _load_error = str(error)
        if _engine is None:
            raise HTTPException(status_code=503, detail={"message": "Model unavailable", "reason": _load_error})
        return _engine

    async def read_image(file: UploadFile) -> Image.Image:
        if file.content_type and not file.content_type.startswith("image/"):
            raise HTTPException(status_code=415, detail="Upload an image file.")
        raw = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"Image exceeds {MAX_UPLOAD_BYTES} bytes.")
        try:
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise HTTPException(status_code=422, detail="The upload is not a readable image.") from error

    @app.get("/health")
    def health() -> dict[str, Any]:
        checkpoint = Path(os.getenv("CHECKPOINT_PATH", str(DEFAULT_CHECKPOINT)))
        return {
            "status": "checkpoint_present" if checkpoint.is_file() else "model_not_loaded",
            "checkpoint": str(checkpoint),
            "prediction_endpoint": "/predict",
        }

    @app.post("/predict", response_class=StreamingResponse)
    async def predict(file: UploadFile = File(...)) -> StreamingResponse:
        mask = get_engine().predict(await read_image(file))
        buffer = io.BytesIO()
        Image.fromarray(mask).save(buffer, format="PNG")
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="image/png")

    @app.post("/predict-overlay", response_class=StreamingResponse)
    async def predict_overlay(file: UploadFile = File(...)) -> StreamingResponse:
        overlay = get_engine().predict_overlay(await read_image(file))
        buffer = io.BytesIO()
        Image.fromarray(overlay).save(buffer, format="PNG")
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="image/png")
except ImportError:  # pragma: no cover
    app = None
