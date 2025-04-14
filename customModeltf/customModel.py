import asyncio
import os
import time
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from pathlib import Path
from PIL import Image
import shutil
import uuid
import io
import tensorflow as tf
import numpy as np

app = FastAPI()
model = None
last_used = time.time()
in_use = False

MODEL_DIR = Path("uploaded_models")
MODEL_DIR.mkdir(exist_ok=True)
SUPPORTED_EXTENSIONS = [".h5", ".keras"]


@app.on_event("startup")
async def startup_event():
    print("🟢 Container started. Awaiting model upload...")
    asyncio.create_task(shutdown_monitor())


@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    global model, last_used, in_use
    in_use = True

    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Only {', '.join(SUPPORTED_EXTENSIONS)} files are supported.")

    model_id = str(uuid.uuid4())
    model_path = MODEL_DIR / f"{model_id}{ext}"

    with open(model_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        model = tf.keras.models.load_model(model_path)
        model.summary()
        last_used = time.time()
    except Exception as e:
        in_use = False
        model_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Model could not be loaded: {str(e)}")
    
    last_used = time.time()
    in_use = False
    return {
        "message": "Model uploaded and validated successfully!",
        "model_id": model_id,
        "path": str(model_path)
    }


def prepare_image(image: Image.Image):
    image = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)


@app.post("/classify")
async def classify(
    file: UploadFile = File(...),
    prediction_count: int = Query(5),
    confidence_threshold: float = Query(0.1),
):
    global last_used, model, in_use
    in_use = True
    if model is None:
        raise HTTPException(status_code=503, detail="No model loaded. Please upload a model first.")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        prepped = prepare_image(image)

        predictions = model.predict(prepped)

        try:
            decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=prediction_count)[0]

            results = [
                {"label": label, "confidence": float(confidence)}
                for (_, label, confidence) in decoded
                if confidence > confidence_threshold
            ]
        except Exception:
            results = [
                {"label": str(i), "confidence": float(score)}
                for i, score in enumerate(predictions[0])
                if score > confidence_threshold
            ]
    except Exception as e:
        in_use = False
        raise HTTPException(status_code=500, detail=f"Failed to classify image: {str(e)}")
    
    last_used = time.time()
    in_use = False
    return {"results": results}


async def shutdown_monitor(timeout: int = 120):
    """Shuts down the container after inactivity timeout (default: 120 seconds)."""
    global last_used
    global in_use
    while True:
        await asyncio.sleep(10)
        if not in_use and (time.time() - last_used > timeout):
            print("💤 No activity detected. Shutting down container...")
            os._exit(0)
