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
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications import (
    mobilenet_v2, efficientnet, xception, inception_v3, densenet
)

app = FastAPI()
model = None
last_used = time.time()
in_use = False
current_preprocessor = None
resize_to = (224, 224)  # fallback

MODEL_DIR = Path("uploaded_models")
MODEL_DIR.mkdir(exist_ok=True)
SUPPORTED_EXTENSIONS = [".h5", ".keras", ".zip"]


@app.on_event("startup")
async def startup_event():
    print("🟢 Container started. Awaiting model upload...")
    asyncio.create_task(shutdown_monitor())


@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    global model, last_used, in_use, current_preprocessor, resize_to
    in_use = True

    ext = Path(file.filename).suffix.lower()
    SUPPORTED_EXTENSIONS = [".h5", ".keras", ".zip"]
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Only {', '.join(SUPPORTED_EXTENSIONS)} files are supported.")

    model_id = str(uuid.uuid4())
    model_path = MODEL_DIR / f"{model_id}{ext}"

    # Save the uploaded file
    with open(model_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Case 1: HDF5 or Keras
        if ext in [".h5", ".keras"]:
            model = tf.keras.models.load_model(model_path)
        elif ext == ".zip":
            extracted_dir = MODEL_DIR / model_id  # unzip folder
            shutil.unpack_archive(model_path, extracted_dir)
            model = tf.keras.models.load_model(extracted_dir)

        else:
            raise HTTPException(status_code=400, detail="Unsupported model format.")
        
        model_input_shape = model.input_shape
        print(f"📥 Loaded model input shape: {model_input_shape}")
        
        # Infer image size
        if len(model_input_shape) == 4:
            h, w = model_input_shape[1:3]
            if isinstance(h, int) and isinstance(w, int):
                resize_to = (w, h)
                print(f"📐 Using resize_to = {resize_to}")
        
        # Choose appropriate preprocessor
        name = model.name.lower()
        print(f"🧠 Model filename: {model_path.name}")
        print(f"🧠 Preprocessor assigned: {current_preprocessor}")
        if "mobilenet" in name:
            current_preprocessor = mobilenet_v2.preprocess_input
        elif "efficientnet" in name:
            current_preprocessor = efficientnet.preprocess_input
        elif "xception" in name:
            current_preprocessor = xception.preprocess_input
        elif "inception" in name:
            current_preprocessor = inception_v3.preprocess_input
            print(f"🧠 Model filename: {model_path.name}")
            print(f"🧠 Preprocessor assigned: {current_preprocessor}")
        elif "densenet" in name:
            current_preprocessor = densenet.preprocess_input
        else:
            current_preprocessor = None
        
        if current_preprocessor:
            print(f"🧠 Selected preprocessing function: {current_preprocessor.__name__}")
        else:
            print("⚠️ No recognized preprocessor – will use manual normalization.")
            

        model.summary()

    except Exception as e:
        last_used = time.time()
        in_use = False
        # Clean up on failure
        model_path.unlink(missing_ok=True)
        extracted_dir = MODEL_DIR / model_id
        if extracted_dir.exists() and extracted_dir.is_dir():
            shutil.rmtree(extracted_dir)
        raise HTTPException(status_code=422, detail=f"Model could not be loaded: {str(e)}")
    
    last_used = time.time()
    in_use = False
    return {
        "message": "Model uploaded and validated successfully!",
        "model_id": model_id,
        "path": str(model_path)
    }

@app.post("/classify")
async def classify(
    file: UploadFile = File(...),
    prediction_count: int = Query(5),
    confidence_threshold: float = Query(0.1),
    resize_height: int = Query(28),
    resize_width: int = Query(28),
    normalize: bool = Query(True),
    color_mode: str = Query("auto"),  # options: auto, RGB, L
    dry_run: bool = Query(False)
):
    global last_used, model, in_use, current_preprocessor, resize_to
    in_use = True
    adjustments = []

    print("🔍 /classify endpoint hit")
    print(f"📥 Received file: {file.filename}")
    print(f"📏 Parameters - resize: ({resize_width}, {resize_height}), normalize: {normalize}, color_mode: {color_mode}, dry_run: {dry_run}")

    if model is None:
        print("❌ No model loaded!")
        raise HTTPException(status_code=503, detail="No model loaded. Please upload a model first.")

    try:
        contents = await file.read()
        print(f"📦 File read into memory, size: {len(contents)} bytes")
        image = Image.open(io.BytesIO(contents))
        print(f"🖼️ Image opened: size={image.size}, mode={image.mode}, format={image.format}")

        original_mode = image.mode
        expected_channels = model.input_shape[-1] if len(model.input_shape) == 4 else 1
        target_mode = "L" if expected_channels == 1 else "RGB"

        if color_mode == "auto" and image.mode != target_mode:
            adjustments.append(f"converted from {original_mode} to {target_mode}")
            image = image.convert(target_mode)
        elif color_mode != "auto" and image.mode != color_mode:
            adjustments.append(f"converted from {original_mode} to {color_mode}")
            image = image.convert(color_mode)

        target_size = model.input_shape[1:3]
        
        if image.size != target_size:
            adjustments.append(f"resized from {image.size} to {target_size}")
            image = image.resize(target_size)

        img_array = tf.keras.preprocessing.image.img_to_array(image)

        if expected_channels == 1 and img_array.ndim == 2:
            img_array = np.expand_dims(img_array, axis=-1)

        if current_preprocessor is not None:
            print(f"🔍 Preprocessor: {current_preprocessor}")
            print(f"📊 Before preprocess: min={np.min(img_array)}, max={np.max(img_array)}")
            img_array = current_preprocessor(img_array)
            adjustments.append(f"preprocessed with {current_preprocessor.__name__}")
            print(f"✅ After preprocess: min={np.min(img_array)}, max={np.max(img_array)}")
        # else:
        #     if normalize:
        #         img_array = img_array.astype("float32") / 255.0
        #         adjustments.append("normalized manually to [0, 1]")

        input_data = np.expand_dims(img_array, axis=0)

        if dry_run:
            in_use = False
            return {
                "adjustments": adjustments,
                "dry_run": True,
                "message": "This is a dry run — no classification performed."
            }
        print("🖼️ Pixel range after preprocessing:", np.min(img_array), "-", np.max(img_array))
        print("📐 Input shape before model.predict:", input_data.shape)
        predictions = model.predict(input_data)

        try:
            if predictions.shape[1] == 1000:
                print("🧠 Model output shape:", predictions.shape)
                print("📈 Top 5 raw probs:", predictions[0][:5])
                print("📦 Sum of all probs:", np.sum(predictions[0]))
                print("🧪 Max class index:", np.argmax(predictions[0]))
                decoded = decode_predictions(predictions, top=prediction_count)[0]
                results = [{"label": label, "confidence": float(conf)} for (_, label, conf) in decoded if conf > confidence_threshold]
            else:
                raise ValueError("Not ImageNet-compatible")
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
    return {
        "adjustments": adjustments,
        "results": results
    }


@app.get("/has-model")
async def has_model():
    global model
    return {"loaded": model is not None}

async def shutdown_monitor(timeout: int = 120):
    """Shuts down the container after inactivity timeout (default: 120 seconds)."""
    global last_used
    global in_use
    while True:
        await asyncio.sleep(10)
        if not in_use and (time.time() - last_used > timeout):
            print("💤 No activity detected. Shutting down container...")
            os._exit(0)
