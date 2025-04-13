import asyncio
import os
import time
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from pydantic import BaseModel
from pathlib import Path
import shutil
import uuid
import tensorflow as tf

app = FastAPI()
model = None
last_used = time.time()
in_use = False

MODEL_DIR = Path("uploaded_models")
MODEL_DIR.mkdir(exist_ok=True)
SUPPORTED_EXTENSIONS = [".h5", ".keras"]

class ModelMetadata(BaseModel):
    model_id: str
    filename: str
    input_height: int
    input_width: int
    color_mode: str
    preprocessing: str = "normalize"
    storage_path: str


@app.on_event("startup")
async def startup_event():
    print("🟢 Container started. Awaiting model upload...")
    asyncio.create_task(shutdown_monitor())

@app.post("/extract-metadata", response_model=ModelMetadata)
async def extract_metadata(file: UploadFile = File(...)):
    global last_used, in_use
    in_use = True
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Only {', '.join(SUPPORTED_EXTENSIONS)} files are supported.")

    model_id = str(uuid.uuid4())
    model_path = MODEL_DIR / f"{model_id}{ext}"

    with open(model_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        temp_model = tf.keras.models.load_model(model_path)
        input_shape = temp_model.input_shape  # e.g. (None, 224, 224, 3) or (None, 28, 28)

        input_shape_safe = [d if d is not None else -1 for d in input_shape]

        # Defaults in case shape doesn't match expectations
        height = width = channels = None
        color_mode = "unknown"

        if len(input_shape) >= 3:
            height = input_shape[1]
            width = input_shape[2]
            channels = input_shape[3] if len(input_shape) == 4 else 1
            color_mode = "RGB" if channels == 3 else "L"

    except Exception as e:
        model_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Model could not be loaded: {str(e)}")
    in_use = False
    last_used = time.time()
    return ModelMetadata(
        model_id=model_id,
        filename=file.filename,
        input_height=height if height else -1,
        input_width=width if width else -1,
        color_mode=color_mode,
        preprocessing="normalize",
        input_shape=input_shape_safe,
        storage_path=str(model_path)
    )

@app.post("/shutdown")
def shutdown():
    print("🔻 Manual shutdown triggered")
    os._exit(0)

async def shutdown_monitor(timeout: int = 120):
    global last_used
    global in_use
    while True:
        print("⏳ Monitoring for inactivity...", in_use, time.time() - last_used)
        await asyncio.sleep(10)
        if not in_use and (time.time() - last_used > timeout):
            print("💤 No activity detected. Shutting down container...")
            os._exit(0)
