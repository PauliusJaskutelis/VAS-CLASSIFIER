
from fastapi import FastAPI, UploadFile, File
from pathlib import Path
from tensorflow.keras.models import load_model  # or torch.load
import shutil
import uuid

app = FastAPI()

MODEL_DIR = Path("uploaded_models")
MODEL_DIR.mkdir(exist_ok=True)

@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    if not file.filename.endswith(".h5"):
        return {"error": "Only .h5 files are supported right now."}

    model_id = str(uuid.uuid4())
    model_path = MODEL_DIR / f"{model_id}.h5"

    with open(model_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        model = load_model(model_path)
        # Optionally: test dummy prediction here
        model.summary()
    except Exception as e:
        return {"error": f"Model could not be loaded: {str(e)}"}

    return {
        "message": "Model uploaded successfully!",
        "model_id": model_id,
        "path": str(model_path)
    }