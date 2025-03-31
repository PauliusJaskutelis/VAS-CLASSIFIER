from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

app = FastAPI()

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Placeholder for model logic
    # For now, return dummy classification
    return {
        "label": "t-shirt",
        "confidence": 0.95
    }