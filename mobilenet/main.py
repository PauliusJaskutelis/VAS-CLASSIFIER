from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from PIL import Image
import io
import tensorflow as tf
import numpy as np

app = FastAPI()

model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Image preprocessing function
def prepare_image(image: Image.Image):
    image = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

@app.post("/classify")
async def classify(
    file: UploadFile = File(...),
    prediction_count: int = Query(5, ge=1, le=100),
    confidence_threshold: float = Query(0.1, ge=0.0, le=1.0),    
):
    if prediction_count > 100:
        raise HTTPException(status_code=400, detail="prediction count must be <= 100")
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    prepped = prepare_image(image)
    predictions = model.predict(prepped)

    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=prediction_count)[0]
    
    results = [ 
    {
        "label": label,
        "confidence": float(confidence),
    }
    for(_, label, confidence) in decoded
    if confidence > confidence_threshold
    ]

    return {"results": results}