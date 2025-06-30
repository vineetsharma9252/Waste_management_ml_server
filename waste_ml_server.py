from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os
import uuid
# predict.py

from tensorflow.keras.preprocessing import image
import numpy as np

# Load your model and categories only once at module level
from tensorflow.keras.models import load_model

model = load_model('waste_model_3.h5')
categories = ['cardboard', 'compost', 'glass', 'metal', 'paper', 'plastic', 'trash']# example categories

def predict_waste_type(img_path, model=model, categories=categories, target_size=(256, 256)):
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = categories[predicted_class]
        class_probabilities = {categories[i]: float(prob) for i, prob in enumerate(prediction[0])}

        return predicted_label, class_probabilities
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise

# main.py

 # <- this line connects it all

UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()

# Allow requests from Django (adjust for security)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with ["http://localhost:8000"] or your Django domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/classify-waste")
async def classify_waste(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid image format")

    try:
        unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
        image_path = os.path.join(UPLOAD_DIR, unique_filename)

        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        prediction, confidence = predict_waste_type(img_path=image_path)

        return JSONResponse(content={
            "prediction": prediction,
            "confidence": confidence,
            "top_confidence": max(confidence.values(), default=0)
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
    
    
import uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9000)
