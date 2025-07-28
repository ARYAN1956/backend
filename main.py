from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io
import os
import requests

app = FastAPI()

# Allow frontend access
origins = [
    "http://localhost:3000",
    "http://localhost:3002",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3002",
    "https://ai-brain-tumor-classification-apps.onrender.com",  # Replace with actual Render frontend
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Google Drive File ID from your shared link
GOOGLE_DRIVE_FILE_ID = "1NWARf_1m_ouX0O98O1qmGA2oYs9yUEn-"
MODEL_URL = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}"
MODEL_PATH = os.path.join("model", "brain_tumor.h5")


def download_model():
    """Download model from Google Drive if not present"""
    if not os.path.exists(MODEL_PATH):
        print("🔄 Downloading model from Google Drive...")
        os.makedirs("model", exist_ok=True)
        try:
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            print("✅ Model downloaded successfully.")
        except Exception as e:
            print(f"❌ Failed to download model: {e}")


# ⬇️ Download and load model
download_model()
model = None
try:
    model = load_model(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")

# Class labels
class_names = ["Glioma", "Meningioma", "notumor", "Pituitary"]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((200, 200))  # Resize to model input shape
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        class_index = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))

        return {
            "prediction": class_names[class_index],
            "confidence": f"{confidence * 100:.2f}%",
        }

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
