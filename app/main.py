from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import base64
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = os.getenv("HF_MODEL")

# Initialize FastAPI app
app = FastAPI(
    title="Breast Cancer Classifier",
    description="Upload histopathology image for prediction"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to ["http://localhost:5173"] for frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Hugging Face client
client = InferenceClient(token=HF_API_TOKEN)

@app.get("/")
def home():
    return {
        "message": "Welcome to Breast Cancer Image Classifier API",
        "metrics": {
            "accuracy": 0.85,
            "benign": {
                "precision": 0.78,
                "recall": 0.74,
                "f1_score": 0.76,
                "support": 176
            },
            "malignant": {
                "precision": 0.88,
                "recall": 0.90,
                "f1_score": 0.89,
                "support": 369
            },
            "macro_avg": {
                "precision": 0.83,
                "recall": 0.82,
                "f1_score": 0.83,
                "support": 545
            },
            "weighted_avg": {
                "precision": 0.85,
                "recall": 0.85,
                "f1_score": 0.85,
                "support": 545
            }
        }
    }

@app.post("/predict")
async def classify_image(file: UploadFile = File(...)):
    try:
        # Read and convert image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Convert image to base64 string
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Send request to Hugging Face Hosted API
        result = client.post(
            HF_MODEL,
            {"image": img_str}
        )
        
        return JSONResponse({"prediction": result})
    
    except Exception as e:
        return JSONResponse({"error": str(e)})
