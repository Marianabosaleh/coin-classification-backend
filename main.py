from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import io
import os
from fastapi.middleware.cors import CORSMiddleware

# **Check if CUDA (GPU) is available**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# **Class Labels (Must match training)**
class_names = [
    "Vespasian", "Hadrian", "Trajan", "Antoninus", "Nerva", "Pertinax",
    "Alexander", "Vitellius", "Augustus", "Caligula", "Caracalla", "Claudius",
    "Commodus", "Didius", "Domitian", "Elagabalus", "Galba", "Geta",
    "Lucius", "Macrinus", "Marcus", "Nero", "Otho", "Septimius",
    "Tiberius", "Titus", "Maximinus Thrax", "Pupienus", "Balbinus",
    "Gordian III", "Philip the Arab", "Decius", "Trebonianus Gallus",
    "Aemilian", "Valerian", "Gallienus", "Claudius Gothicus", "Quintillus",
    "Aurelian", "Tacitus", "Florian", "Probus", "Carus", "Numerian",
    "Carinus", "Diocletian", "Maximian", "Constantius I", "Galerius",
    "Severus II", "Gordian II AND I"
]

# **Load trained model**
def load_model():
    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(model.classifier[1].in_features, len(class_names))
    )
    try:
        model.load_state_dict(torch.load("roman_emperors_FINAL.pth", map_location=device))
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        raise RuntimeError("Failed to load model")
    return model

model = load_model()

# **Image Preprocessing**
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# **Create FastAPI app**
app = FastAPI()

# **Enable CORS for React Frontend**
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# **Root Endpoint to Fix 404 Errors**
@app.get("/")
async def root():
    return {"message": "Welcome to the Ancient Coin Classifier API!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        print(f"📂 Received file: {file.filename}")

        # **Read and preprocess image**
        image_data = await file.read()
        try:
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Invalid image format")

        image = transform(image).unsqueeze(0).to(device)

        # **Run model inference**
        with torch.no_grad():
            output = model(image)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # **Get top 3 predictions**
        top3_prob, top3_indices = torch.topk(probabilities, 3)
        top3_results = [
            {"emperor": class_names[idx.item()], "confidence": round(prob.item() * 100, 2)}
            for prob, idx in zip(top3_prob, top3_indices)
        ]

        # **Check if Augustus (Caesar) is identified with high confidence**
        for result in top3_results:
            if result["emperor"] == "Augustus" and result["confidence"] > 90:
                top3_results = [result]  # If Augustus (Caesar) is above 90%, return only this result
                break

        # **Assign colors based on confidence levels**
        for result in top3_results:
            if result["confidence"] >= 90:
                result["color"] = "green"  # High confidence
            elif result["confidence"] >= 60:
                result["color"] = "orange"  # Medium confidence
            else:
                result["color"] = "red"  # Low confidence

        print(f"✅ Prediction: {top3_results}")
        return {"predictions": top3_results}

    except HTTPException as e:
        print(f"⚠️ Error: {e.detail}")
        raise e  # Re-raise FastAPI exceptions

    except Exception as e:
        print(f"❌ Unexpected Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

import os

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))  # Ensure it uses 8080 for Google Cloud Run
    print(f"Cloud Run assigned port: {port}")

    uvicorn.run(app, host="0.0.0.0", port=port)

