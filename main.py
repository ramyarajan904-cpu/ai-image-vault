import hashlib
from fastapi import FastAPI, UploadFile, File
from typing import List
import torch
import torch.nn as File
from torchvision import transforms, models # Using standard torchvision models
from PIL import Image
import io
import imagehash
import uvicorn
import os
import gc # Garbage Collector to save RAM
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

processed_images = []

# --- ULTRA LIGHT MODEL: MobileNetV2 ---
# This uses much less RAM than ViT
model = models.mobilenet_v2(pretrained=True)
model.classifier = torch.nn.Identity() # Remove the last layer to get features
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_md5(content):
    return hashlib.md5(content).hexdigest()

def get_embedding(image_pil):
    img_tensor = preprocess(image_pil).unsqueeze(0)
    with torch.no_grad():
        features = model(img_tensor)
        # Flatten and Normalize
        features = torch.flatten(features, 1)
        features = torch.nn.functional.normalize(features, p=2, dim=1)
    return features

@app.get("/")
def health():
    return {"status": "online", "mode": "Ultra-Light-MobileNet"}

@app.get("/reset_session")
def reset_session():
    global processed_images
    processed_images = []
    gc.collect() # Force clear RAM
    return {"status": "success", "message": "Memory Cleared"}

@app.post("/compare")
async def compare_batch(files: List[UploadFile] = File(...)):
    global processed_images
    batch_results = []

    for file in files:
        try:
            content = await file.read()
            current_md5 = get_md5(content)
            img_pil = Image.open(io.BytesIO(content)).convert('RGB')
            current_dhash = imagehash.dhash(img_pil)
            feat = get_embedding(img_pil)
            
            is_match = False
            match_data = None

            for old_img in processed_images:
                # 1. MD5 Check
                if current_md5 == old_img["md5"]:
                    is_match = True
                    match_data = {"pair": [file.filename, old_img["filename"]], "similarity": 100, "status": "Exact Duplicate"}
                    break
                
                # 2. dHash Check
                hash_diff = current_dhash - old_img["dhash"]
                if hash_diff == 0:
                    is_match = True
                    match_data = {"pair": [file.filename, old_img["filename"]], "similarity": 100, "status": "Exact Duplicate"}
                    break
                
                # 3. MobileNet Similarity
                cos_sim = torch.nn.functional.cosine_similarity(feat, old_img["features"]).item()
                sim_percent = round(cos_sim * 100, 2)

                if hash_diff <= 2 or sim_percent > 92.0: # MobileNet needs higher threshold
                    is_match = True
                    match_data = {"pair": [file.filename, old_img["filename"]], "similarity": sim_percent, "status": "Near-Duplicate"}
                    break

            processed_images.append({
                "filename": file.filename,
                "md5": current_md5,
                "dhash": current_dhash,
                "features": feat
            })

            if is_match:
                batch_results.append(match_data)

        except Exception:
            continue
    
    # After batch is done, clear cache
    gc.collect() 
    return {"duplicates": batch_results}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
