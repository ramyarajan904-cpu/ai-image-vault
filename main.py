import hashlib
from fastapi import FastAPI, UploadFile, File
from typing import List
import torch
import torch.nn.functional as F
from torchvision import transforms
import timm
from PIL import Image
import io
import imagehash
import uvicorn
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 1. CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Session Memory
processed_images = []

# 2. AI Model (ViT) Setup - Using a very small model for Render Free Tier
model_name = 'vit_tiny_patch16_224'
# Pretrained model-ah load panni RAM-ah optimize panrom
model = timm.create_model(model_name, pretrained=True, num_classes=0)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- HELPER FUNCTIONS ---
def get_md5(content):
    return hashlib.md5(content).hexdigest()

def get_vit_embedding(image_pil):
    img_tensor = preprocess(image_pil).unsqueeze(0)
    with torch.no_grad():
        features = model(img_tensor)
        features = F.normalize(features, p=2, dim=1)
    return features

@app.get("/")
def health_check():
    return {"status": "online", "model": "ViT-Tiny Live"}

@app.get("/reset_session")
def reset_session():
    global processed_images
    processed_images = []
    return {"status": "success", "message": "Memory Reset"}

# --- MAIN API ---
@app.post("/compare")
async def compare_batch(files: List[UploadFile] = File(...)):
    global processed_images
    batch_results = []

    for file in files:
        try:
            content = await file.read()
            
            # STAGE 1: MD5 (Byte-level)
            current_md5 = get_md5(content)
            
            # Convert to PIL
            img_pil = Image.open(io.BytesIO(content)).convert('RGB')
            
            # STAGE 2: dHash (Structural)
            current_dhash = imagehash.dhash(img_pil)
            
            # STAGE 3: ViT (Semantic)
            feat = get_vit_embedding(img_pil)
            
            is_match = False
            match_data = None

            for old_img in processed_images:
                # 1. Check MD5 (Bit-by-bit same)
                if current_md5 == old_img["md5"]:
                    is_match = True
                    match_data = {"pair": [file.filename, old_img["filename"]], "similarity": 100, "status": "Exact Duplicate"}
                    break
                
                # 2. Check dHash (Structure same)
                hash_diff = current_dhash - old_img["dhash"]
                if hash_diff == 0:
                    is_match = True
                    match_data = {"pair": [file.filename, old_img["filename"]], "similarity": 100, "status": "Exact Duplicate"}
                    break
                
                # 3. Check AI Similarity (Content similarity)
                cos_sim = F.cosine_similarity(feat, old_img["features"]).item()
                sim_percent = round(cos_sim * 100, 2)

                # Similarity threshold: 88% and above is considered Near-Duplicate
                if hash_diff <= 2 or sim_percent > 88.0:
                    is_match = True
                    match_data = {"pair": [file.filename, old_img["filename"]], "similarity": sim_percent, "status": "Near-Duplicate"}
                    break

            # Save for next comparison
            processed_images.append({
                "filename": file.filename,
                "md5": current_md5,
                "dhash": current_dhash,
                "features": feat
            })

            if is_match:
                batch_results.append(match_data)

        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            continue

    return {"duplicates": batch_results}

if __name__ == "__main__":
    # Render logic dynamic port
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
