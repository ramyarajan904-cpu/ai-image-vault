import os, io, hashlib, zipfile, uvicorn, numpy as np, base64
import asyncio
from PIL import Image, ImageOps, ImageFilter
import imagehash
import cloudinary, cloudinary.uploader
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

cloudinary.config(
    cloud_name="dniukejk0",
    api_key="485414975478379",
    api_secret="GiIGpX5Sa2R57sl7WvJ9mz9sAEw"
)

model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

# -------------------------
# 🔥 STRONG QUALITY FUNCTION
# -------------------------
def get_image_quality(img):
    gray = np.array(img.convert('L'))

    sharpness = np.var(gray)
    brightness = np.mean(gray)
    contrast = np.std(gray)

    penalty = 0

    if brightness > 170:
        penalty += 2000
    if brightness < 60:
        penalty += 2000
    if contrast < 30:
        penalty += 1000

    score = sharpness + contrast - penalty
    return score

# -------------------------
# PROCESS IMAGE
# -------------------------
def process_image(img_bytes):
    raw_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    raw_img = ImageOps.exif_transpose(raw_img)

    md5 = hashlib.md5(img_bytes).hexdigest()

    denoised = raw_img.filter(ImageFilter.GaussianBlur(radius=1.0))
    dh = imagehash.dhash(denoised)

    norm_img = ImageOps.autocontrast(raw_img).resize((224, 224))
    img_arr = keras_image.img_to_array(norm_img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = preprocess_input(img_arr)

    vec = model.predict(img_arr, verbose=0).flatten()
    vec = vec / (np.linalg.norm(vec) + 1e-7)

    quality = get_image_quality(raw_img)

    # preview
    buf = io.BytesIO()
    preview = raw_img.copy()
    preview.thumbnail((300, 300))
    preview.save(buf, format="JPEG", quality=80)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')

    return md5, dh, vec, img_str, quality, img_bytes

async def get_image_fingerprint(img_bytes):
    return await asyncio.to_thread(process_image, img_bytes)

# -------------------------
# WEBSOCKET
# -------------------------
@app.websocket("/ws/check-duplicate")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    clusters = []

    try:
        data = await websocket.receive_bytes()

        if zipfile.is_zipfile(io.BytesIO(data)):
            with zipfile.ZipFile(io.BytesIO(data)) as z:

                files = [
                    f for f in z.infolist()
                    if not f.is_dir() and f.filename.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]

                total = len(files)

                for i, f_info in enumerate(files):
                    fname = os.path.basename(f_info.filename)
                    img_bytes = z.read(f_info.filename)

                    md5, dh, vec, preview, quality, raw_bytes = await get_image_fingerprint(img_bytes)

                    found_cluster = None

                    # 🔥 DYNAMIC REFERENCE FIX
                    for cluster in clusters:
                        ref = max(cluster, key=lambda x: x["quality"])

                        hash_diff = dh - ref["dh"]
                        cos_sim = np.dot(vec, ref["vec"])

                        if hash_diff < 10 or cos_sim > 0.80:
                            found_cluster = cluster
                            break

                    img_data = {
                        "filename": fname,
                        "md5": md5,
                        "dh": dh,
                        "vec": vec,
                        "preview": preview,
                        "quality": quality,
                        "bytes": raw_bytes
                    }

                    if found_cluster:
                        found_cluster.append(img_data)
                    else:
                        clusters.append([img_data])

                    # progress
                    await websocket.send_json({
                        "type": "progress",
                        "current": i + 1,
                        "total": total,
                        "file": fname
                    })

                    await asyncio.sleep(0.05)

        # -------------------------
        # FINAL RESULT
        # -------------------------
        final_results = []

        for cluster in clusters:
            best = max(cluster, key=lambda x: x["quality"])

            for img in cluster:
                if img == best:
                    status = "Unique"
                    similarity = 100.0

                    # upload best only
                    try:
                        await asyncio.to_thread(
                            cloudinary.uploader.upload,
                            img["bytes"],
                            folder="ai_vault"
                        )
                    except:
                        pass

                else:
                    status = "Near-Duplicate"

                    hash_diff = img["dh"] - best["dh"]
                    cos_sim = np.dot(img["vec"], best["vec"])

                    similarity = float(round(max(
                        cos_sim * 100,
                        (1 - hash_diff / 64) * 100
                    ), 2))

                final_results.append({
                    "uploaded_file": img["filename"],
                    "status": status,
                    "similarity_percentage": similarity,
                    "image_data": img["preview"]
                })

        await websocket.send_json({
            "type": "complete",
            "results": final_results
        })

    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })

    finally:
        await websocket.close()

# -------------------------
# RUN SERVER
# -------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)