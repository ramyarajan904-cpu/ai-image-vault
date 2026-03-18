import os, io, hashlib, zipfile, base64, asyncio
from PIL import Image, ImageOps, ImageFilter
import imagehash
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ ROOT ROUTE (IMPORTANT)
@app.get("/")
def home():
    return {"message": "Server is running"}

# 🔹 Image Fingerprint Function
def get_image_fingerprint(img_bytes):
    raw_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    raw_img = ImageOps.exif_transpose(raw_img)

    # Blur (remove brightness noise)
    processed = raw_img.filter(ImageFilter.GaussianBlur(1))

    dh = imagehash.dhash(processed)

    # Preview for UI
    buf = io.BytesIO()
    preview = raw_img.copy()
    preview.thumbnail((300, 300))
    preview.save(buf, format="JPEG")
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')

    return dh, img_str

# 🔥 WebSocket
@app.websocket("/ws/check-duplicate")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    originals_bank = []  # [(hash, filename)]

    try:
        data = await websocket.receive_bytes()
        results_list = []

        if zipfile.is_zipfile(io.BytesIO(data)):
            with zipfile.ZipFile(io.BytesIO(data)) as z:

                files = [
                    f for f in z.namelist()
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]

                # ❗ IMPORTANT: preserve original order (NO SORT)
                total = len(files)

                for i, file in enumerate(files):
                    img_bytes = z.read(file)
                    fname = os.path.basename(file)

                    curr_hash, preview = get_image_fingerprint(img_bytes)

                    status = "Unique"
                    similarity = 100
                    found = False

                    # 🔍 Compare with originals
                    for orig_hash, _ in originals_bank:
                        diff = curr_hash - orig_hash

                        if diff < 10:  # threshold
                            status = "Duplicate"
                            similarity = round((1 - diff / 64) * 100, 2)
                            found = True
                            break

                    # ✅ Store ONLY if original
                    if not found:
                        originals_bank.append((curr_hash, fname))

                    results_list.append({
                        "uploaded_file": fname,
                        "status": status,
                        "similarity_percentage": similarity,
                        "image_data": preview
                    })

                    # 🔥 PROGRESS FIX
                    await websocket.send_json({
                        "type": "progress",
                        "current": i + 1,
                        "total": total,
                        "message": f"Processing {fname}"
                    })

                    await asyncio.sleep(0.1)

        await websocket.send_json({
            "type": "complete",
            "results": results_list
        })

    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })

    finally:
        await websocket.close()
