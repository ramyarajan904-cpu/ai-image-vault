import os, io, hashlib, zipfile, uvicorn, numpy as np, base64
import asyncio
from PIL import Image, ImageOps, ImageFilter
import imagehash
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def get_image_fingerprint(img_bytes):
    md5_hash = hashlib.md5(img_bytes).hexdigest()

    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = ImageOps.exif_transpose(img)

    # Slight blur → avoid brightness noise
    processed = img.filter(ImageFilter.GaussianBlur(radius=1.0))

    # dHash
    dhash = imagehash.dhash(processed)

    # Preview (for Flutter UI)
    buf = io.BytesIO()
    preview = img.copy()
    preview.thumbnail((300, 300))
    preview.save(buf, format="JPEG", quality=80)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')

    return md5_hash, dhash, img_str


@app.websocket("/ws/check-duplicate")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    known_md5s = set()
    originals_bank = []  # (dhash, filename)

    try:
        data = await websocket.receive_bytes()
        results_list = []

        if zipfile.is_zipfile(io.BytesIO(data)):
            with zipfile.ZipFile(io.BytesIO(data)) as z:

                all_files = [
                    f for f in z.infolist()
                    if not f.is_dir() and f.filename.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]

                # 🔥 IMPORTANT FIX → preserve original order (NO sorting)
                total_count = len(all_files)

                for i, f_info in enumerate(all_files):
                    img_bytes = z.read(f_info.filename)
                    fname = os.path.basename(f_info.filename)

                    curr_md5, curr_dh, preview_b64 = get_image_fingerprint(img_bytes)

                    status = "Unique"
                    similarity = 0.0
                    found_match = False

                    # ✅ 1. Exact duplicate
                    if curr_md5 in known_md5s:
                        status = "Exact Duplicate"
                        similarity = 100.0
                        found_match = True

                    else:
                        # ✅ 2. Near duplicate using dHash
                        for orig_dh, _ in originals_bank:
                            hash_diff = curr_dh - orig_dh

                            if hash_diff < 10:  # 🔥 tuned threshold
                                status = "Near-Duplicate"
                                similarity = float(round((1 - hash_diff / 64) * 100, 2))
                                found_match = True
                                break

                    # ✅ 3. Store only TRUE originals
                    if not found_match:
                        known_md5s.add(curr_md5)
                        originals_bank.append((curr_dh, fname))
                        similarity = 100.0

                    results_list.append({
                        "uploaded_file": fname,
                        "status": status,
                        "similarity_percentage": similarity,
                        "image_data": preview_b64
                    })

                    # ✅ FIXED progress (no stuck at 1)
                    await websocket.send_json({
                        "type": "progress",
                        "current": i + 1,
                        "total": total_count,
                        "message": f"Analyzing {fname}..."
                    })

                    await asyncio.sleep(0.05)

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


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
