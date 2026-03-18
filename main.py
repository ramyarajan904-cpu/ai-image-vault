import os, io, hashlib, zipfile, base64, asyncio, gc
from PIL import Image, ImageOps, ImageFilter
import imagehash
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def home():
    return {"message": "Server is running"}

def get_image_fingerprint(img_bytes):
    with Image.open(io.BytesIO(img_bytes)) as raw_img:
        raw_img = raw_img.convert('RGB')
        raw_img = ImageOps.exif_transpose(raw_img)
        # Blur panni noise koraikirom
        processed = raw_img.filter(ImageFilter.GaussianBlur(1))
        dh = imagehash.dhash(processed)

        # Preview size-ah 150-ku korainga (Render memory-kaaga)
        buf = io.BytesIO()
        preview = raw_img.copy()
        preview.thumbnail((150, 150))
        preview.save(buf, format="JPEG", quality=70)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    return dh, img_str

@app.websocket("/ws/check-duplicate")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    originals_bank = [] 

    try:
        data = await websocket.receive_bytes()
        
        if zipfile.is_zipfile(io.BytesIO(data)):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                # ❗ FIX 1: SORTING IS MANDATORY
                files = [f for f in z.namelist() if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                files.sort(key=lambda x: (len(x), x))
                
                total = len(files)

                for i, file in enumerate(files):
                    img_bytes = z.read(file)
                    fname = os.path.basename(file)
                    curr_hash, preview = get_image_fingerprint(img_bytes)

                    status = "Unique"
                    similarity = 100
                    found = False

                    for orig_hash, _ in originals_bank:
                        diff = curr_hash - orig_hash
                        if diff < 12: # Threshold kuncham increase panni irukkaen
                            status = "Near-Duplicate"
                            similarity = round((1 - diff / 64) * 100, 2)
                            found = True
                            break

                    if not found:
                        originals_bank.append((curr_hash, fname))

                    # ❗ FIX 2: SEND EACH RESULT LIVE (To avoid Connection Error)
                    await websocket.send_json({
                        "type": "progress",
                        "current": i + 1,
                        "total": total,
                        "message": f"Processing {fname}",
                        "single_result": { # Flutter-ku live update anuppuraen
                            "uploaded_file": fname,
                            "status": status,
                            "similarity_percentage": similarity,
                            "image_data": preview
                        }
                    })

                    # RAM clear panna
                    if i % 20 == 0:
                        gc.collect()
                    
                    await asyncio.sleep(0.01)

        await websocket.send_json({"type": "complete"})

    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
