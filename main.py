import os, io, zipfile, base64, asyncio, gc
from PIL import Image, ImageOps, ImageFilter
import imagehash
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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
        processed = raw_img.filter(ImageFilter.GaussianBlur(1))
        dh = imagehash.dhash(processed)
        
        buf = io.BytesIO()
        preview = raw_img.copy()
        preview.thumbnail((120, 120)) # Reduced size for faster transfer
        preview.save(buf, format="JPEG", quality=60)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    return dh, img_str

@app.websocket("/ws/check-duplicate")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    originals_bank = [] 

    try:
        # Wait for ZIP bytes
        data = await websocket.receive_bytes()
        
        if zipfile.is_zipfile(io.BytesIO(data)):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                files = [f for f in z.namelist() if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                files.sort(key=lambda x: (len(x), x))
                
                total = len(files)
                for i, file in enumerate(files):
                    try:
                        img_bytes = z.read(file)
                        fname = os.path.basename(file)
                        curr_hash, preview = get_image_fingerprint(img_bytes)

                        status, similarity, found = "Unique", 100, False
                        for orig_hash, _ in originals_bank:
                            diff = curr_hash - orig_hash
                            if diff < 12: 
                                status = "Near-Duplicate"
                                similarity = round((1 - diff / 64) * 100, 2)
                                found = True
                                break

                        if not found:
                            originals_bank.append((curr_hash, fname))

                        # LIVE UPDATE
                        await websocket.send_json({
                            "type": "progress",
                            "current": i + 1,
                            "total": total,
                            "message": f"Processing {fname}",
                            "single_result": {
                                "uploaded_file": fname,
                                "status": status,
                                "similarity_percentage": similarity,
                                "image_data": preview
                            }
                        })
                        
                        # Prevent Timeout: Small sleep to let event loop breathe
                        if i % 5 == 0:
                            await asyncio.sleep(0.01)
                        if i % 20 == 0:
                            gc.collect()

                    except Exception as img_err:
                        print(f"Error processing {file}: {img_err}")
                        continue

        await websocket.send_json({"type": "complete"})
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        try:
            await websocket.close()
        except:
            pass
