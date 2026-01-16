from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import whisper

app = FastAPI()

# CORS (React Native / Web dono ke liye)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once (startup pe)
# tiny = fastest, small = better accuracy
MODEL_NAME = os.getenv("WHISPER_MODEL", "small")
model = whisper.load_model(MODEL_NAME)

@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_NAME}

@app.post("/api/stt")
async def stt(
    audio: UploadFile = File(...),
    language: str = Form("hi")
):
    # Save temp file
    suffix = os.path.splitext(audio.filename)[-1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        result = model.transcribe(tmp_path, language=language)
        text = (result.get("text") or "").strip()
        return {"text": text}
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass
