import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "detection"))

import torch
import soundfile as sf
import numpy as np
from fastapi import FastAPI, UploadFile, File
import io

from detection.aasist_wrapper import SpoofDetector
from detection.speaker_verification import SpeakerVerifier
from detection.replay_detector import classify_replay
from risk_engine.fusion import compute_risk_score, get_tier

app = FastAPI(title="Svara Risk Scoring API")

# load models once at startup, not per-request
spoof_detector = SpoofDetector(
    config_path="detection/aasist_repo/config/AASIST-L.conf",
    checkpoint_path="detection/aasist_repo/models/weights/AASIST-L.pth",
)
speaker_verifier = SpeakerVerifier()

def load_audio_from_upload(file_bytes):
    audio, sr = sf.read(io.BytesIO(file_bytes))
    return torch.tensor(audio, dtype=torch.float32), sr

@app.post("/enroll")
async def enroll(file: UploadFile = File(...)):
    """Enroll a reference voiceprint for a claimed identity."""
    contents = await file.read()
    waveform, sr = load_audio_from_upload(contents)
    speaker_verifier.enroll(waveform)
    return {"status": "enrolled"}

@app.post("/verify_call")
async def verify_call(file: UploadFile = File(...)):
    """Score an incoming call audio clip against all three signals."""
    contents = await file.read()
    waveform, sr = load_audio_from_upload(contents)

    spoof_score = spoof_detector.predict(waveform)
    speaker_similarity = speaker_verifier.verify(waveform)
    replay_result = classify_replay(waveform.numpy(), sr)

    risk_score = compute_risk_score(spoof_score, speaker_similarity, replay_result)
    tier = get_tier(risk_score)

    return {
        "spoof_score": round(spoof_score, 4),
        "speaker_similarity": round(speaker_similarity, 4),
        "replay_classification": replay_result["classification"],
        "risk_score": round(risk_score, 2),
        "tier": tier,
    }

@app.get("/health")
async def health():
    return {"status": "ok"}