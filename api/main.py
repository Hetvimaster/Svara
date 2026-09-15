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
from risk_engine.audit_log import AuditLog

audit_log = AuditLog(log_path="data/audit_log.jsonl")

app = FastAPI(title="Svara Risk Scoring API")

# load models once at startup, not per-request
spoof_detector = SpoofDetector(
    config_path="detection/aasist_repo/config/AASIST-L.conf",
    checkpoint_path="detection/aasist_repo/models/weights/AASIST-L_finetuned_indic.pth",
)
speaker_verifier = SpeakerVerifier()

import librosa

TARGET_SR = 16000

def load_audio_from_upload(file_bytes):
    audio, sr = sf.read(io.BytesIO(file_bytes))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    return torch.tensor(audio, dtype=torch.float32), sr

@app.post("/enroll")
async def enroll(files: list[UploadFile] = File(...)):
    """Enroll a reference voiceprint, averaged across multiple clips for
    stability — a single clip's specific conditions (noise, mic distance,
    phrasing) don't dominate the whole reference embedding."""
    tensors = []
    for file in files:
        contents = await file.read()
        waveform, sr = load_audio_from_upload(contents)
        tensors.append(waveform)
    speaker_verifier.enroll_multi(tensors)
    return {"status": "enrolled", "num_clips": len(tensors)}

@app.post("/verify_call")
async def verify_call(file: UploadFile = File(...)):
    """Score an incoming call audio clip against all three signals."""
    contents = await file.read()
    waveform, sr = load_audio_from_upload(contents)

    spoof_score, _ = spoof_detector.predict_windowed(waveform, aggregate="max")
    speaker_similarity = speaker_verifier.verify(waveform)
    replay_result = classify_replay(waveform.numpy(), sr)

    risk_score = compute_risk_score(spoof_score, speaker_similarity, replay_result)
    tier = get_tier(risk_score)

    verdict = {
        "spoof_score": round(spoof_score, 4),
        "speaker_similarity": round(speaker_similarity, 4),
        "replay_classification": replay_result["classification"],
        "risk_score": round(risk_score, 2),
        "tier": tier,
    }
    verdict["entry_hash"] = audit_log.append(verdict)
    return verdict

@app.get("/health")
async def health():
    return {"status": "ok"}