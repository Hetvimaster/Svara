import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import soundfile as sf
import numpy as np

# from aasist_wrapper import SpoofDetector
# from speaker_verification import SpeakerVerifier
# from replay_detector import classify_replay
# from risk_engine.fusion import compute_risk_score, get_tier
from aasist_wrapper import SpoofDetector
from speaker_verification import SpeakerVerifier
from replay_detector import classify_replay
from risk_engine.fusion import compute_risk_score, get_tier
from dual_window_ewma import DualWindowEWMA
TARGET_SR = 16000

def load_audio(path):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != TARGET_SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    return audio, sr


def run(enrolled_paths, test_files):
    # detector = SpoofDetector(
    #     config_path="aasist_repo/config/AASIST-L.conf",
    #     # checkpoint_path="aasist_repo/models/weights/AASIST-L_finetuned_v2.pth",
    #     checkpoint_path="aasist_repo/models/weights/AASIST-L_finetuned_indic_merged.pth",
    # )
    # verifier = SpeakerVerifier()
    detector = SpoofDetector(
        config_path="aasist_repo/config/AASIST-L.conf",
        # checkpoint_path="aasist_repo/models/weights/AASIST-L_finetuned_v2.pth",
        checkpoint_path="aasist_repo/models/weights/AASIST-L_finetuned_indic_merged.pth",
    )
    verifier = SpeakerVerifier()

    # # detector.predict expects a torch tensor, but DualWindowEWMA feeds it
    # # numpy chunks — this lambda bridges that.
    # ewma_scorer = DualWindowEWMA(
    #     score_fn=lambda chunk: detector.predict(torch.tensor(chunk, dtype=torch.float32)),
    #     sample_rate=TARGET_SR,
    # )
    ewma_scorer = DualWindowEWMA(alpha_fast=0.6, alpha_slow=0.2, combine="max")
    enrolled_tensors = []
    for path in enrolled_paths:
        audio, _ = load_audio(path)
        enrolled_tensors.append(torch.tensor(audio))
    verifier.enroll_multi(enrolled_tensors)
    print(f"Enrolled identity from: {enrolled_paths}\n")

    header = f"{'file':20s} {'spoof':>8s} {'speaker_sim':>12s} {'replay':>10s} {'risk':>8s} {'tier':>8s}"
    print(header)
    print("-" * len(header))
    for fname in test_files:
        audio, sr = load_audio(f"../test_audio/{fname}")
        waveform = torch.tensor(audio)

        # spoof_score = detector.predict_windowed(waveform, aggregate="max")[0]
        # ewma_result = ewma_scorer.score(audio, sr=TARGET_SR)
        # spoof_score = ewma_result["final_score"]
        _, window_scores = detector.predict_windowed(waveform, aggregate="max")
        ewma_result = ewma_scorer.score_from_windows(window_scores)
        spoof_score = ewma_result["final_score"]
        speaker_sim = verifier.verify(waveform)
        replay_result = classify_replay(audio, sr)

        risk = compute_risk_score(spoof_score, speaker_sim, replay_result)
        tier = get_tier(risk)

        print(f"{fname:20s} {spoof_score:8.3f} {speaker_sim:12.3f} "
              f"{replay_result['classification']:>10s} {risk:8.2f} {tier:>8s}")


if __name__ == "__main__":
    test_files = ["hetvi3.wav", "hetvi3Spoof.wav",
                  "hetviSpoof.wav", "spoof1.wav", "spoof2.wav"]
    run(enrolled_paths=["../test_audio/hetvi2.wav", "../test_audio/hetvi3.wav"],
        test_files=test_files)