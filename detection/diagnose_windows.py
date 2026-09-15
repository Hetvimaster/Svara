# save as detection/diagnose_windows.py
import torch, numpy as np, soundfile as sf
from aasist_wrapper import SpoofDetector, NB_SAMP

HOP = NB_SAMP // 2  # 50% overlap

def load(path):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), sr

def window_scores(detector, audio):
    """Score every 4s window across the file instead of only the first one."""
    if len(audio) < NB_SAMP:
        return None  # too short to window; handled separately
    scores = []
    for start in range(0, len(audio) - NB_SAMP + 1, HOP):
        chunk = torch.tensor(audio[start:start + NB_SAMP])
        with torch.no_grad():
            _, out = detector.model(chunk.unsqueeze(0))
            scores.append(torch.softmax(out, dim=1)[0, 0].item())
    return scores

def zero_pad(audio):
    out = np.zeros(NB_SAMP, dtype=np.float32)
    out[:len(audio)] = audio[:NB_SAMP]
    return out

if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "aasist_repo/models/weights/AASIST-L_finetuned_indic.pth"
    print(f"checkpoint: {ckpt}\n")
    det = SpoofDetector(config_path="aasist_repo/config/AASIST-L.conf", checkpoint_path=ckpt)

    files = ["hetvi.wav","hetvi2.wav","binita.wav","hetviSpoof2.wav",
             "binitaSpoof.wav","spoof1.wav","spoof2.wav"]
    for f in files:
        audio, sr = load(f"../test_audio/{f}")
        scores = window_scores(det, audio)
        if scores is None:
            # short file: compare loop-pad vs zero-pad to isolate splice artifact
            looped = det.predict(torch.tensor(audio))
            zeroed = det.predict(torch.tensor(zero_pad(audio)))
            print(f"{f:18s} SHORT  loop_pad={looped:.3f}  zero_pad={zeroed:.3f}")
        else:
            a = np.array(scores)
            print(f"{f:18s} n={len(a):3d}  min={a.min():.3f}  mean={a.mean():.3f}  "
                  f"max={a.max():.3f}  frac>0.5={np.mean(a>0.5):.2f}")