import numpy as np
import librosa

def compute_spectral_flatness(audio, sr=16000):
    """Replayed audio (played through a speaker, re-recorded) tends to have
    different spectral flatness than a live/direct recording, due to the
    extra speaker+mic transfer function in the signal path."""
    flatness = librosa.feature.spectral_flatness(y=audio)
    return float(np.mean(flatness))

def compute_high_freq_energy_ratio(audio, sr=16000):
    """Replay/re-recording through consumer speakers often attenuates high
    frequencies more than direct recording does — a coarse but real signal."""
    stft = np.abs(librosa.stft(audio))
    freqs = librosa.fft_frequencies(sr=sr)
    high_freq_mask = freqs > 4000
    high_energy = np.sum(stft[high_freq_mask, :])
    total_energy = np.sum(stft)
    return high_energy / total_energy if total_energy > 0 else 0

def classify_replay(audio, sr=16000):
    """
    Coarse 3-way heuristic classifier: live / replayed / converted.
    NOT a trained model — this is a scoped-down placeholder using known
    replay-attack signal characteristics, per the feasibility plan.
    Thresholds are rough starting points, not tuned on real data yet.
    """
    flatness = compute_spectral_flatness(audio, sr)
    hf_ratio = compute_high_freq_energy_ratio(audio, sr)

    if hf_ratio < 0.05 and flatness > 0.3:
        classification = "replayed"
        confidence = 0.6  # low confidence — heuristic only
    elif flatness > 0.4:
        classification = "converted"
        confidence = 0.5
    else:
        classification = "live"
        confidence = 0.6

    return {"classification": classification, "confidence": confidence,
            "flatness": flatness, "hf_ratio": hf_ratio}

if __name__ == "__main__":
    import soundfile as sf
    audio, sr = sf.read("../test_audio/sample.wav")
    result = classify_replay(audio.astype(np.float32), sr)
    print(result)