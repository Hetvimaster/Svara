import sys
import os
import json
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "aasist_repo"))

from models.AASIST import Model as AASISTModel

NB_SAMP = 64600  # AASIST-L expects exactly this many samples (~4s at 16kHz)

def pad_or_trim(audio_tensor, target_len=NB_SAMP):
    """AASIST requires a fixed-length input. Pad short clips by looping them,
    trim long clips to the target length."""
    length = audio_tensor.shape[0]
    if length >= target_len:
        return audio_tensor[:target_len]
    num_repeats = (target_len // length) + 1
    repeated = audio_tensor.repeat(num_repeats)
    return repeated[:target_len]

class SpoofDetector:
    def __init__(self, config_path, checkpoint_path, device="cpu"):
        self.device = device
        with open(config_path, "r") as f:
            config = json.load(f)

        self.model = AASISTModel(config["model_config"]).to(device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.model.eval()

    def predict(self, audio_tensor):
        """audio_tensor: 1D torch tensor of raw waveform samples at 16kHz."""
        audio_tensor = pad_or_trim(audio_tensor)
        with torch.no_grad():
            x = audio_tensor.unsqueeze(0).to(self.device)  # add batch dim
            _, output = self.model(x)
            score = torch.softmax(output, dim=1)[0, 1].item()  # prob of "spoof" class
        return score

    def predict_windowed(self, audio_tensor, hop_ratio=0.5, aggregate="mean"):
        """Score a long clip across multiple overlapping 4s windows instead
        of just the first NB_SAMP samples. Returns (aggregated_score, all_scores).

        aggregate: "mean" (default, smooths out one noisy window) or
                   "max" (flags the clip if ANY window looks spoofed —
                   more sensitive, useful if you'd rather over-flag than miss).
        """
        length = audio_tensor.shape[0]
        hop = int(NB_SAMP * hop_ratio)

        if length < NB_SAMP:
            # too short to window — fall back to single pad_or_trim score
            score = self.predict(audio_tensor)
            return score, [score]

        scores = []
        with torch.no_grad():
            for start in range(0, length - NB_SAMP + 1, hop):
                chunk = audio_tensor[start:start + NB_SAMP]
                x = chunk.unsqueeze(0).to(self.device)
                _, output = self.model(x)
                s = torch.softmax(output, dim=1)[0, 1].item()
                scores.append(s)

        if aggregate == "max":
            agg_score = max(scores)
        else:
            agg_score = sum(scores) / len(scores)

        return agg_score, scores
if __name__ == "__main__":
    detector = SpoofDetector(
        config_path="aasist_repo/config/AASIST-L.conf",
        checkpoint_path="aasist_repo/models/weights/AASIST-L_finetuned_v2.pth",
    )
    import soundfile as sf
    import librosa

    TARGET_SR = 16000
    test_files = ["hetvi.wav", "binita.wav", "hetviSpoof2.wav", "binitaSpoof.wav","spoof1.wav","spoof2.wav"]
    for fname in test_files:
        audio, sr = sf.read(f"../test_audio/{fname}")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != TARGET_SR:
            audio = librosa.resample(audio.astype("float32"), orig_sr=sr, target_sr=TARGET_SR)
        waveform = torch.tensor(audio, dtype=torch.float32)
        score, window_scores = detector.predict_windowed(waveform, aggregate="mean")
        # score = detector.predict(waveform)
        label = "SPOOF" if score > 0.5 else "bonafide"
        print(f"{fname:20s} spoof_score={score:.4f}  -> predicted: {label}")