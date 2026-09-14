import sys
import os
import json
import torch

# make the cloned repo importable
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

if __name__ == "__main__":
    detector = SpoofDetector(
        config_path="aasist_repo/config/AASIST-L.conf",
        checkpoint_path="aasist_repo/models/weights/AASIST-L.pth",
    )
    import soundfile as sf
    audio, sr = sf.read("../test_audio/sample.wav")
    waveform = torch.tensor(audio, dtype=torch.float32)
    score = detector.predict(waveform)
    print(f"Spoof score: {score:.4f}")