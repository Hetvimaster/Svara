import torch
from speechbrain.inference.speaker import EncoderClassifier

class SpeakerVerifier:
    def __init__(self, device="cpu"):
        self.device = device
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )
        self.enrolled_embedding = None  # stores the "claimed identity" reference

    def enroll(self, audio_tensor):
        """Store a reference voiceprint for the claimed identity.
        audio_tensor: 1D torch tensor, raw waveform at 16kHz."""
        embedding = self.model.encode_batch(audio_tensor.unsqueeze(0))
        self.enrolled_embedding = embedding.squeeze()
        return self.enrolled_embedding

    def enroll_multi(self, audio_tensors):
        """Average embeddings across several real clips of the same person —
        more stable than a single clip, since any one recording's specific
        conditions (noise, mic distance, phrasing) get averaged out rather
        than becoming the entire reference voiceprint."""
        embeddings = [self.model.encode_batch(a.unsqueeze(0)).squeeze() for a in audio_tensors]
        self.enrolled_embedding = torch.stack(embeddings).mean(dim=0)
        return self.enrolled_embedding

    def verify(self, audio_tensor):
        """Compare incoming audio against the enrolled voiceprint.
        Returns cosine similarity: closer to 1 = same speaker, closer to 0 = mismatch."""
        if self.enrolled_embedding is None:
            raise ValueError("No enrolled identity yet — call enroll() first")

        incoming_embedding = self.model.encode_batch(audio_tensor.unsqueeze(0)).squeeze()
        similarity = torch.nn.functional.cosine_similarity(
            self.enrolled_embedding.unsqueeze(0),
            incoming_embedding.unsqueeze(0),
        ).item()
        return similarity

def load(path):
    import soundfile as sf
    audio, sr = sf.read(path)
    return torch.tensor(audio, dtype=torch.float32)

if __name__ == "__main__":
    verifier = SpeakerVerifier()

    hetvi = load("../test_audio/hetvi.wav")
    binita = load("../test_audio/binita.wav")
    hetvi_spoof = load("../test_audio/hetviSpoof.wav")
    binita_spoof = load("../test_audio/binitaSpoof.wav")

    # enrolled identity: Hetvi's real voice
    verifier.enroll(hetvi)
    print(f"Hetvi real  vs Hetvi's clone    : {verifier.verify(hetvi_spoof):.4f}  (does the clone fool it?)")
    print(f"Hetvi real  vs Binita real      : {verifier.verify(binita):.4f}  (different speaker, expect low)")

    # switch enrolled identity: Binita's real voice
    verifier.enroll(binita)
    print(f"Binita real vs Binita's clone   : {verifier.verify(binita_spoof):.4f}  (does the clone fool it?)")
    print(f"Binita real vs Hetvi real       : {verifier.verify(hetvi):.4f}  (different speaker, expect low)")