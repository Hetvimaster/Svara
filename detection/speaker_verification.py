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

if __name__ == "__main__":
    import soundfile as sf

    verifier = SpeakerVerifier()

    audio, sr = sf.read("../test_audio/sample.wav")
    waveform = torch.tensor(audio, dtype=torch.float32)

    # enroll and verify against the SAME clip — sanity check, should be ~1.0
    verifier.enroll(waveform)
    similarity = verifier.verify(waveform)
    print(f"Self-similarity (sanity check): {similarity:.4f}")