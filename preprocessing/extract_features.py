import librosa
import numpy as np

def load_audio(filepath, sample_rate=16000):
    """Load a WAV file and resample to 16kHz (what AASIST expects)."""
    audio, sr = librosa.load(filepath, sr=sample_rate)
    return audio, sr

def extract_mfcc(audio, sample_rate=16000, n_mfcc=40):
    """Extract MFCC features from raw audio."""
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    return mfcc

if __name__ == "__main__":
    audio, sr = load_audio("test_audio/sample.wav")
    features = extract_mfcc(audio, sr)
    print(f"Audio shape: {audio.shape}, MFCC shape: {features.shape}")