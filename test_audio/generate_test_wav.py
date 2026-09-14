import numpy as np
import soundfile as sf

def generate_test_tone(filepath="sample.wav", duration=3, sample_rate=16000, freq=440):
    """Generate a simple sine wave WAV file for pipeline testing."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    sf.write(filepath, audio, sample_rate)
    print(f"Wrote {filepath} — {duration}s at {sample_rate}Hz")

if __name__ == "__main__":
    generate_test_tone()