# Svara — AI Voice Impersonation Detection

Real-time risk scoring system for AI-cloned voice impersonation attacks, built for
Smart India Hackathon 2026 (Problem Statement SIH26104).

## What it does

Svara sits between an incoming call and any sensitive action, fusing three independent
signals into one continuous risk score:

1. **Spoof detection** — is this audio synthetic? (AASIST-L)
2. **Speaker verification** — does this voice match the claimed identity? (ECAPA-TDNN)
3. **Replay/conversion detection** — live, replayed, or converted audio? (heuristic classifier)

Scores are fused into a 0–100 risk score with a 3-tier response (low/medium/high).

## Setup

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Download model checkpoints (not included in this repo — too large for git)

**AASIST-L weights:**
Clone [AASIST](https://github.com/clovaai/aasist) separately and copy the checkpoint into:detection/aasist_repo/models/weights/AASIST-L.pth

**ECAPA-TDNN weights:**
Downloaded automatically on first run via SpeechBrain — no manual step needed.

### Run the API

```bash
uvicorn api.main:app --reload
```

Endpoints:
- `POST /enroll` — enroll a reference voiceprint
- `POST /verify_call` — score an incoming call against all three signals
- `GET /health` — health check

## Credits / Attribution

This project builds on the following open-source research and models — full credit to
their original authors:

- **AASIST** — Jung et al., "AASIST: Audio Anti-Spoofing using Integrated
  Spectro-Temporal Graph Attention Networks" (2021).
  [github.com/clovaai/aasist](https://github.com/clovaai/aasist). Source code
  vendored under `detection/aasist_repo/` per its original license (see LICENSE/NOTICE
  in that folder).
- **ECAPA-TDNN** — Desplanques et al. (2020), via
  [SpeechBrain](https://speechbrain.github.io/) pretrained checkpoint
  (`speechbrain/spkrec-ecapa-voxceleb`).
- Fine-tuning datasets referenced in the architecture (not yet integrated in this demo):
  IndicSynth (Sharma, Ekbote & Gupta, ACL 2025), SEA-Spoof (2025).

## Status

Working end-to-end pipeline: preprocessing → three-signal detection → risk fusion →
FastAPI endpoints. Replay/conversion detection is a heuristic placeholder (not a
trained model) — scoped this way for hackathon timeframe, documented as a roadmap item.