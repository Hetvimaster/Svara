# Svara
### AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks
**SIH 2026 — Problem Statement SIH26104** | Theme: Blockchain & Cybersecurity | Category: Software

---

## What it does, in one breath

A caller claims to be someone specific. Svara checks two independent things —
*is this voice synthetic at all*, and *does it actually sound like the person it
claims to be* — fuses both into one risk score, and escalates high-risk calls to
human review instead of silently blocking or silently allowing.

---

## Proof this actually works — not just architecture slides

We didn't stop at benchmark numbers. We fine-tuned the spoof detector ourselves
and stress-tested it against **three independent, real voice-synthesis tools it was
never shown during training**:

| Test | Result |
|---|---|
| Held-out validation (IndicSynth + Kathbath, 12 Indian languages, never seen in training) | Fine-tuned AASIST-L reached single-digit EER, from a near-chance starting point on the un-fine-tuned base checkpoint |
| Real ElevenLabs voice clone (language-matched to fine-tuning data) | Caught with ~99% confidence |
| Real ElevenLabs voice clone (English/mismatched-language output) | Missed by the spoof detector alone — a genuine, documented cross-domain limitation, not hidden |
| macOS-native TTS synthesis | Caught reliably, with measured sensitivity to which portion of a longer clip gets sampled — directly informed our windowed + EWMA scoring design below |

We're stating the miss as clearly as the catch. A system that only shows you its
wins isn't trustworthy — the cross-domain gap above is exactly why continuous
fine-tuning against new synthesis tools is part of the design, not an afterthought.

**A note on where this stands:** the fine-tuned checkpoint above is a working,
tested snapshot — not a finished, final-accuracy model. Fine-tuning is ongoing, and
the numbers above will keep improving as training continues. We're showing where
this is *right now*, including a real miss, rather than waiting for a polished
number before being honest about it.

---

## Architecture — what's actually built vs. what's designed

| Component | Status |
|---|---|
| Spoof detection (AASIST-L, fine-tuned on IndicSynth + Kathbath + MLAAD + SpoofCeleb) | **Built & tested** |
| Speaker verification (ECAPA-TDNN, multi-clip enrollment) | **Built & tested** |
| Dual-window EWMA score smoothing | **Built & tested** (wired into both the test pipeline and the live API) |
| Two-signal risk fusion (0–100 score, 3-tier response) | **Built & tested** |
| Hash-chained tamper-evident audit log | **Built & tested** (chain-integrity check verified) |
| REST API (`/enroll`, `/verify_call`) | **Built & tested** |
| Dashboard/UI, gRPC, webhook alerts | Designed, not built — reasonable extension path for production, not demoed here |

We're not claiming more than what's in this repo. Anything not marked "built &
tested" is architecture, not implementation.

---

## What it does

Svara sits between an incoming call and any sensitive action, fusing two
independent signals into one continuous risk score:

1. **Spoof detection** — is this audio synthetic? (AASIST-L, fine-tuned on
   Indian-language data, scored across overlapping windows and smoothed with dual
   fast/slow EWMA rather than a single fixed-window read)
2. **Speaker verification** — does this voice match the claimed identity?
   (ECAPA-TDNN, multi-clip enrollment averaged for a stable reference voiceprint)

Scores are fused into a 0–100 risk score with a 3-tier response (low/medium/high),
and every verdict is written to a tamper-evident, hash-chained local audit log.

## Fine-tuning data

- **IndicSynth** (spoof) — Sharma, Ekbote & Gupta, ACL 2025
- **Kathbath / IndicSUPERB** (bonafide) — AI4Bharat; the source real-speech dataset
  IndicSynth's synthetic clips were originally generated from, covering the same
  12 Indian languages
- **MLAAD** and **SpoofCeleb** — broader multi-language / multi-TTS-architecture
  spoof coverage, integrated in a separate development track

`preprocessing/prepare_combined_csv.py` streams and balances IndicSynth/Kathbath
directly from Hugging Face into one combined `train.csv`/`val.csv`.

## Scoring internals

- `detection/aasist_wrapper.py` — single-window (`predict`) and multi-window
  (`predict_windowed`, mean or max aggregation) scoring over a clip.
- `detection/dual_window_ewma.py` — fast/slow exponentially-weighted smoothing
  over `predict_windowed`'s per-window scores, combining both traces' peaks
  rather than their final value, so a genuine mid-clip spike that decays later
  isn't silently erased. Wired into both the local test pipeline
  (`run_full_pipeline.py`) and the live API (`/verify_call`).
- `risk_engine/fusion.py` — equally-weighted fusion of the spoof score and
  identity mismatch into the final 0–100 risk score.
- `risk_engine/audit_log.py` — SHA-256 hash-chained, append-only verdict log.
  Each entry's hash depends on the previous entry's hash, so tampering with or
  deleting a past entry breaks every hash after it. Only the scored verdict is
  stored — no raw audio or voice embeddings.

**Not currently in the active pipeline:** `detection/replay_detector.py` (a
coarse live/replay/converted heuristic) is present in the repo but no longer
imported or called anywhere — scoped out in favor of the two-signal design
above. Left in place as dead code rather than deleted, in case it's revisited.

## Setup

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

**Model weights are not included in this repo** (too large for git). To obtain
the fine-tuned AASIST-L checkpoint:
1. Clone [AASIST](https://github.com/clovaai/aasist) separately for the base
   pretrained checkpoint.
2. Run `detection/finetune_aasist.py` against the fine-tuning data above to
   produce `AASIST-L_finetuned_indic.pth` yourself, or contact the team
   directly for the pre-fine-tuned checkpoint.
3. Place it at `detection/aasist_repo/models/weights/AASIST-L_finetuned_indic.pth`.

ECAPA-TDNN weights download automatically on first run via SpeechBrain — no
manual step needed.

Test audio is also not included in this repo (excluded via `.gitignore`) —
supply your own 16kHz mono `.wav` clips in `test_audio/` to run the demo below.

### Run the API

```bash
cd Svara
uvicorn api.main:app --reload
```

Run this from the `Svara/` root — the config, checkpoint, and audit-log paths
inside `api/main.py` are relative to that directory, not `detection/`.

Endpoints:
- `POST /enroll` — enroll a reference voiceprint from one or more clips
  (averaged for stability)
- `POST /verify_call` — score an incoming call against both signals, log the
  verdict. Returns `400` if called before any identity has been enrolled.
- `GET /health` — health check

In a second terminal, from `Svara/`:
```bash
curl http://127.0.0.1:8000/health
curl -X POST -F "files=@test_audio/<a_real_voice_clip>.wav" http://127.0.0.1:8000/enroll
curl -X POST -F "file=@test_audio/<a_test_clip>.wav" http://127.0.0.1:8000/verify_call
```
Returns a real JSON verdict: spoof score, speaker similarity, fused 0–100 risk
score, tier, and a hash chained into the audit log.

## Why this differs from existing solutions

Most commercial call-deepfake detectors ship spoof detection alone, or spoof
detection plus basic caller-metadata rules. Fusing spoof detection with
claimed-identity speaker verification as two independently-computed signals —
specifically fine-tuned for Indian languages rather than adapted from an
English/Mandarin base — is the actual differentiation. We can defend that claim
with the test results above, not just the architecture diagram.

## Credits / Attribution

- **AASIST** — Jung et al., "AASIST: Audio Anti-Spoofing using Integrated
  Spectro-Temporal Graph Attention Networks" (2021).
  [github.com/clovaai/aasist](https://github.com/clovaai/aasist). Source code
  vendored under `detection/aasist_repo/` per its original license (see
  LICENSE/NOTICE in that folder).
- **ECAPA-TDNN** — Desplanques et al. (2020), via
  [SpeechBrain](https://speechbrain.github.io/) pretrained checkpoint
  (`speechbrain/spkrec-ecapa-voxceleb`).
- **IndicSynth** — Sharma, Ekbote & Gupta (ACL 2025).
- **Kathbath / IndicSUPERB** — AI4Bharat.
- **MLAAD** — Müller et al., "MLAAD: The Multi-Language Audio Anti-Spoofing
  Dataset" (IJCNN 2024).
- **SpoofCeleb** — Jung et al. (2024), a VoxCeleb1-derived spoof/SASV dataset.

## Status

Working end-to-end pipeline: fine-tuned spoof detection (dual-window
EWMA-smoothed) → speaker verification → two-signal risk fusion → hash-chained
audit log → FastAPI endpoints, tested against real recordings and multiple
independent voice-cloning tools.

**This is a checkpoint in progress, not a final model.** Current single-digit
EER on held-out fine-tuning data is a snapshot from ongoing training, expected
to improve with further epochs and broader dataset coverage. Cross-domain
generalization to unseen synthesis tools was directly tested — including a
documented miss, not just a win — and closing that gap through continued
fine-tuning is active, ongoing work, not a solved problem we're claiming credit
for prematurely.