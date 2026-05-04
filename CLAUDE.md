# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Move-UP** is a gamified rehab/exercise companion submitted to **Thailand National Software Contest (NSC) 2026**. It combines clinical-grade computer-vision form correction with Duolingo-inspired game mechanics to solve the #1 problem in home rehabilitation: **adherence**.

- **Target user (Phase 1):** Home-rehab patients (knee rehab, low-back pain) prescribed exercises by a PT.
- **Architecture:** Mobile thin client (React Native, camera + UI + game state) ↔ Laptop AI server (FastAPI, inference, scoring, LLM coach). Communication over WebRTC (video) + WebSocket (events) + REST (state).
- **3 locked exercises (Phase 1):** Squat, Sit-to-stand, Single-leg stance / heel-toe walk.
- **Deadline:** 30 ก.ย. 2026 (NSC submission).

The authoritative project plan is [`docs/plan/Move-UP NSC 2026 — Master Plan.md`](docs/plan/Move-UP%20%20NSC%202026%20%E2%80%94%20Master%20Plan.md). Current progress is tracked in [`docs/progression.md`](docs/progression.md).

> ⚠️ **Legacy code notice:** The Python files in `src/` (MediaPipe Holistic, Tkinter GUI) are **prototype/foundation work from before the NSC plan**. The production pipeline will be **server-side** (RTMPose-l + MotionBERT + ST-GCN + Qwen3-8B) and is scheduled to start in **Week 4 (26 พ.ค. 2026, Phase P2)**. Treat existing code as reference for keypoint extraction concepts; do not extend it without checking against the Master Plan first.

---

## 🔁 Mandatory Claude Workflow

### After completing any implementation task — ALWAYS update `docs/progression.md`

Whenever you finish implementing, modifying, or removing functionality (code, config, tests, infra, dataset prep, model training, etc.), you **must** reflect the change in `docs/progression.md` before considering the task complete. This file is the single source of truth for "where the project is right now" and is read at the start of every session.

**What to update:**
1. **Phase status table** — flip checkboxes / change emoji status (⬜ → 🟡 → ✅) for the affected phase or sub-task.
2. **Week 1 action items** (or whichever week the work belongs to) — tick off completed items, add newly discovered ones.
3. **"สิ่งที่ทำเสร็จแล้ว" section** — if a new module, package, or capability was added, list it with the file path.
4. **Hard cutoffs** — if a deliverable now satisfies a cutoff (e.g. `rules.yaml` signed off → w3 cutoff), note it.
5. **Tech stack section** — only if a new library/model/service was introduced or removed.

**Format rules:**
- Keep Thai voice and emoji conventions consistent with the existing file.
- Update the "*Document version*" footer line if the change is substantial (new phase entered, cutoff hit, scope change).
- Don't duplicate Master Plan content — link to it for detail.
- Don't write speculative progress — only mark something done when it's actually done and verifiable.

**When NOT to update:** pure exploration, read-only research, answering questions, formatting fixes that don't change behavior.

If you skip the progression update, you are leaving the project in an inconsistent state for the next session. Treat it as part of "Definition of Done", not optional housekeeping.

---

## Development Commands

> The commands below are for the **legacy prototype**. Server commands (FastAPI, aiortc, RTMPose) will be added when P2 begins (Week 4).

### Run Legacy Prototype
```bash
python src/record_keypoints.py   # webcam capture + MediaPipe keypoints → JSON
python src/app.py                # Tkinter GUI (record + GIF)
python src/generate_gif.py       # JSON keypoints → animated GIF
python model/extract_pipeline.py # batch extract keypoints from raw videos
```

### Install Dependencies (current prototype)
```bash
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### Process Keypoint Data
Run `src/data_tranfrom.ipynb` in Jupyter to batch-process videos and extract keypoints to JSON.

### Future commands (P2 onwards, not yet implemented)
```bash
# Server (Python 3.12+, FastAPI + aiortc + mmpose)
uvicorn apps.server.main:app --reload

# Mobile (React Native bare workflow)
cd apps/mobile && npx react-native run-ios
```

---

## Architecture

### Target Architecture (per Master Plan §3-4)

```
[Mobile RN client] ──WebRTC video──▶ [Laptop FastAPI server]
       ▲                                      │
       │                                      ├─ RTMPose-l       (2D keypoints, ~15ms)
       │   WebSocket events                   ├─ MotionBERT      (2D→3D, ~10ms/2 frames)
       │ ◀──────────────────────────────────  ├─ ST-GCN+SupCon   (form scoring, ~5ms/rep)
       │                                      └─ Qwen3-8B (MLX)  (LLM coach, 1-3s/session)
       │   REST (state)                              │
       └──────────────────────────────────▶ [Postgres 16 + Redis 7]
```

**Per-frame budget:** ~30-40ms inference + 20-40ms network = 50-80ms perceived; end-to-end target < 200ms.

### Legacy Prototype Pipeline (current `src/`)

1. **Input:** Webcam capture via OpenCV
2. **Detection:** MediaPipe Holistic — 33 pose + 478 face + 42 hand = 553 landmarks
3. **Normalization:** Normalized (0-1) → pixel coordinates
4. **Analysis:** Joint angles via 3-point geometry (shoulder-elbow-wrist etc.)
5. **Output:** Real-time skeleton overlay + angle feedback + JSON keypoint dump

**Key legacy functions:**
- `normalized_to_pixel()` — MediaPipe coords → pixel space
- `angle_between_points()` — 3-landmark angle calculation
- `keypoints_to_dict()` — landmark → named dict

**Legacy data layout:**
```
data/
├── video/     # raw recordings
├── keypoint/  # extracted JSON keypoints
└── gif/       # demo visualizations
```

---

## Tech Stack

### Production target (Master Plan)
- **Mobile:** React Native 0.74+ (bare), `react-native-webrtc`, `@shopify/react-native-skia`, `react-native-reanimated`, `lottie-react-native`, `zustand`, `@tanstack/react-query`, `nativewind`
- **Server:** Python 3.12+, FastAPI + uvicorn, aiortc, PyAV, PyTorch 2.4+ (MPS), mlx + mlx-lm, mmpose/mmdetection, opencv-python, pydantic v2, redis-py, asyncpg + sqlalchemy
- **Storage:** PostgreSQL 16, Redis 7
- **Models:** RTMPose-l, MotionBERT, ST-GCN (Karlov 2024 supervised contrastive), Qwen3-8B
- **Datasets:** NTU RGB+D 120 (pretrain), KIMORE + UI-PRMD + REHAB24-6 (fine-tune), BEDLAM/InfiniteForm (augmentation)

### Current prototype
- Python 3.8+, MediaPipe 0.10.x, OpenCV, NumPy, Matplotlib, Jupyter
- JSON keypoint storage

---

## Critical Path & Hard Cutoffs

```
P1 Rules (w3) → P2 Pose API (w7) → P3 ST-GCN (w11) → P7 Validation (w17-19) → NSC Submit (w21)
```

| Cutoff | Date | Required artifact |
| ------ | ---- | ----------------- |
| End w3  | 25 พ.ค. | `rules.yaml` signed off by PT advisor |
| End w7  | 22 มิ.ย. | `pose-api-spec.md` locked |
| End w11 | 20 ก.ค. | ST-GCN ≥80% on KIMORE held-out |
| End w13 | 3 ส.ค.  | End-to-end live demo working |
| End w16 | 31 ส.ค. | Validation protocol approved |
| End w19 | 14 ก.ย. | Validation report draft |
| End w21 | 30 ก.ย. | Submitted to NSC portal (2 days early) |

**Slipping any of these consumes the buffer for NSC submission.** Flag risk early, don't quietly defer.

---

## Project Conventions

- **Repo structure target:** monorepo with `apps/mobile`, `apps/server`, `packages/schemas`. The current flat `src/` + `model/` layout is legacy and will migrate during P2.
- **Language:** Code/comments in English. Docs (`docs/`) primarily Thai with English technical terms — match existing voice.
- **Scope discipline:** Phase 1 is locked at 3 exercises + 5 game mechanics. Do not propose adding yoga / deadlifts / narrative mode / custom avatars / in-app currency without an explicit user decision to amend the Master Plan.
- **Anti-gaming guard rails:** Any change to scoring/XP must respect `form_multiplier` floor 0.3, min rep duration 1.5s, and quality-over-quantity rewards (Master Plan §4.5.6).
- **Clinical claims:** Never assert clinical validity in code/docs without a citation to the Master Plan §10 metrics or an advisor sign-off. The bar is r ≥ 0.80 vs PT (Kaia 2021 benchmark).

---

## Status Snapshot

**As of 4 พ.ค. 2026:** Entering **Week 1 — Phase P1 (Clinical Dataset & Rule Definition)**. Legacy MediaPipe prototype exists and is usable for capturing reference video; no NSC-track code has been written yet. See `docs/progression.md` for live status.
