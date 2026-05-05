<div align="center">

# 🏋️ Move-UP

**Gamified rehab/exercise companion สำหรับ NSC 2026**

[![Status](https://img.shields.io/badge/สถานะ-Week%201%20(P1%20Starting)-yellow?style=for-the-badge)]()
[![Deadline](https://img.shields.io/badge/Deadline-30%20ก.ย.%202026-red?style=for-the-badge)]()

</div>

---

## 📋 ภาพรวมโครงการ

**Move-UP** คือ mobile game แนว Duolingo-inspired ที่ใช้ AI ตรวจสอบและให้คำแนะนำท่ากายภาพบำบัดแบบ real-time
- **Target user (Phase 1):** ผู้ป่วย home rehab (knee rehab, low-back pain) — gamification แก้ปัญหา adherence
- **Architecture:** Mobile thin client (camera + UI) + Laptop AI server (inference + scoring + LLM)
- **3 ท่าใน Phase 1:** Squat, Sit-to-stand, Single-leg stance / heel-toe walk
- **Submission:** Thailand National Software Contest 2026 — Deadline 30 ก.ย. 2026

> ดูแผนเต็มได้ที่ [`docs/plan/Move-UP NSC 2026 — Master Plan.md`](./plan/Move-UP%20%20NSC%202026%20%E2%80%94%20Master%20Plan.md)

---

## 🗓️ สถานะปัจจุบัน

```
วันนี้: 4 พ.ค. 2026  →  กำลังเข้าสู่ Week 1 (P1: Clinical Dataset & Rule Definition)
เหลือเวลา: ~21 สัปดาห์ ก่อน NSC deadline
```

| Phase | ช่วงเวลา | สถานะ |
| ----- | -------- | ----- |
| **P0 · Prototype Foundation** (legacy) | ก่อน w1 | ✅ Done (โค้ด MediaPipe เดิม — จะปรับ/แทนที่ใน P2) |
| **P1 · Clinical Dataset & Rules** | w1-3 (5-25 พ.ค.) | 🟡 เริ่มต้น |
| **P2 · Pose Pipeline (Server)** | w4-7 (26 พ.ค.-22 มิ.ย.) | ⬜ ยังไม่เริ่ม |
| **P3 · Movement Logic + ST-GCN** | w6-11 (9 มิ.ย.-13 ก.ค.) | ⬜ ยังไม่เริ่ม |
| **P4 · Mobile Real-time WebRTC** | w8-11 (23 มิ.ย.-20 ก.ค.) | ⬜ ยังไม่เริ่ม |
| **P5 · Feedback + LLM Coach** | w11-13 (14 ก.ค.-3 ส.ค.) | ⬜ ยังไม่เริ่ม |
| **P6 · Analytics Dashboard** | w13-17 (28 ก.ค.-31 ส.ค.) | ⬜ ยังไม่เริ่ม |
| **UI Polish + UAT** | w15-19 (11 ส.ค.-14 ก.ย.) | ⬜ ยังไม่เริ่ม |
| **P7 · Clinical Validation** | w17-19 (25 ส.ค.-14 ก.ย.) | ⬜ ยังไม่เริ่ม |
| **NSC Submission Prep** | w19-21 (8-30 ก.ย.) | ⬜ ยังไม่เริ่ม |

---

## ✅ สิ่งที่ทำเสร็จแล้ว (Prototype Foundation)

โค้ดเดิมที่มีก่อนเริ่มแผน NSC — ใช้ MediaPipe Holistic บน Python (standalone) ส่วนนี้จะถูกปรับเปลี่ยนเป็น server pipeline ใหม่ใน **P2 (RTMPose-l + MotionBERT)** แต่ asset ที่เก็บไว้ (videos, keypoints) ยังใช้งานได้

#### Recording & Keypoint Extraction
| ฟีเจอร์ | คำอธิบาย | ไฟล์หลัก |
| ------- | -------- | -------- |
| 🎥 บันทึกวิดีโอ + Keypoints | MediaPipe Holistic (Pose 33, Face 478, Hands 42) | `src/record_keypoints.py` |
| 🖥️ GUI Application | Tkinter app สำหรับบันทึก + สร้าง GIF | `src/app.py` |
| 🎬 GIF จาก Keypoints | แปลง JSON keypoints เป็น animated GIF | `src/generate_gif.py` |
| 📊 Data Pipeline | Extract keypoints จากวิดีโอ raw | `model/extract_pipeline.py` |

#### Utility Modules (14 modules)
`angle.py`, `draw_pose.py`, `draw_face.py`, `draw_hand.py`, `draw_angle.py`, `draw_info.py`, `keypoint_extractor.py`, `keypoint_recorder.py`, `gif_generator.py`, `fps_calibration.py`, `visibility_color.py`

#### PoC Baseline Scaffolding (UI-PRMD via cpm121/uiprmd-full10ex) ✅
Pipeline พิสูจน์ระบบ end-to-end บน dataset สาธารณะ ก่อนย้ายไปสาย RTMPose+ST-GCN ใน P3 — **ดาวน์โหลด + loader ใช้ได้แล้ว**
| ฟีเจอร์ | คำอธิบาย | ไฟล์หลัก |
| ------- | -------- | -------- |
| 📥 Fetch UI-PRMD (Vicon pickle) | ดาวน์โหลด `uiprmd_all_exercises.pkl` จาก Kaggle mirror `cpm121/uiprmd-full10ex` แล้วพิมพ์สรุป (Ex1 squat 900 clips, Ex5 sit-to-stand 1008 clips) | `scripts/fetch_uiprmd.py` |
| 🧱 Pickle dataset loader | โหลด pickle, แบนข้อมูล `(50, 39, 3) → (50, 117)`, รองรับ stratified 2-way (`train_test_split`) และ 3-way (`train_val_test_split`, 64/16/20) split | `model/uiprmd_pickle_dataset.py` |
| 🧪 Dataset split tests | 5 tests ครอบคลุม: ขนาด, alignment, no-overlap, determinism, stratification ratios | `tests/test_uiprmd_split.py` |
| 📦 Dependencies | เพิ่ม `kaggle`, `numpy` ใน `pyproject.toml` (description ก็แก้แล้ว) | `pyproject.toml` |
| 🗄️ Deprecated (kept for ref) | Kinect-25 → MediaPipe adapter — เขียนไว้สำหรับ raw-txt mirror, mirror จริงใช้ Vicon pickle จึงไม่ได้ใช้ | `model/kinect_to_features.py` |
| 🧠 PyTorch LSTM scorer | `nn.Module` regressor — input (B, 50, num_features), sigmoid output ∈ [0,1] | `model/lstm_torch.py` |
| 🏃 Training driver + ablation knobs | seed/device helpers, `fit()` with early stopping, MAE/MSE/Spearman, CLI flags สำหรับ seq-len + joint-subset | `model/training.py`, `scripts/train_baseline.py` |
| 🧪 Unit tests (13 tests) | split (5), model (4), training (13: seed, metrics, train/eval, fit) | `tests/test_*.py` |

**Dataset สรุป**: 1,908 clips (squat 900 + sit-to-stand 1008), shape `(N, 50_frames, 117_features)`, scores ต่อเนื่อง 0.53-0.97 (Vicon มากกว่าจะเป็น Kinect — paper §UI-PRMD)

**Baseline ที่ได้**: PyTorch LSTM (h64-32) บน Ex1+Ex5 stratified 64/16/20 split, seed=42, 76 epochs (early stop) — TEST mae=0.029, mse=0.002, **Spearman ρ=0.818** (above Kaia 2021 clinical benchmark of ≥0.80 on healthy-subject UI-PRMD; not transferable to rehab patients). Run dir: `model/runs/2026-05-05_both_seq50_jall_h64-32_s42/`

**Ablation results** (`model/runs/`):
| Config | Patience | Joints | Frames | Test MAE | **Test ρ** | Δ ρ |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 10 | 39 | 50 | 0.029 | **0.818** | — |
| seq=25 (all joints) | 10 | 39 | 25 | 0.037 | **0.655** | −16.3 pp |
| Random 20/39 | 50 | 20 | 50 | 0.031 | **0.740** | −7.8 pp |
| Top-variance 10/39 | 50 | 10 | 50 | 0.032 | **0.766** | −5.2 pp |

> 💡 **Findings**: (1) temporal context dominates — halving sequence length costs ~16pp; halving markers costs ~8pp. (2) top-variance 10 markers BEATS random 20 markers (+2.6pp) — quality > quantity. (3) joint-subset configs hit a "predict-mean" local minimum for ~50 epochs before learning starts; default `--patience 10` kills them prematurely → use `--patience 50` for joint-subset ablations.

> ⚠️ UI-PRMD เป็น **skeleton-only** (ไม่มี RGB video) — feature evaluation ทำได้เฉพาะ temporal/joint-subset ablation, ไม่ใช่ named-angle ablation production pipeline ยังเป็น RTMPose ตามแผน

#### โครงสร้างข้อมูล
```
data/
├── video/     # วิดีโอที่บันทึก
├── keypoint/  # JSON keypoints
└── gif/       # GIF สาธิต
```

> ⚠️ **หมายเหตุ:** ตามแผน Master Plan การประมวลผลจริงใน production จะย้ายไป **server-side** (FastAPI + aiortc + RTMPose-l) ส่วน MediaPipe Holistic จะใช้เป็น reference/validation tool เท่านั้น

---

## 🎯 Tech Stack (ตามแผน NSC)

### Mobile thin client (React Native)
- `react-native-webrtc` (video upstream)
- `@shopify/react-native-skia` (skeleton overlay)
- `react-native-reanimated` 3 + `lottie-react-native` (game animations)
- `zustand` + `@tanstack/react-query` (state)
- `nativewind` (Tailwind for RN)

### Server (Laptop, Python 3.12+)
- **FastAPI + uvicorn** (REST + WebSocket)
- **aiortc** (WebRTC server-side)
- **PyTorch 2.4+ (MPS)** — model training/inference
- **mlx + mlx-lm** — LLM inference (Apple Silicon)
- **mmpose / mmdetection** — RTMPose pretrained
- **PostgreSQL 16** + **Redis 7** (sessions, state)

### AI Models (4-stage pipeline)
| Model | หน้าที่ | Latency |
| ----- | ------- | ------- |
| **RTMPose-l** | 2D keypoints (26 จุด รวมนิ้วเท้า) | ~15ms |
| **MotionBERT** | 2D → 3D lifting, view-invariant | ~10ms / 2 frames |
| **ST-GCN + Supervised Contrastive** (Karlov 2024) | Form scoring + mistake localization (GradCAM) | ~5ms / rep |
| **Qwen3-8B** (MLX) | Personalized coaching ภาษาไทย/อังกฤษ | 1-3s / session |

### Game design layer (Move-UP unique)
- XP + Levels (log curve, anti-gaming floor 0.3×)
- Streaks (soft loss + weekly freeze)
- Daily quests (required / variety / bonus)
- Achievements (~30 ใน Phase 1)
- Weekly leaderboard + friend challenges

---

## 🔨 งานสำคัญสัปดาห์นี้ (Week 1: 5-11 พ.ค. 2026)

### Mandatory
1. ✅ **Lock scope** — ยืนยัน 3 ท่า: Squat, Sit-to-stand, Single-leg stance
2. ⬜ **Lock game mechanics** — เขียน `game-spec.md`
3. ⬜ **Contact PT advisor** — primary + backup, นัด 30-60 min session
4. ⬜ **Set up infrastructure** — monorepo (`apps/mobile`, `apps/server`, `packages/schemas`)
5. ⬜ **Hardware verification** — RTMPose-l ≥30 FPS, Qwen3-8B บน MLX, thermal test

### Nice to have
- ✅ UI-PRMD ดาวน์โหลดเรียบร้อย (1,908 clips, Ex1+Ex5); KIMORE, REHAB24-6 ยังไม่ได้
- ⬜ อ่าน Karlov 2024 (arXiv 2403.02772)
- ⬜ Clone reference repos (`yakupzengin/fitness-trainer-pose-estimation`, `NgoQuocBao1010/Exercise-Correction`)
- ⬜ Thai PDPA consent flow draft

---

## 🛣️ Critical Path (ห้าม slip)

```
P1 Rules (w3) → P2 Pose API (w7) → P3 ST-GCN (w11) → P7 Validation (w17-19) → NSC Submit (w21)
```

### Hard cutoffs
- **w3:** Clinical rules signed off by advisor
- **w7:** Pose API contract locked
- **w11:** ST-GCN ≥80% accuracy on KIMORE
- **w13:** End-to-end live demo working
- **w16:** Validation protocol approved, testers recruited
- **w19:** Validation report draft complete
- **w21:** Submitted to NSC portal (2 days early)

---

## 🎯 Success Criteria

### Technical
- End-to-end latency < 200ms
- Pose accuracy ≥ 85% vs PT ground truth

### Clinical (ตามมาตรฐาน Kaia 2021)
- Spearman ρ ≥ 0.80 vs PT scoring
- ICC(2,1) ≥ 0.75
- Cohen's κ ≥ 0.61 (substantial agreement)
- F1 ≥ 0.85 (mistake detection)

### Engagement (gamification working)
- Day-7 retention ≥ 60% ในกลุ่มทดสอบ
- Mean streak length ≥ 5 days
- ≥ 80% ของ users กลับมา 3+ วันใน 7 วันแรก

### NSC submission
- Final report (PDF) + demo video (MP4 + Thai captions) + clean repo + clinical validation evidence + engagement data

---

## 🚀 วิธีใช้งาน (Prototype ปัจจุบัน — pre-NSC pipeline)

### บันทึกวิดีโอพร้อม Keypoints
```bash
python src/record_keypoints.py
```

### ใช้ GUI Application
```bash
python src/app.py
```

### สร้าง GIF จาก Keypoints
```bash
python src/generate_gif.py
```

### Extract Keypoints จากวิดีโอ Raw
```bash
python model/extract_pipeline.py
```

### PoC Baseline (UI-PRMD via Vicon pickle)
```bash
# 1. ตั้งค่า Kaggle API token ที่ ~/.kaggle/kaggle.json (chmod 600)
# 2. ดาวน์โหลด pickle (~107 MB) แล้วพิมพ์สรุป
uv run python scripts/fetch_uiprmd.py

# 3. ทดสอบ loader + stratified train/test split
uv run python model/uiprmd_pickle_dataset.py

# จะได้ X.shape = (1908, 50, 117), y.shape = (1908,) พร้อมป้อนให้ LSTM
# (เปลี่ยน scoring_model.create_scoring_model(num_features=117, sequence_length=50))
```

> ⚠️ Server stack (FastAPI + aiortc + RTMPose) จะเริ่ม setup ใน **Week 4 (26 พ.ค.)**

---

<div align="center">

**Move-UP — สู้เพื่อ NSC 2026 🇹🇭**

*Document version: aligned with Master Plan v1.0 (2 พ.ค. 2026) — last update 5 พ.ค. 2026 (PoC training driver + ablation table; baseline ρ=0.818)*

</div>
