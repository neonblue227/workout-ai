<div align="center">

# 🏋️ Move-UP

**Gamified Rehab Companion — NSC 2026 Submission**

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![uv](https://img.shields.io/badge/Managed%20by-uv-DE5FE9?style=for-the-badge)](https://docs.astral.sh/uv/)
[![Status](https://img.shields.io/badge/Status-Week%201%20·%20P1%20Starting-yellow?style=for-the-badge)]()
[![Deadline](https://img.shields.io/badge/NSC%20Deadline-30%20Sep%202026-red?style=for-the-badge)]()

</div>

---

## 📖 About The Project

**Move-UP** is a gamified rehabilitation companion built for the **Thailand National Software Contest (NSC) 2026**. It combines clinical-grade computer-vision form correction with Duolingo-inspired game mechanics (XP, streaks, daily quests, leaderboards) to address the #1 problem in home physical therapy: **adherence**.

Research shows 30–50% of patients drop out of home exercise programs within 4–6 weeks. Existing clinical apps (Kaia Health, Sword, Hinge) achieve form-correction accuracy but lack engagement; gamified fitness apps (Ring Fit, Just Dance) are engaging but lack clinical validity. **Move-UP sits in the gap**: clinical accuracy + sustained engagement, designed for Thai rehab patients.

### 🎯 Mission

Help home-rehab patients (knee rehab, low-back pain) stay consistent with their PT-prescribed exercises by making correct form measurable, daily practice rewarding, and progress visible — without compromising clinical safety.

### 👥 Target Users (Phase 1)

| Tier      | Who                                                              | Use case                                                           |
| --------- | ---------------------------------------------------------------- | ------------------------------------------------------------------ |
| Primary   | Patients with PT-prescribed home exercise programs               | Daily rehab session + streak + XP                                  |
| Secondary | Physical therapists                                              | Dashboard for remote patient adherence + progress                  |
| Tertiary  | Family members of patients                                       | Supporter role in social features                                  |

> **Not in Phase 1:** casual fitness users, gym athletes — focus is rehab for clinical positioning.

### ✨ Key Features

| Feature                          | Description                                                                          |
| -------------------------------- | ------------------------------------------------------------------------------------ |
| 🎥 **Real-time form scoring**    | Mobile camera → server inference → per-rep score in <200ms                           |
| 🎯 **Mistake localization**      | Identifies which joint at which moment ("knee cave at rep 5, second 2.3")            |
| 💬 **Personalized LLM coach**    | Thai/English coaching after each session, contextual to user history                 |
| 🏅 **XP, levels, streaks**       | Anti-gaming design: form quality gates XP, streak loss is soft (not 0-reset)        |
| 📅 **Daily quests**              | 3 per day — required (PT prescription) / variety / bonus skill                       |
| 🏆 **Leaderboard + challenges**  | Weekly league system + 1-on-1 friend challenges (privacy-aware)                      |
| 📊 **PT dashboard**              | Remote adherence tracking + per-exercise breakdown                                   |

### 🏃 Phase 1 Exercises (locked)

1. **Squat** — correctness, depth, knee tracking (knee rehab)
2. **Sit-to-stand** — general mobility, rehab indicator
3. **Single-leg stance / heel-toe walk** — balance, fall prevention

### 📐 Architecture (high level)

```
[Mobile thin client] ──WebRTC video──▶ [Laptop AI server]
        ▲                                       │
        │  WebSocket (events) + REST (state)    │
        └◀──────────────────────────────────────┘
```

- **Mobile:** React Native — camera capture, skeleton overlay, game UI, audio/haptic feedback
- **Server:** FastAPI + aiortc — pose inference, form scoring, LLM coach, session storage
- **Storage:** PostgreSQL (sessions, users, exercises) + Redis (live session state)

For the full design — model choices, datasets, validation methodology, risk register, and 21-week timeline — see the [**Master Plan**](docs/plan/Move-UP%20%20NSC%202026%20%E2%80%94%20Master%20Plan.md).

---

## 🎯 Success Criteria

| Track          | Metric                                            | Target                       |
| -------------- | ------------------------------------------------- | ---------------------------- |
| Technical      | End-to-end latency                                | < 200 ms                     |
| Technical      | Pose accuracy vs PT ground truth                  | ≥ 85 %                       |
| Clinical       | Spearman ρ vs PT scoring (Kaia 2021 benchmark)    | ≥ 0.80                       |
| Clinical       | ICC(2,1) continuous scoring                       | ≥ 0.75                       |
| Clinical       | Cohen's κ binary mistake classification           | ≥ 0.61 (substantial)         |
| Engagement     | Day-7 retention                                   | ≥ 60 %                       |
| Engagement     | Mean streak length                                | ≥ 5 days                     |

---

## 📁 Project Structure

```
workout-ai/
├── src/                          # Legacy prototype (MediaPipe Holistic, Tkinter)
│   ├── app.py                    #   GUI app for recording + GIF generation
│   ├── record_keypoints.py       #   Webcam capture → JSON keypoints
│   └── generate_gif.py           #   JSON keypoints → animated GIF
├── model/                        # Data pipeline + scoring experiments
│   ├── extract_pipeline.py       #   Batch extract keypoints from raw videos
│   ├── feature_extractor.py
│   ├── posture_dataset.py
│   ├── scoring_model.py
│   └── train_scoring_model.ipynb
├── utils/                        # Shared utilities (14 modules)
│   ├── angle.py, joint_angles.py
│   ├── draw_pose.py, draw_face.py, draw_hand.py, draw_angle.py, draw_info.py
│   ├── keypoint_extractor.py, keypoint_recorder.py
│   ├── gif_generator.py, fps_calibration.py
│   ├── visibility_color.py, file_utils.py
│   └── __init__.py
├── data/
│   ├── video/                    # Raw recordings
│   ├── keypoint/                 # Extracted JSON keypoints
│   └── gif/                      # Demo visualizations
├── docs/
│   ├── progression.md            # Live project status
│   └── plan/
│       └── Move-UP NSC 2026 — Master Plan.md
├── main.py
├── pyproject.toml                # uv-managed dependencies
├── uv.lock
├── CLAUDE.md                     # Guidance for Claude Code sessions
└── README.md
```

> ⚠️ The current `src/` code is a **pre-NSC prototype** built around MediaPipe Holistic. The production server pipeline (RTMPose-l + MotionBERT + ST-GCN + Qwen3-8B) starts in Phase P2 (Week 4, 26 May 2026) and will live under `apps/server/` in a monorepo layout.

---

## 🗓️ 21-Week Roadmap

| Phase                                  | Weeks       | Dates                       | Focus                                                  |
| -------------------------------------- | ----------- | --------------------------- | ------------------------------------------------------ |
| **P1 · Clinical Dataset & Rules**      | w1–3        | 5–25 May 2026               | Lock scope, rules.yaml, PT advisor sign-off            |
| **P2 · Pose Pipeline (server)**        | w4–7        | 26 May–22 Jun 2026          | FastAPI + aiortc + RTMPose, API contract freeze        |
| **P3 · Movement Logic + ST-GCN**       | w6–11       | 9 Jun–13 Jul 2026           | Hybrid rule + ST-GCN, supervised contrastive training  |
| **P4 · Mobile Real-time WebRTC**       | w8–11       | 23 Jun–20 Jul 2026          | RN client, live skeleton overlay, edge cases           |
| **P5 · Feedback + LLM Coach**          | w11–13      | 14 Jul–3 Aug 2026           | Visual/audio/haptic + Qwen3-8B coaching                |
| **P6 · Analytics Dashboard**           | w13–17      | 28 Jul–31 Aug 2026          | History, charts, streak counter, PDF export            |
| **UI Polish + UAT**                    | w15–19      | 11 Aug–14 Sep 2026          | Bug bash, accessibility, performance                   |
| **P7 · Clinical Validation**           | w17–19      | 25 Aug–14 Sep 2026          | n ≥ 25, 2 PT raters, stratified metrics                |
| **NSC Submission Prep**                | w19–21      | 8–30 Sep 2026               | Report, demo video, clean repo, submit 2 days early    |

Live progress: [`docs/progression.md`](docs/progression.md).

---

## ⚡ Quick Start (using uv)

This project is managed with [**uv**](https://docs.astral.sh/uv/) — a fast, single-binary Python package & project manager that replaces `pip`, `venv`, `pip-tools`, and `pyenv`. The repo ships a `pyproject.toml` and `uv.lock` so installs are fully reproducible.

### Prerequisites

- **Python 3.12+** (uv can install this for you — see below)
- **Webcam** for live capture
- **macOS / Linux / Windows**

### 1. Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via Homebrew / pipx / cargo / pip
brew install uv
```

Verify the install:

```bash
uv --version
```

### 2. Clone the repository

```bash
git clone https://github.com/yourusername/workout-ai.git
cd workout-ai
```

### 3. Sync dependencies

`uv sync` reads `pyproject.toml` + `uv.lock`, creates `.venv/` automatically, installs the right Python version if missing, and pins every package to the locked version.

```bash
uv sync
```

That's it — no `python -m venv`, no `pip install -r ...`, no activation step required when you run via `uv run`.

### 4. Run the prototype

Use `uv run` to execute scripts inside the project's environment without manually activating it:

```bash
# Webcam capture + MediaPipe keypoint extraction → JSON
uv run python src/record_keypoints.py

# Tkinter GUI (record video + generate GIF)
uv run python src/app.py

# Convert saved JSON keypoints → animated GIF
uv run python src/generate_gif.py

# Batch extract keypoints from raw videos in data/video/
uv run python model/extract_pipeline.py
```

If you prefer the classic activated-shell workflow:

```bash
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows
python src/record_keypoints.py
```

### 5. Common uv tasks

| Task                                    | Command                                  |
| --------------------------------------- | ---------------------------------------- |
| Install/refresh deps from lockfile      | `uv sync`                                |
| Add a new dependency                    | `uv add <package>`                       |
| Add a dev-only dependency               | `uv add --dev <package>`                 |
| Remove a dependency                     | `uv remove <package>`                    |
| Upgrade a package                       | `uv lock --upgrade-package <package>`    |
| Run any script in the env               | `uv run python <script>`                 |
| Run a Jupyter notebook                  | `uv run jupyter lab`                     |
| Show the resolved dependency tree       | `uv tree`                                |

> **Note:** Keep `uv.lock` committed. It's the source of truth for reproducible builds and CI.

---

## 🤝 Contributing

This is an active NSC 2026 submission with a hard deadline of **30 Sep 2026**. Contributions are welcome but must respect the locked Phase 1 scope:

- ✅ 3 exercises: squat, sit-to-stand, single-leg stance
- ✅ 5 game mechanics: XP/levels, streaks, daily quests, achievements, leaderboard
- ❌ No yoga, deadlifts, narrative mode, in-app currency, or custom avatars in Phase 1

Workflow:

1. Fork the project
2. Create your feature branch (`git checkout -b feature/your-feature`)
3. Run `uv sync` and verify your changes locally
4. Update [`docs/progression.md`](docs/progression.md) with what changed
5. Open a Pull Request

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 📧 Contact

- Project Link: [https://github.com/yourusername/workout-ai](https://github.com/yourusername/workout-ai)
- Author: นักเรียน & คุณครู เสรีคอม, Chitralada School

---

<div align="center">

**Move-UP — สู้เพื่อ NSC 2026 🇹🇭**

*Built for healthier home rehab, one rep at a time.*

</div>
