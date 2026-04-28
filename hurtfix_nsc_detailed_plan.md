# hurtfix · NSC 2026 — แผนโครงการแบบละเอียด

> **ระยะเวลารวม:** 22 สัปดาห์ (28 เม.ย. – 30 ก.ย. 2026)
> **โครงสร้าง:** 2 lanes ขนานกัน — AI / Core track และ UI / Frontend track
> **Phases ทั้งหมด:** 11 phases + NSC submission prep

---

## 1. ภาพรวม timeline

| Week | ช่วงเวลา | AI / Core track | UI / Frontend track |
|------|---------|----------------|---------------------|
| 1-3 | 1-21 พ.ค. | P1 Dataset & Rules | (รอจน w2) |
| 2-5 | 8 พ.ค. - 4 มิ.ย. | — | UI Discovery + Design |
| 3-6 | 15 พ.ค. - 11 มิ.ย. | P2 Posture Pipeline | (overlap UI Discovery) |
| 5-8 | 29 พ.ค. - 25 มิ.ย. | (P3 เริ่ม w6) | UI Components (mock) |
| 6-10 | 5 มิ.ย. - 9 ก.ค. | P3 Movement Logic | (P4 เริ่ม w8) |
| 8-11 | 19 มิ.ย. - 16 ก.ค. | (P5 เริ่ม w10) | P4 Real-time integration |
| 10-12 | 3-30 ก.ค. | P5 Feedback Engine | (P6 เริ่ม w11) |
| 11-15 | 10 ก.ค. - 13 ส.ค. | (AI quiet + bug fix) | P6 Analytics Dashboard |
| 15-19 | 7 ส.ค. - 10 ก.ย. | (P7 เริ่ม w17) | UI Polish + UAT |
| 17-19 | 21 ส.ค. - 10 ก.ย. | P7 Clinical Validation | (parallel) |
| 19-22 | 4-30 ก.ย. | NSC Submission Prep (ทั้ง 2 lanes รวม) | |

---

## 2. หลักการออกแบบแผน

**Parallel lanes ลด total calendar time** — เมื่อ AI lane ใช้เวลา ~16 weeks (P1+P2+P3+P5+P7 พร้อม overlap) และ UI lane ใช้เวลา ~17 weeks (Discovery → Polish พร้อม overlap) ถ้าเดิน sequential รวมจะ ~30+ weeks เกิน deadline แน่นอน การ run parallel ลงได้ใน 22 weeks พอดี

**UI ไม่ต้องรอ AI พร้อมก่อน** — UI lane เริ่มจาก Discovery + Design ที่ทำงานบนกระดาษ (Figma) แล้วต่อด้วย Component build ที่ใช้ mock data ตามที่ AI lane จะส่งมาในอนาคต API contract ต้องล็อกตั้งแต่ w3-4 ไม่งั้น UI lane หลงทาง

**Handoff ที่ชัดเจนคือกุญแจ** — มี 3 จุด handoff หลัก: P2→P4 (pose API), P3→P4 (event/classification API), P5→P6 (event schema) ทุกจุดต้องมี contract เป็นเอกสาร schema ก่อนถึงเวลา handoff ไม่ใช่ "พร้อมแล้วค่อยส่ง"

**Buffer ปลายทาง 3-4 weeks** — สัปดาห์ 19-22 สำหรับ NSC report, demo video, และ submission ซึ่งเป็นงานอีกชุดหนึ่งโดยตัวมันเอง ห้ามใช้ buffer นี้แก้บั๊กระบบ

**Solo-dev pacing** — ถ้าทำคนเดียว แนะนำ time-box ตามวัน เช่น จันทร์-พุธ AI / พฤหัส-เสาร์ UI หรือ AM AI / PM UI กันสมองสับสน

---

# AI / Core Track

## P1 · Clinical Dataset & Rule Definition
**ช่วงเวลา:** week 1-3 (1-21 พ.ค. 2026, 3 weeks)

### เป้าหมาย
สร้าง "ความจริงทางคลินิก" ที่ระบบทั้งหมดจะ build บนนั้น — ตอบคำถาม "ท่าทางที่ถูกต้องคืออะไร เมื่อไหร่ผิด ผิดยังไง"

### สิ่งที่ต้องมีก่อนเริ่ม
- ที่ปรึกษาทางคลินิก (นักกายภาพบำบัด, แพทย์เวชศาสตร์การกีฬา, หรืออาจารย์พลศึกษา) ติดต่อได้
- รายชื่อ target conditions/exercises เบื้องต้น

### แผนรายสัปดาห์

**Week 1 (1-7 พ.ค.) — Scope freeze**
ตัดสินใจเลือกท่ากายภาพ 1-3 อย่างเท่านั้น โครงการ NSC แสดงผลคุณภาพดีบนขอบเขตเล็ก ดีกว่าผลตื้นๆ บนขอบเขตใหญ่ ตัวอย่างขอบเขตที่เหมาะสม: squat correctness, lower-back hyperextension detection, shoulder external rotation rehab ติดต่อที่ปรึกษาทางคลินิก ขอ session 30-60 นาที เพื่อเข้าใจ pain points จริงในงานกายภาพบำบัด

**Week 2 (8-14 พ.ค.) — Data collection**
รวบรวม reference data จาก public datasets:
- **KIMORE** — Kinect-based, 5 rehabilitation exercises, มี ground truth จาก PT จริง
- **UI-PRMD** — 10 exercises, 10 subjects, ทั้ง correct + incorrect performances
- **Fit3D / MM-Fit** — ถ้าเน้น fitness ทั่วไป

ถ้าต้องเก็บเอง: ถ่าย reference video กับนักกายภาพ/นักกีฬา 2-3 คน × 2-3 ท่า ทั้ง correct และ common mistakes (อย่างน้อย 5 mistakes ต่อท่า) จดบันทึก clinical commentary ขณะถ่าย — "ท่านี้ผิดเพราะหลังโก่งช่วงท้าย"

**Week 3 (15-21 พ.ค.) — Rule formalization**
เขียน Clinical Rules Document — สำหรับแต่ละท่ากำหนด:
- keypoints ที่เกี่ยวข้อง (เช่น hip, knee, ankle สำหรับ squat)
- มุมข้อต่อที่ถูกต้อง (ranges)
- common deviations + severity tier (mild/moderate/severe)
- rep counting heuristic (เช่น "นับเมื่อ knee angle ผ่าน 90° ลง แล้วกลับขึ้นมา 160°+")

กำหนด evaluation rubric สำหรับ P7 ล่วงหน้า เช่น "ระบบต้องตรวจจับ deep squat error ได้ ≥85% sensitivity, ≤15% false positive rate" ขอที่ปรึกษา sign-off

### Deliverables
- `rules.yaml` — formal rule spec ที่ machine-readable
- `clinical_commentary.md` — บันทึก rationale (ใช้ตอนเขียน NSC report ได้)
- `evaluation_rubric.md` — เกณฑ์ที่ P7 จะใช้
- (ถ้าเก็บเอง) `dataset/` — videos + annotations

### Tech / Tools
Public datasets KIMORE, UI-PRMD ฟรีสำหรับ academic ใช้ VIA Annotator หรือ CVAT (ฟรี, web-based) สำหรับ annotation เอกสารใช้ Markdown หรือ Notion

### ความเสี่ยง & การรับมือ
- **Scope creep** ("ลองเพิ่มท่านี้สิ") → ตั้ง hard limit 3 ท่าใน w1 และห้ามเปลี่ยนหลังจากนั้น
- **Advisor unavailable** → identify backup ตั้งแต่ w1 (อาจารย์พลศึกษา, รุ่นพี่ที่เรียน sports science)
- **Ambiguous rules** ("แล้วแต่ context") → ทุก rule ต้องเขียนเป็น deterministic predicate ได้ ถ้าเขียนไม่ได้ ให้แยก context เป็น sub-rules

### Definition of Done
ที่ปรึกษาคลินิก sign-off ใน Clinical Rules Document ทุก rule แปลงเป็น Python predicate function ได้ (เช่น `is_squat_too_shallow(keypoints) -> bool`) UI team ได้ rule list ไปเริ่ม design แล้ว

---

## P2 · Posture Pipeline Integration
**ช่วงเวลา:** week 3-6 (15 พ.ค. - 11 มิ.ย. 2026, 4 weeks · overlap 1w กับ P1)

### เป้าหมาย
สร้าง pipeline ที่รับ video stream → ปล่อย stable keypoints + derived metrics ที่ downstream consume ได้ ที่ ≥25 FPS บน target hardware

### สิ่งที่ต้องมีก่อนเริ่ม
- รู้แล้วว่าต้อง track keypoints ไหนจาก P1
- target hardware ตัดสินใจแล้ว (browser desktop? mobile web? native?)

### แผนรายสัปดาห์

**Week 3 (15-21 พ.ค.) — Model bake-off** (overlap กับ P1)
เปรียบเทียบ pose models บน reference clips:
- **MediaPipe Pose (BlazePose)** — 33 keypoints, ทำงานบน browser ผ่าน MediaPipe Tasks API, ฟรี, น่าจะเป็น default choice สำหรับ 2026
- **MoveNet Lightning/Thunder** — TensorFlow.js, เร็วมาก, accuracy รองจาก MediaPipe เล็กน้อย
- **YOLOv8-pose** — server-side, accurate มาก แต่ overkill สำหรับ browser
- **RTMPose** — state-of-the-art accuracy แต่ deploy ยากในเบราว์เซอร์

เกณฑ์เปรียบเทียบ: latency, accuracy บน reference clips, deployment ease, license เขียน decision doc สั้น ๆ + commit choice

**Week 4 (22-28 พ.ค.) — Pipeline scaffold**
สร้าง pipeline architecture:

```
Camera → Preprocessor (resize, normalize) → Pose Model
       → Postprocessor (smooth, derive angles) → Output Schema
```

Implement:
- joint angle calculator — function รับ 3 keypoints คืน angle (degrees)
- coordinate normalization (relative to hip center, scale-invariant ต่อระยะกล้อง)
- timestamp + frame ID tracking

**Week 5 (29 พ.ค. - 4 มิ.ย.) — Performance + smoothing**
- Latency optimization — target: pose inference + postprocess < 40ms/frame
- Smoothing filter — ใช้ **One Euro Filter** (mincutoff=1.0, beta=0.007 เป็น good defaults) ลด keypoint jitter
- Web Worker offload — ไม่ block main thread
- Profile บน target hardware จริง (laptop browser + mobile browser)

**Week 6 (5-11 มิ.ย.) — API freeze + stress test** (overlap กับ P3)
ล็อก output schema:

```typescript
interface PoseFrame {
  timestamp: number;
  frameId: number;
  keypoints: Array<{
    id: string;       // "left_knee", "right_hip", etc.
    x: number;        // normalized 0-1
    y: number;
    z: number;        // depth (ถ้ามี)
    confidence: number;  // 0-1
  }>;
  angles: Record<string, number>;  // computed joint angles, degrees
  confidenceOverall: number;
}
```

Stress test scenarios: low light, side angle, partial occlusion, body sizes ต่างกัน, ระยะกล้องต่าง publish API spec ให้ UI team (P4 เริ่ม integrate ใน w8)

### Deliverables
- `posture-pipeline/` package — ใช้เป็น library/service ได้
- `api-spec.md` — output schema + edge case behaviors
- `benchmark-report.md` — FPS, accuracy, hardware tested

### Tech / Tools
**MediaPipe Pose Tasks API** แนะนำเป็น default — มี Web/iOS/Android SDK, license อนุญาตเชิงพาณิชย์, mature TypeScript + Web Worker เพื่อ performance WebGL backend ถ้าใช้ TF.js / MoveNet

### ความเสี่ยง & การรับมือ
- **Low-light accuracy ต่ำ** → document limitation + ใส่ UX hint ("กรุณาเปิดไฟ")
- **Keypoint flickering** → One Euro Filter
- **Browser incompatibility** → test Chrome desktop + Safari + Chrome mobile เป็นอย่างน้อย
- **Mid-range mobile ช้า** → fallback profile (lower FPS, coarser model)

### Definition of Done
Pipeline รัน ≥25 FPS บน MacBook Air M1 + iPhone 13 (หรือ target hardware ใกล้เคียง) Output schema lock — UI team สร้าง mock generator ที่ตรง schema ได้แล้ว API spec อยู่ใน repo

---

## P3 · Movement Logic Development
**ช่วงเวลา:** week 6-10 (5 มิ.ย. - 9 ก.ค. 2026, 5 weeks · overlap 1w กับ P2)

### เป้าหมาย
แปลง pose stream + clinical rules → real-time correction signals ที่ stable, accurate, ใช้งานจริงได้

### สิ่งที่ต้องมีก่อนเริ่ม
- P2 keypoint stream (mock ก็ได้ในช่วงแรก)
- P1 clinical rules formalized

### แผนรายสัปดาห์

**Week 6 (5-11 มิ.ย.) — Architecture** (overlap กับ P2)
ออกแบบ rule engine:
- Finite state machine ต่อ exercise (states: idle / starting / down-phase / bottom / up-phase / completed)
- Plugin architecture — แต่ละ exercise เขียนเป็น module แยก (ขยายในอนาคตง่าย)
- Event emitter pattern สำหรับส่ง correction signals downstream

**Week 7 (12-18 มิ.ย.) — Per-exercise state machines**
- Implement state machine สำหรับท่าที่ 1 (เช่น squat) — phase detection, rep counting
- Stub สำหรับท่าที่ 2 และ 3
- Unit tests on synthetic pose sequences

**Week 8 (19-25 มิ.ย.) — Error detection logic**
- เขียน predicate functions ที่แปลง P1 rules → code (เช่น `is_knee_caving_inward()`)
- Severity classifier — mild/moderate/severe ตาม magnitude ของ deviation
- Build event types (CorrectionEvent, RepCompletedEvent, ExerciseStartedEvent)

**Week 9 (26 มิ.ย. - 2 ก.ค.) — Confidence + temporal**
- Confidence calibration — ไม่ fire correction ถ้า pose confidence < threshold
- Temporal smoothing — require sustained error N frames ก่อน flag (กัน micro-deviations)
- Hysteresis — ไม่ toggle on/off ตลอดเวลา

**Week 10 (3-9 ก.ค.) — Lock down + tests** (overlap กับ P5)
- Run on annotated reference dataset, compute accuracy
- Tune thresholds ให้ผ่าน evaluation rubric ของ P1
- API contract สำหรับ event stream → publish ให้ P5/P6
- Lock public API

### Deliverables
- `movement-logic/` library — pure logic, no I/O
- Test suite ที่ replay reference clips
- Event schema spec
- Performance report (accuracy on reference data)

### Tech / Tools
Pure TypeScript / Python (rule-based, ไม่ต้อง train ML) State machine ใช้ XState หรือ implement เอง testing ด้วย Vitest / pytest

### ความเสี่ยง & การรับมือ
- **Rules เข้มเกินไป → false positive เยอะ** → tune thresholds บน reference set, allow user-adjustable sensitivity
- **State machine bug ที่ exercise transitions** → snapshot test ของ state sequences
- **Edge cases ใน rep counting** (เริ่มไม่ครบรอบ, paused mid-rep) → explicit "incomplete" state, ไม่นับ
- **API ออกแบบไม่ดี ต้อง refactor ภายหลัง** → review API design กับตัวเอง 2 รอบก่อน lock

### Definition of Done
≥85% accuracy vs annotated reference clips Event API stable, P5 และ P6 พร้อม consume Test suite ผ่านทั้งหมด

---

## P5 · Multimodal Feedback Engine
**ช่วงเวลา:** week 10-12 (3-30 ก.ค. 2026, 3 weeks · overlap 1w กับ P3)

### เป้าหมาย
แปลง correction events → human-perceivable signals ที่ทันเวลา, ไม่รบกวน, ปรับได้

### สิ่งที่ต้องมีก่อนเริ่ม
- P3 event stream + schema
- UI overlay specs จาก UI track

### แผนรายสัปดาห์

**Week 10 (3-9 ก.ค.) — Channel design** (overlap กับ P3)
ออกแบบ feedback channels:
- **Visual** — overlay บน skeleton (highlight ข้อต่อที่ผิด), text prompt
- **Audio** — pre-recorded short cues หรือ TTS
- **Haptic** — Vibration API (mobile only)

กำหนด priority + cooldown rules — corrections ที่สำคัญสุดได้สิทธิ์ก่อน, มี cooldown 2-3 วินาทีต่อ correction เดียวกัน เพื่อไม่สแปม

**Week 11 (10-16 ก.ค.) — Audio + visual implementation**
- Audio cue library — pre-record 5-10 short prompts เป็นภาษาไทย ("ลึกกว่านี้", "ก้นถอย", "หลังตรง") ใช้คนพากย์จริง > TTS
- Visual overlay logic — emit display events กับ severity, UI render เอง
- TTS fallback ถ้าไม่มี recording (ใช้ browser native Speech Synthesis)

**Week 12 (17-23 ก.ค.) — Tuning + preferences**
- Tone/intensity scaling ตาม severity (เสียงดัง = severe, เสียงเบา = mild)
- User preference layer — เปิด/ปิดเฉพาะ channel, volume control, language
- Latency tuning end-to-end — target < 200ms จาก event → feedback fire
- Accessibility — caption mode สำหรับ audio cues

### Deliverables
- `feedback-engine/` module
- `audio-assets/` — recorded prompts (Thai, อาจมี English)
- Preference schema + UI integration spec

### Tech / Tools
Web Audio API (audio playback, volume control) Vibration API (mobile haptic) Browser Speech Synthesis API (TTS fallback) recording ใช้ Audacity (ฟรี) หรือ มือถือ

### ความเสี่ยง & การรับมือ
- **Audio annoying ถี่เกินไป** → cooldown + priority queue, default volume เบา
- **Lag event → feedback** → buffer events ใน RAF, ไม่ใช้ setTimeout
- **Browser audio policy** (ต้อง user gesture เริ่ม) → init audio context ตอน start session
- **Vibration API ไม่ support iOS Safari** → graceful fallback to visual + audio

### Definition of Done
Feedback fire < 200ms จาก event No spam (ตรวจ cooldown ทำงาน) User configure preferences ได้ Tested บน Chrome desktop + iOS Safari + Android Chrome

---

## P7 · Clinical Accuracy Validation
**ช่วงเวลา:** week 17-19 (21 ส.ค. - 10 ก.ย. 2026, 3 weeks)

### เป้าหมาย
พิสูจน์ว่าระบบผ่าน clinical accuracy bar ที่ตั้งไว้ใน P1 — มี evidence จริงสำหรับ NSC report

### สิ่งที่ต้องมีก่อนเริ่ม
- ระบบ integrated ครบ (AI + UI lanes รวมแล้ว)
- ที่ปรึกษาคลินิกพร้อมประเมิน
- รายชื่อ test users (5-10 คน)

### แผนรายสัปดาห์

**Week 17 (21-27 ส.ค.) — Protocol design**
- เขียน validation protocol — script ที่ tester ทำตาม (ทำท่าไหน ทำกี่ rep แต่ละแบบ)
- Test cases — สำหรับแต่ละท่า มี correct trials + intentional mistakes แต่ละชนิด
- Scoring sheet สำหรับ advisor (ground truth)
- Recruit testers — เพื่อนนักกีฬา, รุ่นน้อง, สมาชิกครอบครัว, ถ้าได้นักศึกษากายภาพยิ่งดี

**Week 18 (28 ส.ค. - 3 ก.ย.) — Run sessions**
- รัน validation sessions — บันทึก video + system events log
- ที่ปรึกษาคลินิก annotate manually (ground truth) จาก video
- เก็บ qualitative feedback จาก users (UX ใช้งานง่าย/ยาก)

**Week 19 (4-10 ก.ย.) — Analysis + report**
- คำนวณ metrics:
  - Sensitivity (recall) — ระบบจับ error ได้กี่ % ของ error จริง
  - Specificity — ระบบไม่ flag เมื่อท่าถูกได้กี่ %
  - False positive rate
  - Per-exercise breakdown
- เขียน validation report (เป็นส่วนหนึ่งของ NSC submission)
- ที่ปรึกษา sign-off ผลการทดสอบ

### Deliverables
- `validation-report.pdf` — formal report พร้อม metrics + advisor sign-off
- Raw data: video recordings + system logs + ground truth annotations
- Qualitative summary จาก user interviews

### Tech / Tools
- OBS Studio (ฟรี) — screen + camera recording
- Spreadsheet (Google Sheets / Excel) — annotation + scoring
- Python notebook — metric computation

### ความเสี่ยง & การรับมือ
- **Tester cancel last minute** → over-recruit (10 คน เพื่อให้ได้ 7-8)
- **Advisor schedule conflict** → จองเวลาตั้งแต่ ก.ค. เลย, มี backup window
- **Accuracy ต่ำกว่าเกณฑ์** → buffer ของแผนคือ w19 + buffer NSC ก่อน — มีเวลาแก้ urgent issues; ถ้าแย่จริงๆ ปรับ scope ของการ claim ใน NSC report (honest reporting > overclaim)
- **Recording corrupted** → 2 cameras / sources backup

### Definition of Done
Validation report เสร็จ มี metrics ครบ ที่ปรึกษาคลินิก sign-off พร้อมใส่ใน NSC submission

---

# UI / Frontend Track

## UI · Discovery + Design
**ช่วงเวลา:** week 2-5 (8 พ.ค. - 4 มิ.ย. 2026, 4 weeks)

### เป้าหมาย
ล็อก user flow, visual language, screen inventory ก่อน build code — ลด rework

### สิ่งที่ต้องมีก่อนเริ่ม
- P1 รายการท่ากายภาพเริ่มมีรูป (ตอน w2 ได้คร่าวๆ แล้ว)
- target persona ที่ชัด (ผู้ป่วย? นักกีฬา? PT?)

### แผนรายสัปดาห์

**Week 2 (8-14 พ.ค.) — Research**
- User interviews 3-5 คน — potential users, สอบถาม pain points ตอนทำกายภาพที่บ้าน
- Competitive analysis — Kaia Health, Sword Health, SmartPT, Physitrack สังเกต:
  - onboarding (ใช้เวลาเท่าไหร่)
  - real-time correction display
  - progress dashboard
  - audio/visual cue patterns

**Week 3 (15-21 พ.ค.) — Flows + low-fi**
- User flow diagrams — onboarding, exercise selection, session flow, post-session results, history
- Low-fi wireframes — Figma หรือ pen+paper สำหรับ ทุก main screen
- กำหนด screen inventory — รวมจะมีกี่ screen, แต่ละ screen ทำหน้าที่อะไร

**Week 4 (22-28 พ.ค.) — Design system + hi-fi**
- Design tokens — colors, typography, spacing, radii (ใช้ shadcn/ui defaults เป็น baseline)
- Component inventory — buttons, cards, sliders, dialogs, exercise card, session screen
- Hi-fi mockups สำหรับ key screens 5-7 หน้า: home, exercise list, session, post-session, history, settings

**Week 5 (29 พ.ค. - 4 มิ.ย.) — Validate + iterate** (overlap กับ Components)
- Clickable Figma prototype — เดิน flow จากต้นจนจบได้
- Quick usability test 3-5 คน — สังเกตจุดสะดุด
- Iterate based on feedback
- Spec hand-off ให้ตัวเอง (component specs, interaction details, edge states)

### Deliverables
- Figma file — design system + all hi-fi screens
- Clickable prototype URL
- `screens.md` — รายการ screens + functionality
- Component specs

### Tech / Tools
Figma (ฟรีสำหรับ student, สมัคร education plan) shadcn/ui เป็น baseline reference Tailwind variable palette สำหรับ token consistency

### ความเสี่ยง & การรับมือ
- **Over-designing** ("perfect first" trap) → time-box week 4 hi-fi 5 screens, ไม่เกิน
- **Conflicting feedback** จาก users → priority on actual users (ผู้ป่วย/นักกีฬา) > developer friends
- **Design ที่ AI lane implement ไม่ได้** → keep design realistic, ดู MediaPipe overlay capabilities ก่อน

### Definition of Done
Clickable Figma prototype ครบทุก main flow Component specs ครบสำหรับ build w5+ Self-walkthrough ราบลื่นไม่สะดุด

---

## UI · Components (with mock data)
**ช่วงเวลา:** week 5-8 (29 พ.ค. - 25 มิ.ย. 2026, 4 weeks · overlap 1w กับ Discovery)

### เป้าหมาย
Build production component library ทำงานบน mock data — ทุก screen navigate ได้, deploy preview ได้, ไม่ต้องรอ AI

### สิ่งที่ต้องมีก่อนเริ่ม
- Figma design system พร้อม
- API contract draft จาก AI lane (P2 schema, P3 event types)

### แผนรายสัปดาห์

**Week 5 (29 พ.ค. - 4 มิ.ย.) — Scaffold** (overlap กับ Discovery)
- Project init — Next.js 15 + Tailwind v4 + shadcn/ui (เหมือน CD Smart Campus stack)
- TypeScript strict mode
- Routing structure (app router)
- Layout + nav skeleton
- Auth shell — Supabase Auth (matches existing stack), pages: login, signup, profile
- Vercel deploy preview hookup ตั้งแต่วันแรก

**Week 6 (5-11 มิ.ย.) — Core session components**
- Exercise card component — title, thumbnail, difficulty
- Session screen layout — video panel + skeleton overlay placeholder + correction text panel + rep counter
- Skeleton overlay component — รับ keypoints array, render เป็น Canvas/SVG (mock data)
- Correction overlay component — รับ event objects, แสดง text + visual indicator

**Week 7 (12-18 มิ.ย.) — Dashboard + settings**
- Dashboard skeleton — chart placeholders, summary cards
- Exercise list page
- Settings page — preferences (audio on/off, language, etc.)
- Onboarding flow — first-time user (mock data)

**Week 8 (19-25 มิ.ย.) — Mock harness + polish** (overlap กับ P4)
- Mock pose stream generator — ปล่อย fake `PoseFrame` ที่ตรง schema ของ P2
- Mock event generator — fake `CorrectionEvent` schema P3
- Both replay จาก fixture data ที่ realistic
- Loading/error/empty states for all pages
- Storybook (optional) — visual regression for components

### Deliverables
- Repo + Vercel preview URL
- Component library ใช้งานได้ทุก screen
- Mock data harness ที่ตรง P2/P3 schema
- Component README

### Tech / Tools
- **Next.js 15** + App Router
- **Tailwind v4** + shadcn/ui
- **Supabase** — auth + (later) database (CD Smart Campus stack consistency)
- **TanStack Query** — data fetching state
- **Zustand** — client UI state
- **Recharts** — chart components (used in P6)

### ความเสี่ยง & การรับมือ
- **API contract drift** → ใช้ Zod schema เดียวกันทั้งสอง lanes, single source of truth
- **Component API change ตอนกลาง** → review API ทุก week ใน standup กับตัวเอง
- **Mock data ไม่ realistic** → ใช้ recorded data จาก P2 (ตอน w6+) แทน fake data ทันทีที่ได้

### Definition of Done
ทุก main screen navigate ได้บน mock data Vercel preview deploy ทุก commit Components ใช้ใน P4 ได้โดยไม่ต้อง refactor

---

## P4 · Real-time UI Integration
**ช่วงเวลา:** week 8-11 (19 มิ.ย. - 16 ก.ค. 2026, 4 weeks · overlap 1w กับ Components)

### เป้าหมาย
แทนที่ mock streams ด้วย AI streams จริง — end-to-end live session ทำงานได้ smooth บน target devices

### สิ่งที่ต้องมีก่อนเริ่ม
- P2 pose API ปล่อย w6 (อาจยังไม่ stable แต่พร้อมใช้)
- P3 event API ปล่อย w10
- Components พร้อม consume mock streams (จาก w8)

### แผนรายสัปดาห์

**Week 8 (19-25 มิ.ย.) — Camera wiring** (overlap กับ Components)
- WebRTC / `getUserMedia` integration
- Camera permission flow — ถามอนุญาต + handle denial
- Camera selection UI (ถ้ามีหลายตัว)
- Video preview component พร้อม mirror toggle

**Week 9 (26 มิ.ย. - 2 ก.ค.) — Pose stream → skeleton overlay**
- Connect P2 pose pipeline output → skeleton overlay
- Tune visualization — smooth keypoints, hide low-confidence joints, color-code based on confidence
- Performance audit — render บน 30 FPS, ไม่ block UI

**Week 10 (3-9 ก.ค.) — Event stream → corrections** (overlap กับ P5)
- Connect P3 event stream → correction overlay (visual indicator on skeleton + text prompt)
- Connect P5 feedback engine → audio playback
- Rep counter wiring
- Session summary screen at end

**Week 11 (10-16 ก.ค.) — Latency + edge cases** (overlap กับ P6)
- End-to-end latency tuning — target perceived < 100ms (event → user sees correction)
- Edge cases:
  - Camera lost mid-session → graceful handle
  - AI pipeline crashed → restart logic
  - Browser tab backgrounded → pause session
- Mobile-specific testing — orientation, screen size, touch

### Deliverables
- Live integrated session screen
- `latency-report.md` — measured perceived latency บน target devices
- Edge-case handling documented

### Tech / Tools
- WebRTC `getUserMedia` API
- Canvas 2D สำหรับ skeleton overlay (WebGL ถ้าต้องเยอะ)
- Web Worker สำหรับ pose pipeline (ไม่ block main thread)
- requestAnimationFrame สำหรับ render loop

### ความเสี่ยง & การรับมือ
- **Cross-browser webcam quirks** (Safari, Firefox) → test ทั้ง 3 หลัก, polyfills ถ้าจำเป็น
- **Mobile performance** → fallback to lower-resolution model
- **Latency feels laggy** → optimize critical path, ลด unnecessary re-renders, useMemo เยอะ
- **Pose model crashes บน some devices** → error boundary + fallback message

### Definition of Done
End-to-end live session ทำงานบน Chrome desktop, Safari desktop, Chrome mobile, Safari mobile Perceived latency < 100ms 1 full session ทำได้ตั้งแต่เริ่มจนจบโดยไม่มี error

---

## P6 · Analytics Dashboard
**ช่วงเวลา:** week 11-15 (10 ก.ค. - 13 ส.ค. 2026, 5 weeks · overlap 1w กับ P4)

### เป้าหมาย
ให้ user เห็น progress over time — sessions, accuracy trends, common errors, streak

### สิ่งที่ต้องมีก่อนเริ่ม
- P5 event schema lock (w12)
- Database schema design ready

### แผนรายสัปดาห์

**Week 11 (10-16 ก.ค.) — Database + ingestion** (overlap กับ P4)
- Schema design — Supabase tables: `users`, `sessions`, `events`, `exercises`
- ใช้ Row Level Security (RLS) สำหรับ user data isolation
- Ingestion API — POST endpoint รับ session data + events bulk
- Migration script

**Week 12 (17-23 ก.ค.) — Session summary** (overlap กับ P5)
- Post-session summary card — duration, reps completed, accuracy %, top corrections
- ทันทีหลัง session จบ
- Save session button → persist to Supabase

**Week 13 (24-30 ก.ค.) — Historical charts**
- Accuracy over time — line chart (Recharts)
- Sessions per week — bar chart
- Time spent per exercise — pie/donut chart
- Date range filter (7d / 30d / all time)

**Week 14 (31 ก.ค. - 6 ส.ค.) — Insights + streak**
- Most common errors per exercise — ranked list
- Personal best per exercise (best accuracy session)
- Streak counter (consecutive days)
- Weekly digest summary

**Week 15 (7-13 ส.ค.) — Export + polish** (overlap กับ Polish)
- PDF export — ส่งให้ PT ได้ (react-pdf หรือ Puppeteer server-side)
- Mobile responsive deep pass
- Performance — virtualize lists ถ้ายาว

### Deliverables
- Dashboard pages ครบทุก view
- Database schema + migrations
- PDF export functionality
- Demo data seeder (สำหรับ NSC demo video)

### Tech / Tools
- **Supabase** — Postgres + auth + RLS
- **Recharts** — React charting library, มาตรฐาน, ปรับ theme ได้
- **react-pdf** — PDF generation client-side
- **date-fns** — date manipulation

### ความเสี่ยง & การรับมือ
- **Slow queries** บน large data → index ที่ถูกต้อง (user_id, session_id, timestamp), aggregate ใน DB ไม่ใช่ client
- **Chart UX confusion** — ลด data density, เลือก default range ที่ make sense
- **PDF export ที่ render ช้า** → server-side generation
- **Demo data ไม่ realistic** → seed จาก validation sessions (ของจริง)

### Definition of Done
ทุก chart render < 1s บน 30 sessions ของ data PDF export ได้ Mobile responsive ผ่าน Demo data plausible

---

## UI · Polish + UAT
**ช่วงเวลา:** week 15-19 (7 ส.ค. - 10 ก.ย. 2026, 5 weeks · overlap 1w กับ P6 และ P7)

### เป้าหมาย
Production-grade UX, ready for clinical validation + NSC demo

### แผนรายสัปดาห์

**Week 15 (7-13 ส.ค.) — Bug bash** (overlap กับ P6)
- Internal testing — เดินทุก flow บน 3 browsers × 3 devices, จด bugs
- Triage P0/P1/P2 priority
- Fix P0/P1 bugs

**Week 16 (14-20 ส.ค.) — Accessibility + mobile**
- Accessibility audit — keyboard nav, screen reader (VoiceOver/TalkBack), color contrast (WCAG AA)
- Mobile responsive deep pass — ทุก screen ทุก breakpoint
- Touch target size ≥44px

**Week 17 (21-27 ส.ค.) — UX refinement** (overlap กับ P7)
- Onboarding refinement — first-time user experience smooth
- Error states comprehensive (network error, camera error, AI crashed)
- Empty states (no sessions yet)
- Loading states ที่ informative

**Week 18 (28 ส.ค. - 3 ก.ย.) — Validation support**
- Recording mode สำหรับ P7 — บันทึก session video + events sync
- Session export ในรูปแบบที่ advisor analyze ได้

**Week 19 (4-10 ก.ย.) — Final polish**
- Performance audit — Lighthouse, bundle size optimization
- Last-mile bug fixes
- Pre-submission walkthrough

### Deliverables
- Production-ready app deployed
- Accessibility compliance report
- Bug tracker zeroed (P0/P1)

### ความเสี่ยง & การรับมือ
- **Endless polish loop** → set hard cutoff, P2 bugs → defer
- **Last-minute regressions** → freeze code w19 mid-week, only critical fixes

### Definition of Done
P0/P1 bugs = 0 Lighthouse score ≥ 85 Demo flow smooth end-to-end

---

# NSC Submission Track

## NSC Submission Prep
**ช่วงเวลา:** week 19-22 (4-30 ก.ย. 2026, 4 weeks)

### เป้าหมาย
ส่งมอบทุก deliverables ที่ NSC requires ก่อน 30 ก.ย.

### แผนรายสัปดาห์

**Week 19 (4-10 ก.ย.) — Outline + setup**
- ดู NSC official template (ถ้ายังไม่ได้ดูตั้งแต่ต้น)
- Outline report sections:
  - Abstract / บทคัดย่อ (ไทย+อังกฤษ)
  - Introduction & motivation
  - Related work
  - System architecture
  - Methodology — แต่ละ phase
  - Results & validation (จาก P7)
  - Discussion & limitations
  - Future work
  - References
- Set up writing environment (LaTeX template ของ NSC ถ้ามี, หรือ Word)

**Week 20 (11-17 ก.ย.) — Writing + recording**
- เขียน technical sections — ใช้ deliverables จากทุก phase เป็น source material
- Record demo video first cut — 3-5 min, multiple takes
- Diagram cleanup — system architecture, data flow, validation results

**Week 21 (18-24 ก.ย.) — Edit + manual**
- Edit demo video — captions (Thai), B-roll, music ที่เหมาะสม
- เขียน user manual — installation, usage, troubleshooting
- Internal review ของ report กับที่ปรึกษา (ส่งไปอ่าน)

**Week 22 (25-30 ก.ย.) — Final review + submit**
- แก้ตามที่ปรึกษา
- Final proofread
- Submit ผ่าน NSC portal — เผื่อเวลา 2 วันสำหรับ technical issues
- Confirm receipt

### Deliverables
- Final report (PDF)
- Demo video (MP4)
- Source code repo (cleaned, README, license)
- User manual
- All NSC required forms

### ความเสี่ยง & การรับมือ
- **NSC portal มีปัญหา** → submit อย่างน้อย 2 วันก่อน deadline
- **ที่ปรึกษา feedback ช้า** → ส่งให้ดู w20 ไม่ใช่ w22
- **Demo video ต้องรีเทค** → record ตั้งแต่ w20, edit ใน w21 มีเวลา 1 week buffer

### Definition of Done
Submitted via NSC portal, confirmation email received

---

# Cross-track Concerns

## API Contracts (ต้องล็อกตอนไหน)

| Contract | ผู้ผลิต | ผู้บริโภค | ล็อกตอน | Schema location |
|----------|--------|----------|---------|-----------------|
| Pose frame | P2 | P3, P4 | end of w6 | `packages/schemas/pose.ts` |
| Correction event | P3 | P4, P5, P6 | end of w10 | `packages/schemas/event.ts` |
| Feedback signal | P5 | P4 (UI), P6 | end of w12 | `packages/schemas/feedback.ts` |
| Session record | P6 | DB, export | end of w11 | `packages/schemas/session.ts` |
| Clinical rule | P1 | P3 | end of w3 | `rules.yaml` |

**กฎทอง:** schema ต้องเป็น single source of truth — ใช้ Zod (TypeScript) หรือ Pydantic (Python) แล้ว generate types ทั้งสองฝั่ง อย่า maintain types แยก ระหว่าง backend/frontend

## คำแนะนำสำหรับ solo developer

**Daily structure:** AM intense work (AI logic, complex UI) / PM review + integration / late afternoon documentation + planning

**Weekly cadence:** Monday plan week / Wednesday self-review (เป็น devil's advocate กับงานตัวเอง) / Friday demo to advisor (สั้นๆ 10 นาที, weekly accountability)

**Context switching:** บล็อกวันละ 1 lane เป็นไปได้ — เช่น จันทร์-อังคาร AI / พุธ-พฤหัส UI / ศุกร์ integration นี่ลด overhead กว่าเปลี่ยนทุก 2 ชั่วโมง

**ใช้ Claude / Cursor effectively:** AI lane ใช้ Claude สำหรับ rule edge cases, math derivations / UI lane ใช้ shadcn-ui blocks + iterate ด้วย Cursor / debugging ใช้ Claude Code

**Energy management:** ห้ามทำงานกลางคืนเป็นนิสัย — productivity ดิ่งและ debugging ที่ผิดพลาดสร้าง bugs ให้ตัวเองพรุ่งนี้

## Risk Register (ภาพรวม)

| Risk | Phase | Likelihood | Impact | Mitigation |
|------|-------|-----------|--------|------------|
| Clinical advisor หาย | P1, P7 | Medium | High | Backup advisor identified w1 |
| Pose model accuracy ไม่ดี low-light | P2 | High | Medium | Document limitations, UX hints |
| Movement logic accuracy ไม่ถึง 85% | P3, P7 | Medium | High | Tune thresholds, ลด scope ถ้าจำเป็น |
| API contract drift | All | Medium | High | Single Zod schema, weekly check |
| Mobile browser quirks | P4 | High | Medium | Test 3 browsers ตั้งแต่ w8 |
| Validation testers cancel | P7 | High | High | Over-recruit (10 → 7 actual) |
| NSC submission portal ล่ม | NSC | Low | Critical | Submit 2 days early |
| Solo burnout | All | Medium | High | Friday demo ritual, no late nights |
| Scope creep | All | High | High | Hard limit 3 exercises ใน w1 |
| Demo video รีเทคหลายครั้ง | NSC | Medium | Medium | Start recording w20 |

---

## ขั้นตอนต่อจากนี้

**ทำทันทีสัปดาห์นี้ (28 เม.ย. - 4 พ.ค. 2026):**
1. ตัดสินใจ project scope — เลือก 1-3 ท่าให้แน่นอน
2. ติดต่อที่ปรึกษาคลินิก จองเวลา session แรก
3. ตั้งชื่อโครงการใหม่แทน "hurtfix" (placeholder)
4. Setup repo + project board (GitHub Projects หรือ Linear)
5. กำหนด target hardware (browser desktop primary? หรือ mobile-first?)

**สัปดาห์ถัดไป (5-11 พ.ค.):** เริ่ม P1 จริง
