# **Move-UP**  **NSC 2026 — Master Plan**

**Project:** Move-UP — gamified rehab/exercise companion   
**Submission:** Thailand National Software Contest 2026   
**Deadline:** 30 กันยายน 2026   
**Today:** 2 พฤษภาคม 2026 — เหลือ \~21 สัปดาห์  
**Author:** นักเรียน และ คุณครู เสรีคอม, Chitralada School   
**Architecture:** Mobile game client \+ AI server (Laptop, prototype phase)   
**Target user:** Rehab patients (primary) — gamification ช่วยแก้ปัญหา adherence **Pricing:** Free for NSC submission (no commercial tier yet)

---

## **1\. Executive Summary**

Move-UP คือ mobile game ในรูปแบบ Duolingo-inspired ที่ใช้ AI ตรวจสอบและให้คำแนะนำท่าออกกำลังกาย/กายภาพบำบัดแบบ real-time mobile app เป็น interface ที่มีรูปแบบเกม (XP, levels, streaks, daily quests, leaderboard) และ Laptop เป็น inference server (ระยะ prototype) ระบบจะตรวจจับ pose จากกล้องโทรศัพท์ ประเมินความถูกต้องของท่า ระบุข้อผิดพลาดเฉพาะเจาะจง (เช่น "เข่าเข้าด้านในตอน rep ที่ 5") และให้คะแนนเป็นแต้มสะสมเพื่อสร้างแรงจูงใจในการใช้งาน

**Target user:** ผู้ป่วยที่ได้รับ home exercise program จาก PT (primary) — gamification ออกแบบเพื่อแก้ปัญหา adherence ที่เป็น \#1 challenge ใน home rehab โดย gamification ทั้งหมดต้อง **ไม่บั่นทอนคุณภาพการออกกำลังกาย** (anti-gaming design — ดูข้อ 11\)

**จุดแข็งทางเทคนิค:**

* ใช้ SOTA models (RTMPose-l \+ MotionBERT \+ ST-GCN supervised contrastive)  
* LLM coach ในตัว (Qwen3-8B local via MLX) — feature ที่คู่แข่ง mobile-only ทำไม่ได้  
* Robustness ครบ 4 มิติ (lighting, angle, body type, occlusion)  
* Validation ตามมาตรฐานทางคลินิก (เป้าหมาย r ≥ 0.80 vs PT, เทียบ Kaia Health JMIR 2021\)

**จุดแข็งทางกลยุทธ์ (NSC):**

* **Gamified rehab/fitness app** — ออกแบบในรูปแบบเกม (Duolingo-inspired) เน้น engagement, streaks, daily quests, XP/levels เพื่อแก้ปัญหา adherence ที่เป็น \#1 challenge ใน home rehab (ผู้ป่วย drop-off ภายใน 4-6 สัปดาห์ ในงานวิจัยทางคลินิก)  
* **Category gap ในตลาด** — clinical apps (Kaia, Sword, Hinge) เน้น professionalism จริงจังแต่ engagement ต่ำ; fitness games (Ring Fit, Just Dance) สนุกแต่ไม่มี clinical accuracy; **Move-UP รวม 2 strengths**  
* **Tele-rehabilitation use case ที่จริงสำหรับ Thailand** — clinics มี PT จำกัด, gamified self-rehab เพิ่มการเข้าถึง  
* **Open และ inspectable** scoring methodology (vs Kaia/Hinge walled garden)  
* **Stratified fairness evaluation** ครบ Monk skin tone × BMI × lighting × angle (ไม่มีคู่แข่งทำ)  
* **Declarative exercise authoring** ให้ PT เพิ่มท่าได้โดยไม่ต้อง code

---

## **2\. Project Vision และ Scope**

### **2.1 Problem statement**

Home physical therapy มีปัญหา 2 ชั้น:

**ชั้นที่ 1 — Form quality**

* ผู้ป่วยทำท่าผิดไม่รู้ตัว → ไม่หาย หรือบาดเจ็บเพิ่ม  
* PT ในไทยมีจำกัด ไม่สามารถมาดูทุกครั้งได้  
* Apps ที่มีในตลาด (Kaia, Hinge, Sword) เป็น employer benefit, ไม่ได้ออกแบบสำหรับไทย, ไม่มีภาษาไทย  
* Apps in Thailand ที่มี (เช่น telehealth) ไม่มี real-time form correction

**ชั้นที่ 2 — Adherence (ปัญหาที่ใหญ่กว่า)**

* งานวิจัยทางคลินิก (Jack 2010, Bassett 2003\) ชี้ว่า **30-50% ของผู้ป่วยไม่ทำ home exercise program ตามที่ PT สั่ง**  
* Drop-off rate สูงสุดในช่วง 4-6 สัปดาห์แรก ก่อนเห็นผล  
* คู่แข่ง clinical apps แก้ปัญหา form quality แล้วแต่ยังไม่แก้ adherence อย่างจริงจัง  
* Move-UP ใช้ proven gamification patterns (Duolingo, Habitica) มาแก้ส่วนนี้

### **2.2 Target users (Phase 1\)**

* **Primary:** ผู้ป่วยที่ได้รับ home exercise program จาก PT (knee rehab, low-back pain) — ใช้ Move-UP ทุกวันเพื่อ rehab \+ เก็บแต้ม \+ รักษา streak  
* **Secondary:** PT ที่ต้องการ dashboard ดู patient adherence \+ progress remote  
* **Tertiary:** Family members ของผู้ป่วย (สามารถเป็น "supporter" ใน social feature)

**ไม่ใช่ target user ใน Phase 1:** casual fitness, gym athletes — focus rehab เพื่อ clinical positioning ที่ชัดสำหรับ NSC

### **2.3 Scope ที่ commit**

**3 ท่าใน Phase 1** — ตัดสินใจสัปดาห์ที่ 1, ห้ามเปลี่ยน:

1. Squat (correctness \+ depth \+ knee tracking) — knee rehab  
2. Sit-to-stand (rehab indicator, common exercise) — general mobility  
3. Single-leg stance / heel-toe walk (balance assessment) — fall prevention

**Game mechanics ใน Phase 1:**

* XP \+ Levels (core progression)  
* Streaks (daily continuity, สำคัญที่สุดสำหรับ adherence)  
* Daily quests (variable structured workout)  
* Achievements / badges (milestone celebration)  
* Leaderboard \+ Friend challenges (social motivation)

**ตัดออกจาก Phase 1:** yoga poses, deadlift, bench press, complex sport-specific movements, narrative/story mode (consider Phase 2\)

### **2.4 Success criteria**

**Technical:**

* End-to-end latency \< 200ms  
* Pose accuracy ≥ 85% vs PT ground truth

**Clinical:**

* r ≥ 0.80 vs PT scoring (matching Kaia 2021 benchmark)  
* Form quality measurable per session (no degradation despite gamification)

**Engagement (gamification working):**

* ≥ 60% Day-7 retention ในการทดสอบ user  
* Mean streak length ≥ 5 days ในกลุ่มทดสอบ  
* ≥ 80% ของ users กลับมา 3+ วันใน 7 วันแรก

**NSC submission:**

* Report \+ demo video \+ clean repo \+ clinical validation evidence \+ engagement data

---

##   **3\. Architecture Decision: Server-side Prototype**

### **3.1 Decision**

**Mobile \= camera \+ UI (thin client)**   
**Laptop \= AI inference \+ scoring \+ LLM (server)** 

Communication: WebRTC (video) \+ WebSocket (events) \+ REST (state)

ตอนแรกออกแบบ on-device pipeline เพราะกลัว latency ของ network แต่ตอน prototype การไป server มีข้อดีหลัก 5 ข้อ:

1. **Iteration speed** — แก้ algorithm \+ redeploy ใน server เร็วกว่า rebuild mobile app 10x  
2. **Model choice unblocked** — ใช้ ViTPose/RTMPose-l/Sapiens ได้ ไม่ต้องจำกัดที่ 5MB TFLite  
3. **LLM coach ทำได้** — Qwen3-8B ต้องการ \~16GB RAM ไม่มีทางรันบน iPhone  
4. **Multi-model ensemble** — run rule \+ ST-GCN \+ transformer พร้อมกันได้  
5. **Validation studies เร็วขึ้น** — เก็บ data \+ retrain \+ redeploy ภายในวันเดียว

---

## **4\. Technical Stack — Layer by Layer**

### **4.1 Mobile thin client**

`React Native 0.74+ (bare workflow)`  
`├── react-native-webrtc           (WebRTC video upstream)`  
`├── @shopify/react-native-skia    (skeleton overlay rendering)`  
`├── react-native-reanimated 3     (game animations, transitions)`  
`├── lottie-react-native           (celebration animations, loaders)`  
`├── react-native-sound or expo-av (SFX + background music)`  
`├── react-native-haptic-feedback  (haptic cues + game feedback)`  
`├── @tanstack/react-query         (REST state)`  
`├── zustand                       (UI + game state)`  
`└── nativewind                    (Tailwind for RN)`

**ไม่มี TFLite, ไม่มี Core ML, ไม่มี GPU delegate** — mobile แค่ส่ง video \+ render UI \+ game logic

### **4.2 Server stack (Laptop)**

`Python 3.12+`  
`├── FastAPI + uvicorn          (REST + WebSocket server)`  
`├── aiortc                     (WebRTC server-side, native Python)`  
`├── av (PyAV)                  (video frame decode)`  
`├── PyTorch 2.4+ with MPS      (model training + inference)`  
`├── mlx + mlx-lm               (LLM inference, Apple Silicon optimized)`  
`├── mmpose / mmdetection       (RTMPose pretrained models)`  
`├── opencv-python              (image preprocessing)`  
`├── numpy, scipy, pandas       (data processing)`  
`├── pydantic v2                (schema validation)`  
`├── redis-py                   (session state, event queue)`  
`└── asyncpg + sqlalchemy       (PostgreSQL access)`

`Storage:`  
`├── PostgreSQL 16              (sessions, users, exercises)`  
`└── Redis 7                    (session state, pub/sub)`

### **4.3 AI Pipeline (server-side, per-frame)**

**Total per-frame latency:** \~30-40ms inference \+ 20-40ms network \= 50-80ms perceived

### **4.4 Why each model choice**

ระบบ Move-UP ใช้ AI models 4 ตัวทำงานร่วมกันเป็นทอดๆ ลองนึกภาพว่ามันเป็น **ทีมงานในคลินิก** ที่แต่ละคนมีหน้าที่ของตัวเอง

1. #### RTMPose-l

**หน้าที่:** ดูภาพจากกล้องโทรศัพท์ แล้วบอกว่าผู้ใช้กำลังอยู่ในท่าทางไหน โดยระบุตำแหน่งของข้อต่อสำคัญ 26 จุดบนร่างกาย (หัว ไหล่ ข้อศอก ข้อมือ สะโพก เข่า ข้อเท้า รวมถึง**นิ้วเท้า**)

**ทำไมต้องมี:** ทุก model ที่เหลือใน pipeline ทำงานบน "ตำแหน่งข้อต่อ" ไม่ใช่ "ภาพดิบ" RTMPose จึงเป็นจุดเริ่มต้นที่แปลงภาพให้กลายเป็นข้อมูลตัวเลขที่ระบบเข้าใจได้ ถ้า RTMPose ผิด — ทุกอย่างที่ตามมาก็ผิดหมด

**ทำไมเลือก RTMPose ไม่ใช่ BlazePose:**

* **ความแม่นยำสูงกว่า** (75.8 mAP บน COCO vs BlazePose \~70) — สำคัญมากเพราะ rehab ผิดเล็กน้อยก็มีผล  
* **มี 26 keypoints รวม นิ้วเท้า** ซึ่ง BlazePose ไม่มี เท้า/นิ้วเท้าจำเป็นมากในท่า single-leg stance, heel-toe walk, sit-to-stand เพราะ balance ต้องดู center of pressure ของเท้า  
* **License Apache 2.0** ใช้เชิงพาณิชย์ได้ (BlazePose ก็เปิด แต่ YOLO-pose ที่ accuracy สูงกว่าเป็น AGPL ใช้ไม่ได้ใน app เชิงพาณิชย์)  
* **มี mobile variant พร้อม** RTMPose-s รัน 9ms บน Snapdragon ในอนาคตถ้าจะ port ลงโทรศัพท์ ใช้ architecture เดียวกันแค่ขนาดเล็กลง — model ที่เรา train logic ไว้ portable ได้

**Latency:** \~15ms บน Laptop — เร็วพอที่ 30 FPS ไม่ทันสะดุด

---

2. #### MotionBERT

**หน้าที่:** รับ keypoints 2D ที่ RTMPose ปล่อยมา (พิกัด x, y บนหน้าจอ) แล้วแปลงเป็น **3D ในพิกัดร่างกาย** (x, y, z จริงในอวกาศ)

**ทำไมต้องมี:** ปัญหาใหญ่ของ exercise scoring คือ **มุมกล้อง** สมมติผู้ใช้ทำ squat แต่กล้องวางเฉียงไป 30 องศา — ภาพ 2D จะดูเหมือน hip-shift (สะโพกเอียง) ทั้งที่จริงๆ ทำถูก ถ้าระบบใช้ 2D อย่างเดียว → false positive เยอะ ผู้ใช้หงุดหงิด

MotionBERT แก้ปัญหานี้ด้วยการ **ยก 2D เป็น 3D** แล้ว normalize ให้อยู่ในมุมมองมาตรฐานเดียวกันทุกครั้ง ไม่ว่ากล้องจะวางตรงไหน ระบบจะ "เห็น" ผู้ใช้จากมุมเดียวกันเสมอ

**ทำไมเลือก MotionBERT:**

* **Pretrain บน AMASS** — corpus ของ motion capture ที่ใหญ่ที่สุด (สิบล้านท่า) ทำให้รู้จัก human motion ในหลากหลายรูปแบบ  
* **Built-in temporal smoothing** — ใช้ context หลาย frame พร้อมกัน (transformer architecture) ทำให้ output ไม่กระตุก  
* **View-invariant by design** — paper ของ MotionBERT (ICCV 2023\) ออกแบบโดยตรงเพื่อแก้ปัญหามุมกล้อง

**Latency:** \~10ms ทุก 2 frames ก็พอ — ไม่ต้องรันทุก frame เพราะ 3D pose เปลี่ยนช้ากว่า 2D detection

---

3. #### ST-GCN with Supervised Contrastive

**หน้าที่:** ดูลำดับ pose ตลอด rep แล้วให้คะแนน "ทำท่าได้ดีแค่ไหน" (0-100) พร้อมระบุว่ามีจุดผิดอะไรบ้าง

**ทำไมต้องมี:** การดู pose ใน frame เดียวบอกไม่ได้ว่าท่าถูกหรือผิด เพราะ exercise เป็น **ลำดับการเคลื่อนไหว** ตัวอย่างเช่น squat ที่ถูกต้องไม่ได้แค่ "ลงต่ำพอ" แต่ต้อง:

* เริ่มจากยืนตรง  
* ลงด้วยจังหวะที่ควบคุมได้  
* เข่าไม่เข้าด้านในตลอด range  
* ลึกพอ  
* ขึ้นด้วย hip drive ไม่ใช่ knee dominant  
* กลับมายืนตรงเสถียร

ST-GCN (Spatio-Temporal Graph Convolutional Network) ดูร่างกายเป็น **กราฟของข้อต่อ** (เข่าเชื่อมกับสะโพกและข้อเท้า) แล้วเรียนรู้ pattern ตามเวลา ทำให้จับความสัมพันธ์ "เข่าเข้าด้านในตอน rep ลงต่ำสุด" ได้ ซึ่งกฎง่ายๆ ที่เราเขียนเองทำไม่ได้

**ทำไมต้อง Supervised Contrastive (Karlov 2024):**

* Loss function สำคัญกว่า architecture — paper ปี 2024-2026 ทุกอันที่ทำ SOTA บน KIMORE ใช้แนวคิดเดียวกัน คือ **สอน model ให้แยก "ท่าถูก" ออกจาก "ท่าผิด" ที่คล้ายกัน**  
* ตัวอย่าง: squat ที่ลึกแต่หลังโก่ง vs squat ที่ลึกถูกต้อง — ภาพรวมคล้ายกันมาก ต่างที่จุดเดียว Supervised contrastive บังคับ model ให้ดึง embeddings ของสองกรณีนี้ออกห่างกัน  
* **Pretrain บน NTU RGB+D 120** (database action recognition 120 ท่า) แล้ว fine-tune บน KIMORE/UI-PRMD ที่เป็น rehab-specific

**Output:** ไม่ใช่แค่คะแนน แต่ใช้ **GradCAM** ดูได้ด้วยว่า model "สนใจ" ข้อต่อไหน frame ไหน → ใช้บอกผู้ใช้ว่า "ผิดที่เข่าซ้ายตอนวินาทีที่ 2.3 ของ rep ที่ 5"

**Latency:** \~5ms ต่อ rep (ไม่ใช่ต่อ frame) เพราะรันตอน rep จบ

---

4. #### Qwen3-8B

**หน้าที่:** อ่านสรุปของ session (คะแนน, จุดผิดที่เกิดบ่อย, ประวัติของผู้ใช้) แล้วเขียน **คำแนะนำส่วนบุคคล** เป็นภาษาธรรมชาติ ทั้งภาษาไทยและอังกฤษ

**ทำไมต้องมี:** การให้คะแนนเป็นเลข (เช่น 82/100) หรือบอกว่า "knee\_valgus\_detected" เป็นภาษา machine ไม่ช่วยให้ผู้ป่วยเข้าใจว่า**ต้องทำอะไรต่อ** LLM แปลผลเชิงเทคนิคเป็นคำแนะนำที่ปฏิบัติได้จริง พิจารณาบริบท (ความเหนื่อย, ประวัติการบาดเจ็บ, progress ที่ผ่านมา)

**ทำไมเลือก Qwen3-8B (ไม่ใช่ GPT-4 หรือ Claude):**

* **รันบน Laptop ได้** — 16GB RAM footprint   
* **รองรับภาษาไทยดี** — Qwen3 train บน corpus ภาษาเอเชียเยอะ output ภาษาไทยธรรมชาติ  
* **Open weights, Apache 2.0** — modify prompts ได้อิสระ

**Latency:** 1-3 วินาที — ไม่ต้องเร็วเพราะรันตอน session จบ ผู้ใช้ดูสรุปอยู่แล้ว ไม่ต้องการ real-time

---

### **4.5 Game design layer (Move-UP unique)**

นี่คือ layer ที่ทำให้ Move-UP ต่างจากคู่แข่ง clinical apps ออกแบบตาม Duolingo \+ Habitica patterns โดยใช้ **Self-Determination Theory (SDT)** — autonomy, competence, relatedness — เป็น framework

**4.5.1 XP \+ Levels system**

XP per session \= base\_xp × form\_multiplier × streak\_multiplier  
  \- base\_xp \= 10 per rep (rep ที่ทำเสร็จ ไม่ว่าผิดถูก เพื่อ encourage continuation)  
  \- form\_multiplier \= 0.5 ถึง 1.5 (ตาม form score, เพื่อ reward quality ไม่ใช่แค่ quantity)  
  \- streak\_multiplier \= 1.0 \+ 0.1 × min(streak\_days, 7\) (max 1.7×)

Levels \= log curve, ไม่ใช่ linear  
  \- Level 1→2: 100 XP  
  \- Level 5→6: 500 XP  
  \- Level 20→21: 5000 XP  
  \- แบบนี้ early levels ให้ dopamine บ่อย, later levels ยังท้าทาย

**Anti-gaming guard:** form\_multiplier มี **floor at 0.3** เพื่อไม่ให้ user ทำเร็วๆ เก็บ rep โดยไม่ care form ถ้า form score ต่ำกว่า threshold → no XP at all \+ show "ลองช้าลงและตั้งสติกับท่า"

**4.5.2 Streak system (สำคัญที่สุดสำหรับ adherence)**

Daily streak \= consecutive days with ≥1 completed session  
  \- Session \= ≥5 reps with form\_score ≥ 60  
  \- Streak freeze: 1 free per week (เกรงผู้ป่วย flare-up หรือพักวันจริงๆ)  
  \- Streak loss \= soft (ลด ไม่ใช่ reset เป็น 0\)  
    \- Lose 1 day → streak ลด 50%  
    \- Lose 2 days → streak ลด 80%  
    \- Lose 3+ days → streak \= 0

**Why soft loss:** Hard reset to 0 (Duolingo style) ทำให้ผู้ป่วยที่หายเหนื่อย/flare-up รู้สึกท้อถึงขั้น quit Soft loss ให้ recovery path ได้ — match กับ rehab reality

**4.5.3 Daily quests**

แต่ละวันมี 3 quests ที่ generate ตาม PT prescription \+ user history:  
  1\. Required quest: PT-prescribed exercise (เช่น "Squat 3 sets × 10 reps")  
  2\. Variety quest: ท่าอื่นใน program (เพื่อไม่ให้น่าเบื่อ)  
  3\. Bonus quest: skill-based (เช่น "ทำ squat 5 ครั้งติดด้วย form ≥ 80%")

Completion → XP \+ streak \+ (sometimes) random reward

**4.5.4 Achievements / badges**

3 categories:  
  \- Milestone: "100 reps", "1000 reps", "Level 10"  
  \- Quality: "Perfect form 10×", "All correct in a session"  
  \- Consistency: "7-day streak", "30-day streak", "100-day streak"

Total \~30 achievements ใน Phase 1

**4.5.5 Leaderboard \+ friend challenges**

Weekly leaderboard:  
  \- League system (Duolingo-style): bronze → silver → gold → diamond  
  \- Top 10 ใน league promote, bottom 5 demote, อื่น stays  
  \- Leaderboard \= XP สัปดาห์นั้น (ไม่ใช่ all-time, ลด barrier สำหรับ new users)

Friend challenges:  
  \- 1-on-1 weekly challenge  
  \- "Who does more sit-to-stand reps with form ≥75% this week"  
  \- Loser ให้ encouragement message ไม่ใช่ shame

**Privacy-aware:** Leaderboard แสดงแค่ display name \+ level \+ weekly XP ไม่แสดง health data, exercise type, หรือ medical condition ใดๆ

**4.5.6 Anti-gaming design (ป้องกัน user game-ifies form)**

ปัญหา: gamification อาจทำให้ user "เก็บแต้ม" โดยทำ rep เร็วๆ ผิดๆ Move-UP design ป้องกันอย่างเป็นระบบ:

| Risk | Mitigation |
| ----- | ----- |
| User ทำเร็วเก็บ rep | Min rep duration enforced (1.5s ต่อ rep), เร็วกว่านั้นไม่นับ |
| User ทำผิดเพื่อสะสม | XP × form\_multiplier; floor 0.3 ป้องกัน garbage reps |
| User skip difficult exercises | Required quest ต้องทำ จะ unlock variety \+ bonus |
| User ทำท่าง่ายซ้ำ | Daily quest variety (rotation enforced) |
| User เน้น quantity ไม่ฟัง feedback | Quality achievements ให้ XP มากกว่า quantity ones |

**4.5.7 Game state schema**

\# Game state เก็บใน Postgres \+ cache ใน Redis  
class GameState:  
    user\_id: UUID  
    level: int  
    total\_xp: int  
    current\_streak: int  
    streak\_freeze\_remaining: int  \# weekly  
    last\_session\_date: date  
    league: Literal\["bronze", "silver", "gold", "diamond"\]  
    weekly\_xp: int  
    achievements: list\[AchievementId\]  
    daily\_quests: list\[Quest\]  \# regenerate each day

---

## 

## 

## 

## **5\. Research Findings สรุป**

ส่วนนี้คือสรุป research ที่ผ่านมา — สำหรับใส่ใน NSC report ใน section "Related Work" และ "Methodology"

### **5.1 Pose Estimation Landscape (2026)**

ทำการเปรียบเทียบ on-device และ server-side pose models:

| Model | Server latency Laptop | Mobile latency | mAP | Keypoints | License |
| ----- | ----- | ----- | ----- | ----- | ----- |
| **RTMPose-l** | **\~15ms** | (-s) \~9ms Snapdragon | **75.8** | 17/26 | Apache 2.0 |
| ViTPose-B | \~25ms | not mobile | 78.0 | 17 | Apache 2.0 |
| ViTPose-L | \~50ms | not mobile | 81.0 | 17 | Apache 2.0 |
| Sapiens-1B | \~80ms | not mobile | 82.3 | 308 | CC-BY-NC ❌ |
| BlazePose Heavy | \~5ms | \~53ms Pixel 3 | medium | 33 \+ 3D | Apache 2.0 |
| Apple Vision 2D | N/A | 5-15ms ANE | medium | 19 | Free, iOS only |
| MoveNet Lightning | \~3ms | \~7ms flagship | 58 | 17 | Apache 2.0 |
| YOLOv8-pose | \~10ms | 6-7 FPS | 50 | 17 | **AGPL ❌** |

**Decision:** RTMPose-l for prototype, mobile port → RTMPose-s

### 

### 

### 

### 

### **5.2 Scoring Approach Comparison**

Literature 2024-2026 converge ที่ **hybrid** (rule-based \+ ML):

| Approach | Best result on KIMORE | Pros | Cons |
| ----- | ----- | ----- | ----- |
| Pure rule-based | F1 \~0.85 (Pose Trainer) | Interpretable, no training | Doesn't generalize, hand-tuned |
| LSTM/BiLSTM | dev \<0.05 normalized | Easy on small data | Loses topology |
| 1D-CNN/TCN (LightPRA) | SOTA on KIMORE+UI-PRMD | Fast, parallel | Limited spatial modeling |
| **ST-GCN family** | **ρ \= 0.95-0.98** (Dual-Stream STGCN 2026\) | Captures inter-joint structure | Black-box (mitigated by GradCAM) |
| Transformer (MotionBERT) | SOTA via SSL-Rehab | Long-range temporal | Data-hungry |
| Hybrid | FormCoach 2025 benchmark | Interpretable \+ nuanced | Complex pipeline |

**Key insight:** Architecture matters less than training procedure

* Karlov 2024: vanilla ST-GCN \+ supervised contrastive loss \+ hard negatives \= \+12-21% on KIMORE  
* SSL-Rehab 2024: masked motion pretraining \+ LoRA fine-tune \= SOTA without architecture change

**Decision:** ST-GCN with supervised contrastive learning (Karlov 2024 recipe), pretrain NTU RGB+D, fine-tune KIMORE

### **5.3 Mistake Detection — 3 Complementary Techniques**

**(a) Joint-angle deviation rules** (interpretable floor)

* UI-PRMD provides formal definitions for knee valgus, pelvis drop, trunk angle  
* Smith 2024 Heliyon: validated against Vicon, RMS error \< 10° across squat, deadlift, drop jump, CMJ  
* Slade 2024: MediaPipe knee valgus mean error \~4° vs Vicon

**(b) Phase-based analysis** (errors happen at specific moments)

* FSM on key-joint trajectory (hip-Y for squat, wrist-Y for press)  
* Exponentially-weighted moving average extrema detection  
* US Patents 11,998,804 / 12,564,763 (Apple/Google)  
* REHAB24-6 dataset provides 1,072 reps with phase boundaries

**(c) Attention-based localization** (which joint, which frame)

* GradCAM on ST-GCN final layer \+ Overlap-Ratio metric (Wang TNSRE 2023\)  
* Attention-guided MQA: \+12% UI-PRMD, \+21% KIMORE (Kanade arXiv 2204.07840)  
* SHAP more locally consistent than GradCAM but more expensive (Dec 2024 paper)

**Mistake taxonomy** (convergent across literature):

* knee-cave (valgus)  
* back-rounding (lumbar flexion)  
* incomplete-depth  
* asymmetry / hip-shift  
* butt-wink (lumbar at bottom)  
* elbow-flare  
* hip-shoot (premature hip rise)  
* bar-path-deviation  
* lockout-failure

### **5.4 Robustness Techniques (4 dimensions)**

**Lighting:**

* Naive enhancement (Zero-DCE) actually hurts pose accuracy (Lee CVPR 2023\)  
* Task-aware methods: illumination-texture frequency decomposition (arXiv 2501.08038, 2025\)  
* **Cheapest intervention: UX prompt** — detect mean luma \< threshold from first 30 frames, ask user to adjust lighting

**Camera angle:**

* 3DPCNet (arXiv 2509.23455, 2025): rotation error 20°→3.4° via canonicalization  
* Cheap baseline: Kabsch-align hip line to y-axis \+ scale-normalize by torso length  
* Combined with MotionBERT 2D→3D lifting → camera angle mostly invisible

**Occlusion:**

* Sárándi 2018 synthetic-occlusion paste recipe (random Pascal VOC over training image, p≈0.5)  
* TAR-ViTPose (arXiv 2603.05929, 2026): \+2.3 mAP PoseTrack2017  
* Runtime: OneEuro filter (min\_cutoff=0.004, β=0.7); switch to Kalman per-joint when sustained low confidence

**Body type, clothing, skin tone:**

* BEDLAM (CVPR 2023): synthetic dataset, 271 body shapes × 100 skin textures, training on BEDLAM alone matches CLIFF  
* BEDLAM 2.0 (arXiv 2511.14394, 2025): adds high-BMI bodies \+ varied focal lengths  
* Sapiens-pretrained backbones: trained on 300M-1B human images, fairness priors built-in  
* Sony AI 2023 audit: no public pose dataset supports clean skin-tone audit  
* STW dataset (arXiv 2603.02475, 2026): 42K images on 10-tone Monk scale

**hurtfix opportunity:** stratified fairness eval (Monk × BMI × lighting × angle) — ไม่มีคู่แข่งทำ

### **5.5 Commercial Landscape**

**Two distinct categories — Move-UP sits in the gap between them:**

#### **Category A: Clinical-grade form correction (high accuracy, low engagement)**

**Players with peer-reviewed validation:**

* **Kaia Health** (JMIR 2021, PMC8317029): n=24, 552 sets, **r=0.828 (CV vs PT) ≈ r=0.833 (PT vs PT)** — non-inferior  
* **Sword Health**: 80+ patents, US 12,456,204 B1 specifically on vision-driven full-body tracking  
* **Hinge Health**: TrueMotion tracks 87 reference points, suggests COCO-WholeBody-class custom model

**Players without peer-reviewed validation:**

* Onyx, Zenia, BetterMe, Tonal, Tempo Move, Peloton IQ, Vay Sports, FormCoach

**Common limitation:** Engagement metrics ไม่ดี ไม่มี gamification ที่จริง — ผู้ป่วย drop-off เร็ว

#### **Category B: Gamified fitness/exercise (high engagement, no clinical accuracy)**

| Product | Strength | Why not a Move-UP competitor |
| ----- | ----- | ----- |
| **Ring Fit Adventure** (Switch) | RPG narrative \+ exercise | ต้องใช้ console \+ Joy-Con; ไม่มี form correction |
| **Just Dance** | Music \+ movement, fun | Movement matching ผ่าน controller, ไม่ใช่ form |
| **Pokemon GO** | Walking gamification | Step counting only, ไม่มี exercise form |
| **Pokemon Sleep** | Sleep gamification | Different domain |
| **Habitica** | Gamified habit tracker | Generic, ไม่มี domain expertise ใน rehab |
| **Zombies, Run\!** | Narrative running app | Audio-only, no form analysis |
| **Strong / Stronger by the Day** | Workout tracker | Manual logging, ไม่มี form check |
| **Apple Fitness+** | Trainer-led classes | One-way video, ไม่ตรวจ form ผู้ใช้ |

**Common limitation:** ไม่มี clinical validity ใช้กับ rehab patients ไม่ปลอดภัย

#### **The Move-UP gap**

**Key insight:** ไม่มีผู้เล่นไหนที่ทำทั้ง 2 ดี Move-UP target \= high clinical \+ high engagement

**Architectural patterns observed:**

* Smartphone-camera-only on-device pose dominates Category A (era of strap-on IMUs in retreat)  
* Custom in-house CNNs at high end (Kaia since 2018, Hinge TrueMotion)  
* MediaPipe BlazePose / MoveNet / Apple Vision dominate the long tail  
* Increasing layering with LLM/VLM agents (Sword Phoenix, Hinge Robin, FormCoach.io)  
* FormCoach 2025: pure VLM "still misses subtle form discrepancies and occasionally hallucinates" — hybrid pose+VLM is the frontier  
* **Game design patterns from Duolingo \+ Habitica are well-validated** but not applied to clinical rehab

### **5.6 Top Open Source References**

1. **NgoQuocBao1010/Exercise-Correction** — full pipeline reference: BlazePose → angles → classifier  
2. **yakupzengin/fitness-trainer-pose-estimation** — **YAML-defined exercise FSM** (pattern hurtfix should adopt)  
3. **MohEsmail143/coach-ai** — Flutter \+ TFLite cross-platform mobile rep counter  
4. **giaongo/RepDetect** — production-style native Android CameraX \+ MediaPipe  
5. **fokhruli/STGCN-rehab** — IEEE TNSRE paper code, ST-GCN on KIMORE+UI-PRMD  
6. **avakanski/A-Deep-Learning-Framework-for-Assessing-Physical-Rehabilitation-Exercises** — Liao 2020 reference implementation  
7. **MichistaLin/mediapipe-Fitness-counter** — Google's BlazePose-embedding \+ kNN \+ EMA  
8. **samkit-jain/physio-pose** — physiotherapy-specific (heel slide, knee bend)  
9. **bruceyo/EGCN** — ensemble GCN fusing position \+ angle features  
10. **stevenzchen/pose-trainer** — Stanford CS230 original (DTW vs reference template)

---

##  **6\. API Design**

### **6.1 Three protocols, three responsibilities**

| Protocol | Used for | Why |
| ----- | ----- | ----- |
| **REST** (FastAPI) | State changes, queries (auth, session create, summary) | Idempotent, cacheable, easy to retry |
| **WebSocket** | Bi-directional events \+ WebRTC signaling | Low overhead, server can push |
| **WebRTC** | Video upstream from camera | UDP, congestion-aware, NAT traversal built-in |

###    **6.2 REST API Surface**

`POST   /v1/auth/login              → { access_token, refresh_token }`  
`POST   /v1/auth/refresh            → { access_token }`

`GET    /v1/exercises               → [{ id, name, rules_version, ... }]`  
`GET    /v1/exercises/{id}          → { full exercise definition }`

`POST   /v1/sessions                → { session_id, ws_url, ice_servers }`  
                                     `body: { exercise_id, target_reps, ... }`  
`GET    /v1/sessions/{id}           → { status, started_at, ... }`  
`GET    /v1/sessions/{id}/summary   → { score, per_rep, llm_coaching, charts }`

`GET    /v1/users/me                → user profile`  
`GET    /v1/users/me/history        → { sessions: [...], stats }`  
`PATCH  /v1/users/me/preferences    → audio settings, language, etc.`

### **6.3 WebSocket Protocol**

**Endpoint:** `/v1/sessions/{session_id}/control?token={jwt}`

**Authentication:** JWT in query string (browser WS API limitation), verified before upgrade

**Message envelope** (uniform format):

`{`  
  `"type": "pose",`  
  `"ts": 1714312345678,`  
  `"session_id": "sess_abc123",`  
  `"data": { ... }`  
`}`

**Client → Server messages:**

| Type | Purpose | Data |
| ----- | ----- | ----- |
| `session.start` | Begin scoring | `{}` |
| `session.stop` | End \+ persist | `{}` |
| `signal.offer` | WebRTC SDP offer | `{ sdp, type }` |
| `signal.ice` | ICE candidate | `{ candidate, sdpMid, sdpMLineIndex }` |
| `feedback.preferences` | Update prefs mid-session | `{ audio_on, haptic_on, language }` |

**Server → Client messages:**

| Type | Frequency | Data |
| ----- | ----- | ----- |
| `session.ready` | Once after handshake | `{ protocol_version, server_capabilities}` |
| `signal.answer` | Once per offer | `{ sdp, type }` |
| `signal.ice` | As needed | `{ candidate, sdpMid, sdpMLineIndex }` |
| `pose` | \~30 Hz | `{ keypoints, angles, confidence }` |
| `rep` | On rep boundary | `{ rep_idx, score, mistakes, saliency }` |
| `error` | On failure | `{ code, message, retryable }` |
| `session.ended` | Final flush | `{ reason, summary_url }` |

###        **6.4 Senior architect decisions**

**Auth on WebSocket:** JWT in query string (`?token=...`), verified before upgrade. Reject with 401 before opening frame.

**Session lifecycle:** REST `POST /sessions` creates session, returns WS URL with `{id}` already bound. Horizontal scale: route by `session_id` (sticky to worker).

**Reconnect:** If WS drops, client reconnects with same `session_id` \+ last received `ts`. Server resumes from Redis state within 30s window. After 30s → `session.ended`.

**Backpressure:** If client lags processing pose updates, server drops to keyframes only (every \~100ms). But `rep` events MUST NOT be dropped (semantic events).

**Versioning:** `/v1/` in REST path \+ `protocol_version` in `session.ready`. Allows breaking changes through coordinated migration.

**Error envelope:** `{ code: "POSE_LOST", message: "...", retryable: true }`. Machine-readable code lets client decide retry/show error deterministically.

**Idempotency:** REST writes carry `Idempotency-Key` header (UUID v4). Server caches result for 24h.

---

##             **7\. Updated 21-Week Timeline**

แผนเดิม 22 สัปดาห์ปรับเป็น 21 สัปดาห์ เพราะ mobile lane ลดงานมาก, AI lane เพิ่มขึ้น

### **7.1 Master timeline overview**

| Week | Date range | AI/Server track | Mobile thin client track | Notes |
| ----- | ----- | ----- | ----- | ----- |
| 1 | 5-11 พ.ค. | P1 Dataset & Rules | (รอ) | Lock scope (3 ท่า) |
| 2-3 | 12-25 พ.ค. | P1 cont. \+ P2 setup | UI Discovery | Contact PT advisor |
| 4-7 | 26 พ.ค.-22 มิ.ย. | P2 Pose pipeline (RTMPose+MotionBERT) | UI Components (mock data) | API contract lock w6 |
| 6-11 | 9 มิ.ย.-13 ก.ค. | P3 Movement Logic \+ ST-GCN training | (overlap) | Karlov 2024 recipe |
| 8-12 | 23 มิ.ย.-20 ก.ค. | (overlap) | P4 Real-time WebRTC integration | E2E live by w11 |
| 11-13 | 14 ก.ค.-3 ส.ค. | P5 Feedback \+ LLM coach | (overlap) | Qwen3-8B integration |
| 13-17 | 28 ก.ค.-31 ส.ค. | (server stable) | P6 Analytics dashboard | History \+ charts |
| 14-19 | 4 ส.ค.-14 ก.ย. | (overlap) | UI Polish \+ UAT | Bug bash |
| 17-19 | 25 ส.ค.-14 ก.ย. | P7 Clinical Validation | (parallel) | n≥25, 2 PT raters |
| 19-21 | 8-30 ก.ย. | NSC Submission Prep (รวม 2 lanes) |  |  |

###    **7.2 Critical path**

The longest sequential chain ที่ห้าม slip:

1. P1 Rules (w3) → P2 Pose API (w7) → P3 ST-GCN trained (w11) → P7 Validation (w17-19) → NSC Submit (w21)

ถ้า phase ใดในนี้ slip 1 สัปดาห์ → buffer NSC submission หาย ดังนั้นแผนจริงต้องมี hard cutoffs

### **7.3 Hard cutoffs**

* **End of w3:** Clinical rules document signed off by advisor  
* **End of w7:** Pose API contract locked, returns stable schema  
* **End of w11:** ST-GCN model trained, accuracy ≥ 80% on KIMORE  
* **End of w13:** End-to-end live demo working  
* **End of w16:** Validation protocol approved, testers recruited  
* **End of w19:** Validation report draft complete, advisor reviewing  
* **End of w21:** Submitted to NSC portal

---

## **8\. Phase-by-Phase Detailed Plans**

### **P1 · Clinical Dataset & Rule Definition**

**Week 1-3 (5-25 พ.ค. 2026, 3 weeks)**

**Goal:** Define ground truth — "ท่าที่ถูกคืออะไร, ผิดเมื่อไหร่, ผิดยังไง"

**Week 1 (5-11 พ.ค.):**

* \[ \] **Lock scope**: เลือก 3 ท่า (squat, sit-to-stand, single-leg stance)  
* \[ \] Contact PT advisor — request 30-60min session  
* \[ \] Identify backup advisor  
* \[ \] Decide project name (replace "hurtfix")

**Week 2 (12-18 พ.ค.):**

* \[ \] Download datasets: KIMORE, UI-PRMD, REHAB24-6  
* \[ \] Review existing UI-PRMD rules for chosen exercises  
* \[ \] First PT advisor session — understand pain points  
* \[ \] (If recording own data) shoot reference videos: 2-3 subjects × 3 exercises × correct \+ 5 mistakes each

**Week 3 (19-25 พ.ค.):**

* \[ \] Write `rules.yaml` — formal rule spec, machine-readable  
* \[ \] Write `clinical_commentary.md` — rationale (for NSC report)  
* \[ \] Write `evaluation_rubric.md` — what P7 will measure  
* \[ \] PT advisor sign-off

**Deliverables:**

* `rules.yaml`  
* `clinical_commentary.md`  
* `evaluation_rubric.md`  
* (optional) `dataset/` with own recordings

**Definition of Done:**

* Advisor sign-off on rules document  
* Every rule expressible as Python predicate function  
* UI team has rule list to begin design

---

### **P2 · Pose Pipeline Integration**

**Week 4-7 (26 พ.ค.-22 มิ.ย. 2026, 4 weeks)**

**Goal:** Server pipeline that ingests video → emits stable keypoints \+ 3D \+ angles at ≥30 FPS

**Week 4 (26 พ.ค.-1 มิ.ย.):**

* \[ \] Set up FastAPI \+ uvicorn skeleton  
* \[ \] Set up aiortc WebRTC server  
* \[ \] Set up RTMPose-l with mmpose, verify inference works  
* \[ \] Decide: COCO-17 or Halpe-26 keypoints (recommend Halpe-26)

**Week 5 (2-8 มิ.ย.):**

* \[ \] Integrate MotionBERT 2D→3D lifter  
* \[ \] Implement OneEuro filter (Python port)  
* \[ \] Implement Kabsch alignment for canonical view  
* \[ \] Test pipeline on reference videos

**Week 6 (9-15 มิ.ย.):**

* \[ \] Wire pose pipeline to WebRTC frame stream  
* \[ \] Define `PoseFrame` schema (Pydantic)  
* \[ \] Implement event emission via Redis pub/sub  
* \[ \] Set up Postgres \+ initial schemas

**Week 7 (16-22 มิ.ย.):**

* \[ \] **API contract freeze**: publish `pose-api-spec.md`  
* \[ \] Stress test: low light, side angle, occlusion, body sizes  
* \[ \] Benchmark: target FPS, latency p50/p95/p99  
* \[ \] Document edge case behaviors

**Deliverables:**

* `posture-pipeline/` Python package  
* `pose-api-spec.md` — schema \+ behaviors  
* `benchmark-report.md`

**Definition of Done:**

* Pipeline ≥30 FPS on Laptop  
* API spec locked, mobile team can mock  
* Robust to lighting/angle/occlusion (qualitative pass)

---

### **P3 · Movement Logic Development**

**Week 6-11 (9 มิ.ย.-13 ก.ค. 2026, 6 weeks · overlaps P2)**

**Goal:** Convert pose stream → real-time correction signals via hybrid rule \+ ST-GCN

**Week 6-7 (9-22 มิ.ย., overlap with P2):**

* \[ \] Design FSM architecture per exercise  
* \[ \] Implement YAML-loadable state machine (study `yakupzengin/fitness-trainer-pose-estimation` pattern)  
* \[ \] Build phase detector for squat (hip-Y trajectory)

**Week 8 (23-29 มิ.ย.):**

* \[ \] Convert P1 rules → Python predicate functions  
* \[ \] Implement severity classifier (mild/moderate/severe)  
* \[ \] Build event types: `CorrectionEvent`, `RepCompletedEvent`, `ExerciseStartedEvent`

**Week 9 (30 มิ.ย.-6 ก.ค.):**

* \[ \] Set up ST-GCN training pipeline  
* \[ \] Pretrain on NTU RGB+D 120 (download \+ preprocess)  
* \[ \] Implement supervised contrastive loss (Karlov 2024 recipe)

**Week 10 (7-13 ก.ค.):**

* \[ \] Fine-tune ST-GCN on KIMORE \+ UI-PRMD  
* \[ \] Tune thresholds against P1 evaluation rubric  
* \[ \] Implement GradCAM for saliency

**Week 11 (14-20 ก.ค., overlap with P5):**

* \[ \] Confidence calibration \+ temporal smoothing  
* \[ \] Hysteresis to prevent toggle  
* \[ \] Lock event API → publish to P5/P6/UI

**Deliverables:**

* `movement-logic/` package (pure logic, no I/O)  
* ST-GCN trained checkpoint  
* Test suite replaying reference clips  
* Event schema spec

**Definition of Done:**

* ≥85% accuracy on annotated reference clips  
* ≥80% accuracy on KIMORE held-out set  
* Event API stable, downstream consumers ready  
* All tests pass

---

### **P5 · Multimodal Feedback \+ LLM Coach**

**Week 11-13 (14 ก.ค.-3 ส.ค. 2026, 3 weeks · overlap with P3)**

**Goal:** Convert correction events → human-perceivable signals \+ LLM-generated personalized feedback

**Week 11 (14-20 ก.ค., overlap with P3):**

* \[ \] Design feedback channels: visual / audio / haptic  
* \[ \] Define priority queue \+ cooldown rules  
* \[ \] Set up MLX environment for Qwen3-8B

**Week 12 (21-27 ก.ค.):**

* \[ \] Record audio cue library: 8-10 short prompts ภาษาไทย (พากย์จริง \> TTS)  
* \[ \] TTS fallback (browser native or server-side)  
* \[ \] Tone/intensity scaling by severity  
* \[ \] User preference layer

**Week 13 (28 ก.ค.-3 ส.ค.):**

* \[ \] LLM coach prompt engineering: pose features \+ mistake history → personalized text  
* \[ \] Test prompts in both Thai and English  
* \[ \] Latency: target \<3s for post-rep / \<10s for post-session summary  
* \[ \] Caching strategy for similar mistake patterns

**Deliverables:**

* `feedback-engine/` module  
* `audio-assets/` recorded prompts (Thai)  
* `llm-coach/` with prompt templates  
* Preference schema

**Definition of Done:**

* Feedback fires within latency budget (rep \<500ms, session \<10s)  
* No spam (cooldown verified)  
* LLM output quality good in Thai (advisor review)

---

### **Mobile Track: UI Discovery → Components → Real-time → Polish**

**Week 2-21 (parallel)**

#### **UI Discovery \+ Design (w2-5, 4 weeks)**

**Week 2-5 (12 พ.ค.-8 มิ.ย.):**

* \[ \] User interviews 3-5 potential users  
* \[ \] Competitive analysis (Kaia, Sword, Physitrack)  
* \[ \] User flow diagrams \+ low-fi wireframes  
* \[ \] Hi-fi mockups for 5-7 key screens  
* \[ \] Clickable Figma prototype \+ usability test

#### **UI Components with Mock Data (w5-8, 4 weeks)**

**Week 5-8 (2-29 มิ.ย., overlap):**

* \[ \] RN bare workflow \+ nativewind setup  
* \[ \] Auth shell (Supabase or custom JWT)  
* \[ \] Exercise card \+ session screen layout  
* \[ \] Skeleton overlay component (Skia)  
* \[ \] Mock pose stream generator (matches P2 schema)  
* \[ \] Vercel-equivalent preview deploy (Expo Updates or EAS)

#### **P4 Real-time WebRTC Integration (w8-11, 4 weeks)**

**Week 8-11 (23 มิ.ย.-20 ก.ค.):**

* \[ \] react-native-webrtc setup  
* \[ \] Camera permission flow  
* \[ \] WebRTC offer/answer signaling via WS  
* \[ \] Connect pose stream → skeleton overlay  
* \[ \] Connect event stream → correction overlay \+ audio  
* \[ \] Edge cases: camera lost, AI crash, backgrounded  
* \[ \] Latency tuning end-to-end

#### **P6 Analytics Dashboard (w13-17, 5 weeks)**

**Week 13-17 (28 ก.ค.-31 ส.ค.):**

* \[ \] Database schema design  
* \[ \] Session summary card (post-session)  
* \[ \] Historical charts (accuracy, sessions per week, time per exercise)  
* \[ \] Common errors per exercise  
* \[ \] Streak counter \+ weekly digest  
* \[ \] PDF export

#### **UI Polish \+ UAT (w15-19, 5 weeks)**

**Week 15-19 (11 ส.ค.-14 ก.ย.):**

* \[ \] Bug bash 3 browsers × 3 devices (well, mobile only for this version)  
* \[ \] Accessibility audit (TalkBack, VoiceOver)  
* \[ \] Mobile responsive deep pass  
* \[ \] Onboarding refinement  
* \[ \] Performance audit, bundle optimization

---

### **P7 · Clinical Accuracy Validation**

**Week 17-19 (25 ส.ค.-14 ก.ย. 2026, 3 weeks)**

**Goal:** Demonstrate clinical validity per Kaia 2021 standard

**Week 17 (25-31 ส.ค.):**

* \[ \] Write validation protocol (study script)  
* \[ \] Define test cases: correct trials \+ intentional mistakes per exercise  
* \[ \] Build scoring sheet for advisor  
* \[ \] Recruit 10 testers (over-recruit for 7 actual)

**Week 18 (1-7 ก.ย.):**

* \[ \] Run validation sessions (record video \+ system events)  
* \[ \] PT advisor annotates ground truth from video  
* \[ \] Collect qualitative UX feedback

**Week 19 (8-14 ก.ย.):**

* \[ \] Compute metrics: sensitivity, specificity, F1, ICC, kappa  
* \[ \] Per-exercise breakdown  
* \[ \] Stratified breakdown (gender, BMI, lighting)  
* \[ \] Write validation report  
* \[ \] Advisor sign-off

**Target metrics:**

* r ≥ 0.80 vs PT (matching Kaia 2021\)  
* ICC(2,1) ≥ 0.75  
* κ ≥ 0.61 substantial agreement  
* Joint-angle accuracy ≤ 5° clinical  
* F1 ≥ 0.85 binary mistake classification

**Deliverables:**

* `validation-report.pdf` with metrics \+ advisor sign-off  
* Raw data: videos \+ logs \+ ground truth annotations  
* Qualitative summary

---

### **NSC Submission Prep**

**Week 19-21 (8-30 ก.ย. 2026, 3 weeks)**

**Week 19 (8-14 ก.ย.):**

* \[ \] Review NSC official template  
* \[ \] Outline report sections  
* \[ \] Set up writing environment (LaTeX or Word)

**Week 20 (15-21 ก.ย.):**

* \[ \] Write technical sections (use deliverables from each phase as source)  
* \[ \] Record demo video first cut (3-5min)  
* \[ \] Diagram cleanup (system architecture, data flow, results)

**Week 21 (22-30 ก.ย.):**

* \[ \] Edit demo video \+ Thai captions  
* \[ \] User manual  
* \[ \] Internal review with advisor  
* \[ \] Final proofread \+ submit to NSC portal (2 days early\!)

**Deliverables:**

* Final report (PDF)  
* Demo video (MP4 with Thai captions)  
* Source code repo (cleaned, README, license)  
* User manual  
* All NSC required forms

---

## **9\. Datasets และ Training Strategy**

### **9.1 Dataset selection**

| Dataset | Use | Subjects | Exercises | License |
| ----- | ----- | ----- | ----- | ----- |
| **NTU RGB+D 120** | Pretrain ST-GCN | 106 | 120 actions | Academic, signed |
| **KIMORE** | Fine-tune (continuous scoring) | 78 | 5 rehab exercises | Free academic |
| **UI-PRMD** | Fine-tune (binary correct/incorrect) | 10 | 10 exercises | Public domain |
| **REHAB24-6** | Phase boundary annotation | 10 | 6 exercises | Academic |
| **Fitness-AQA** | Fine-grained gym mistakes | wild | 3 (squat, deadlift, bench) | Academic |
| **BEDLAM** | Fairness augmentation | 271 body shapes | — | Open |
| **InfiniteForm** | Synthetic fitness pose | 60K images | 15 categories | Open |

### **9.2 Training pipeline**

Stage 1: Pretrain ST-GCN backbone  
  Input: NTU RGB+D 120 skeleton sequences  
  Loss: Cross-entropy (action classification)  
  Output: Pretrained encoder  
  Compute: \~24h on M5 Max GPU

Stage 2: Self-supervised refinement (optional, \+3-5% accuracy)  
  Input: Unlabeled exercise videos (own \+ Fitness-AQA)  
  Loss: Masked motion modeling (SSL-Rehab style)  
  Output: Refined encoder

Stage 3: Fine-tune on rehab data  
  Input: KIMORE \+ UI-PRMD \+ REHAB24-6  
  Loss: Supervised contrastive (Karlov 2024\) \+ MSE for continuous scoring  
  Output: Fine-tuned scoring head  
  Compute: \~2-4h on M5 Max GPU  
  Hyperparameters: lr=1e-4, batch=32, epochs=50, hard negative ratio=0.3

Stage 4: Augmentation training  
  Apply: BEDLAM synthetic skeletons \+ InfiniteForm \+ Sárándi occlusion  
  Refresh: Stage 3 with augmented data  
  Output: Robust model

### **9.3 Topology mapping**

KIMORE/UI-PRMD use Kinect-v2 25-joint topology, our pipeline uses Halpe-26 (RTMPose). Need topology-mapping layer:

\# Pseudocode  
def kinect\_to\_halpe26(kinect\_skeleton):  
    \# 25 → 26 mapping  
    \# Kinect SpineMid → Halpe spine center  
    \# Kinect HandLeft \+ Wrist → Halpe wrist  
    \# ...  
    return halpe\_skeleton

def halpe26\_to\_kinect(halpe\_skeleton):  
    \# Inverse for evaluation  
    pass

Train and evaluate consistently in one topology.

---

## **10\. Clinical Validation Methodology**

### **10.1 Frame as non-inferiority study**

**Key insight:** PTs disagree substantially with each other:

* Wiles movement-screen: κ within-session 0.45, between-session 0.35  
* Scapular posture inter-rater κ: 0.04 to 0.76 across PT pairs

So the AI doesn't need to match a single ground truth — it needs to match PTs as well as PTs match each other (Kaia 2021 framing).

### **10.2 Study design**

* **Type:** Prospective cohort  
* **n ≥ 25 participants** (over-recruit to 30 to allow 25 useful)  
* **≥ 2 PT raters** (advisor \+ 1 backup)  
* **≥ 500 exercise repetitions** total  
* **Pre-registered** protocol (write before running)  
* **Reported per CONSORT-AI / STARD-AI**

### **10.3 Stratification**

Stratify metrics by:

* Gender  
* Age bands  
* BMI categories  
* Skin tone (Monk 10-tone if possible)  
* Lighting conditions (good / dim / backlit)  
* Camera angle (front / side / 3-quarter)

Demonstrate worst-stratum performance within 5% F1 of best — none of the consumer competitors do this.

### **10.4 Metrics**

| Metric | Threshold | Source |
| ----- | ----- | ----- |
| Cohen's κ (binary mistake classification) | ≥ 0.61 substantial | Landis & Koch 1977 |
| ICC(2,1) (continuous scoring) | ≥ 0.75 good | Koo & Li 2016 |
| Joint-angle RMS error vs Vicon | ≤ 5° clinical | Smith 2024 Heliyon |
| F1 (mistake detection) | ≥ 0.85 | Field standard |
| Spearman ρ vs PT | ≥ 0.80 | Kaia 2021 benchmark |

### **10.5 Reference benchmarks**

* **Kaia Health 2021:** r=0.828 CV-vs-PT ≈ r=0.833 PT-vs-PT (n=24, 552 sets)  
* **Garbin 2022:** squat AI-vs-PT inter-coder κ=0.96  
* **TuMeke REBA:** software ICC=1.0 vs expert 0.89 vs novice 0.51

---

## **11\. Risk Register**

| Risk | Phase | Likelihood | Impact | Mitigation |
| ----- | ----- | ----- | ----- | ----- |
| Clinical advisor unavailable | P1, P7 | Medium | High | Backup advisor identified w1 |
| Scope creep ("เพิ่มท่านี้สิ") | All | High | High | Hard limit 3 exercises locked w1 |
| RTMPose accuracy ไม่ดีบน real users | P2 | Medium | Medium | Fallback to ViTPose-B if needed |
| ST-GCN ไม่ converge บน KIMORE | P3 | Low | High | Multiple recipes available; fallback to TCN |
| WebRTC NAT traversal ไม่ stable | P4 | Medium | High | LAN demo for NSC; coturn for testing |
| Mobile WebRTC quirks (iOS Safari) | P4 | Medium | Medium | Native RN module \> web |
| LLM hallucinates incorrect feedback | P5 | High | Medium | Constrain via structured output; advisor review prompts |
| Validation testers cancel | P7 | High | High | Over-recruit 10 → 7 actual |
| Validation accuracy \< 80% | P7 | Medium | High | Retrain in w19 buffer; honest reporting |
| NSC portal ล่ม | NSC | Low | Critical | Submit 2 days early |
| Demo video รีเทคหลายครั้ง | NSC | Medium | Medium | Start recording w20 |
| Solo developer burnout | All | High | High | Friday demo ritual; no late nights |
| Network latency degrades demo | NSC | Medium | High | LAN-only demo at presentation |
| Hardware failure (M5 Max) | All | Low | Critical | Cloud backup of trained models \+ code in GitHub |
| **Gamification overrides form quality** | Game | High | High | Anti-gaming design (4.5.6); form\_multiplier floor 0.3; min rep duration |
| **Engagement metrics overshadow clinical** | P7, NSC | Medium | High | Report both engagement \+ clinical separately; clinical primary in NSC |
| **Hard streak loss → user quits** | Game | Medium | High | Soft streak loss \+ weekly freeze; recovery path |
| **Cultural mismatch (Duolingo tone in Thai rehab context)** | UI | Medium | Medium | UX testing with Thai rehab patients in w16; copywriting reviewed by PT |
| **Patient feels "babied" by game elements** | UI | Medium | Medium | Adult-appropriate visual style (not childish); option to mute celebrations |
| **Leaderboard reveals health info** | Privacy | Low | Critical | Display name \+ level \+ XP only; never exercise type or condition |
| **Game scope creep (more mechanics ≠ more engagement)** | Game | High | Medium | Lock 5 mechanics in w2; add more only if user testing demands |

---

## **12\. Action Items สัปดาห์นี้ (5-11 พ.ค. 2026\)**

### **Mandatory (must complete this week)**

1. **Lock project scope** — เลือก 3 ท่าให้แน่นอน:

   * Squat (correctness \+ depth \+ knee tracking)  
   * Sit-to-stand (rehab indicator)  
   * Single-leg stance / heel-toe walk (balance)  
   * **Decision required:** confirm or substitute  
2. **Lock game mechanics** — confirm Phase 1 set:

   * ✅ XP \+ Levels (Duolingo log curve)  
   * ✅ Streaks with soft loss \+ weekly freeze  
   * ✅ Daily quests (3 per day: required / variety / bonus)  
   * ✅ Achievements (\~30 in Phase 1\)  
   * ✅ Weekly leaderboard \+ friend challenges  
   * ❌ NOT in Phase 1: narrative mode, in-app currency, custom avatars  
   * **Decision:** confirm \+ write `game-spec.md` this week  
3. **Contact PT advisor** — secure week 1 session:

   * Identify primary advisor (PT, sports medicine doctor, or PE teacher)  
   * Identify backup advisor  
   * Request 30-60 min session this week  
   * **เพิ่มประเด็นถาม:** opinion on gamification — patient demographic ของเขา accept gamified UX ไหม? ระดับ "babying" ที่เหมาะ?  
4. **Set up infrastructure:**

   * GitHub repo with monorepo structure (`apps/mobile`, `apps/server`, `packages/schemas`)  
   * Project board (GitHub Projects or Linear)  
   * Notion / Obsidian for clinical notes  
5. **Hardware verification:**

   * Test M5 Max can run RTMPose-l at \>30 FPS  
   * Test MLX runs Qwen3-8B at acceptable speed  
   * Verify thermal: sustained inference 10 min OK

### **Nice to have (this week)**

6. Download datasets: KIMORE, UI-PRMD, REHAB24-6  
7. Read Karlov 2024 paper carefully (arXiv 2403.02772)  
8. Clone reference repos: `yakupzengin/fitness-trainer-pose-estimation`, `NgoQuocBao1010/Exercise-Correction`  
9. Set up Thai PDPA consent flow draft  
10. Create test data for early dev (record yourself doing the 3 exercises)

### **NOT this week (defer)**

* Don't start coding mobile app yet (wait w2)  
* Don't start training models yet (need rules first)  
* Don't optimize prematurely (correctness first)

---

## **13\. References**

### **Key research papers**

* **Karlov 2024** — Rehabilitation Exercise Quality Assessment through Supervised Contrastive Learning with Hard and Soft Negatives. arXiv:2403.02772  
* **MotionBERT (ICCV 2023\)** — A Unified Perspective on Learning Human Motion Representations. arXiv:2210.06551  
* **RTMPose** — Real-Time Multi-Person Pose Estimation based on MMPose. arXiv:2303.07399  
* **BlazePose** — On-device Real-time Body Pose tracking. (Google Research blog)  
* **AIFit (CVPR 2021\)** — Automatic 3D Human-Interpretable Feedback Models for Fitness Training  
* **FormCoach (2025)** — Lift Smarter, Not Harder. arXiv:2508.07501  
* **3DPCNet (2025)** — Pose Canonicalization for Robust Viewpoint-Invariant 3D Kinematic Analysis. arXiv:2509.23455  
* **BEDLAM (CVPR 2023\)** — A Synthetic Dataset of Bodies Exhibiting Detailed Lifelike Animated Motion  
* **Smith 2024** — Exercise quantification from single camera view markerless 3D pose estimation. Heliyon  
* **SSL-Rehab (CVIU 2024\)** — Assessment of physical rehabilitation exercises through self-supervised learning of 3D skeleton representations

### **Clinical validation references**

* **Huber et al. JMIR 2021** (Kaia Health) — Computer Vision vs Physical Therapists for exercise corrections. PMC8317029  
* **Wang TNSRE 2023** — Skeleton-Based Rehabilitation Exercise Assessment with Rotation Invariance \+ GradCAM. PMID 37276100  
* **Garbin 2022** — AI provides congruent and prescriptive feedback for squat form. PubMed 35920673  
* **Wiles et al.** — PT inter-rater reliability for movement scoring. PMC11441252

### **Datasets**

* KIMORE: https://vrai.dii.univpm.it/content/kimore-dataset  
* UI-PRMD: https://www.webpages.uidaho.edu/vakanski/Rehabilitation\_project.html  
* REHAB24-6: https://zenodo.org/records/13305826  
* BEDLAM: https://bedlam.is.tue.mpg.de/  
* NTU RGB+D: https://rose1.ntu.edu.sg/dataset/actionRecognition/

### **Open source**

* mmpose: https://github.com/open-mmlab/mmpose  
* aiortc: https://github.com/aiortc/aiortc  
* mlx: https://github.com/ml-explore/mlx  
* yakupzengin/fitness-trainer-pose-estimation  
* fokhruli/STGCN-rehab  
* avakanski/A-Deep-Learning-Framework-for-Assessing-Physical-Rehabilitation-Exercises  
* NgoQuocBao1010/Exercise-Correction

### **Standards**

* CONSORT-AI: https://www.consort-spirit.org/extensions/ai/  
* STARD-AI  
* WCAG 2.1 AA  
* Thai PDPA: https://www.pdpc.or.th/

---

**Document version:** 1.0 **Last updated:** 2 พฤษภาคม 2026   
**Next review:** End of Week 1 (11 พฤษภาคม 2026\)

---

