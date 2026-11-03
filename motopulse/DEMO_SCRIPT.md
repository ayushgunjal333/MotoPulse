# MotoPulse — Demo Script (5–10 minutes)

A step-by-step script for presenting MotoPulse to your project guide/panel. Practice it
once before the real thing — the whole flow takes about 6–8 minutes at a comfortable pace.

**Before you walk in**: have the server already running (`uvicorn main:app --reload` in
`backend/`) and the browser open at `http://localhost:8000` on the **Dashboard** page.

---

### 0. Open with the elevator pitch (30 seconds)

> "MotoPulse is an intelligent motorcycle safety system. It reads motion sensors from
> the rider's phone — GPS, accelerometer, gyroscope, tilt — runs them through a machine
> learning model to detect risky riding, calculates a live Safe Rider Score, detects
> possible crashes, and can trigger an emergency SOS with the rider's location. Since I
> don't have physical hardware here, everything you'll see is powered by a real-time
> sensor simulator — but every prediction and score you see is genuine, not scripted."

Point at the sidebar: *"Eight pages — Dashboard, Live Ride, Ride Analytics, Trip History,
Nearby Services, Ride Journal, Emergency/SOS, and an About page with the full system
architecture and model details."*

---

### 1. Dashboard (30 seconds)

Show the rider card, the Safe Rider Score gauge (currently 100 — "Excellent"), and the
current riding-behaviour status. Mention this is what the rider (or their family, via a
companion app) would see.

---

### 2. Start the ride — Live Ride page (1 minute)

Click **Live Ride** in the sidebar, then click **▶ Start Ride**.

> "This starts the ride simulator — it's now generating realistic sensor readings every
> 1.5 seconds: speed, accelerometer XYZ, gyroscope XYZ, and lean angle. Each reading goes
> through preprocessing, feature extraction, and the trained Random Forest model, which
> you can see updating live here —" *(point at the ML Prediction card, showing SAFE with
> a confidence percentage)*.

Let it run for ~10 seconds so the sensor panel and the live trend chart visibly update.

---

### 3. Trigger the risk scenarios (2 minutes)

Point at the **Demo Mode Controls** panel: *"These buttons don't just change a label on
screen — they inject real sensor patterns into the simulator, which the backend then has
to genuinely detect."*

1. Click **Simulate Overspeed** → speed jumps to ~90–105 km/h → point out the
   **OVERSPEEDING DETECTED** toast, the event appearing in the Risk Event Timeline with
   the exact sensor evidence, the Safe Rider Score dropping by 8, and the status pill
   turning to "RISK DETECTED".
2. Click **Simulate Harsh Braking** → point out the longitudinal acceleration spike in
   the sensor panel and the new event.
3. Click **Simulate Aggressive Turn** → point out the tilt angle and gyroscope values
   spiking together — mention this one requires *at least two* of three signals (lateral
   accel, gyro, tilt) to cross their thresholds, to avoid false positives from a single
   noisy sensor.
4. Click **Simulate Sudden Acceleration**.

> "Notice the Safe Rider Score is now around 70-something, and it's explainable — every
> single point deducted has a logged reason. That's important for the viva: nothing here
> is a black box."

---

### 4. Crash detection + SOS (1.5 minutes) — the highlight

Click **🚨 Simulate Crash**.

> "This simulates an impact spike followed by motion inactivity — the same hybrid
> detection approach described in the project proposal."

After ~3 seconds, the crash modal appears with a 10-second countdown.

> "The system has detected a possible crash and is counting down to an automatic SOS
> alert. The rider — or in this demo, me — has 10 seconds to cancel if it's a false
> alarm."

Demonstrate **both** paths if time allows:
- Click **CANCEL SOS** once to show the false-alarm path (ride resumes, event logged).
- Trigger crash again, then click **SEND SOS NOW** to show the confirmation screen:
  rider name, crash-detected time, GPS coordinates, a live OpenStreetMap link, and the
  emergency contact — clearly labeled as a demo simulation.

> "In a real deployment this would notify a predefined emergency contact with live
> location. For the prototype, it's fully simulated — no real messages are sent."

---

### 5. End the ride → Trip History → Ride Analytics (1.5 minutes)

Click **■ End Ride** → point out the Trip Summary card (distance, duration, avg/max
speed, final score, event count).

Click **Trip History** in the sidebar → *"Every ride is saved automatically."* Click
**Analytics** on the ride you just finished (or the row with the most events) to open:

> "Speed, acceleration, tilt and score over time, plus a Risk Factor Breakdown showing
> what fraction of the score loss came from each type of risky behaviour."

---

### 6. Quick tour of the remaining pages (1 minute)

- **Nearby Services** — switch categories (Fuel / Hospitals / Service / Food) to show
  the map + list.
- **Ride Journal** — show the pre-seeded "Nandi Hills" entry as an example of the digital
  ride journal feature from the proposal.
- **Emergency/SOS** — show the **TEST SOS** button as an always-available way to
  demonstrate the alert screen without needing a full crash sequence.

---

### 7. About / Architecture — close with the technical case (1.5 minutes)

This is the page to linger on if the panel wants technical depth:

- Point at the **8-step pipeline diagram** and narrate it once: sensors → collection →
  preprocessing → feature extraction → ML model → risk analysis → Safe Rider Score →
  dashboard/alerts.
- Point at the **Feature Importance chart** and **test accuracy** (~97%) — mention the
  model is a genuine `RandomForestClassifier` trained on a documented synthetic dataset
  (the disclaimer box explains why).
- Use **"Try the Model"** live: move the sliders (e.g. push longitudinal accel to −7) and
  click **Run Prediction** — show the model correctly flip to AGGRESSIVE with high
  confidence, in real time, via the actual `/api/predict` endpoint.

> "That's MotoPulse end-to-end — real sensor simulation, a real trained model, an
> explainable safety score, and a full trip-management and emergency-response layer, all
> running from a single command."

---

### Timing summary

| Section | Time |
|---|---|
| Pitch + Dashboard | 1 min |
| Start ride + Live sensors | 1 min |
| Scenario buttons | 2 min |
| Crash + SOS | 1.5 min |
| End ride + Analytics | 1.5 min |
| Other pages | 1 min |
| About / architecture + live predictor | 1.5 min |
| **Total** | **~9.5 min** (trim step 6 if you need to land at 5–7 min) |
