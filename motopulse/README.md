# MotoPulse
### Intelligent Mobile System for Motorcycle Safety, Behavioral Analytics and Trip Management

MotoPulse is a working prototype of an AI-based motorcycle safety and ride-management
platform. It simulates real motorcycle sensor data (GPS, accelerometer, gyroscope, tilt),
runs it through a trained machine learning model, computes an explainable **Safe Rider
Score**, detects risky riding behaviour and possible crashes in real time, and presents
everything on a live web dashboard.

Built for a final-year engineering major project demo. **Demo Mode** lets you run the
entire system without any physical hardware — a built-in simulator generates realistic
sensor data and lets you trigger specific riding scenarios on demand.

---

## 1. Features

- **Real-time ride simulation** — GPS, accelerometer, gyroscope and tilt data generated
  every 1.5s and streamed to the dashboard over a WebSocket.
- **Machine learning risk classification** — a Random Forest classifier (scikit-learn)
  trained on a synthetic riding-behaviour dataset, genuinely loaded and run at inference
  time on every sensor tick.
- **Hybrid detection** — ML classification is cross-checked against explainable threshold
  rules (overspeeding, harsh braking, sudden acceleration, aggressive turning), and crash
  detection uses an impact-spike + motion-inactivity rule, matching the project proposal.
- **Explainable Safe Rider Score (0–100)** — starts at 100, moves by documented amounts
  per risk factor, with a live breakdown of exactly why it changed.
- **Crash detection + SOS simulation** — impact → inactivity → 10s countdown → Cancel SOS
  / Send SOS Now, with a simulated emergency alert (rider, time, coordinates, contact).
  Clearly marked as a demo simulation — no real emergency services are contacted.
- **Trip Management** — trip history table, per-trip analytics (speed / acceleration /
  tilt / score over time, risk factor breakdown).
- **Digital Ride Journal** — save trip notes, start/destination, distance and score.
- **Nearby Services** — fuel stations, hospitals, service centers, food stops (demo data,
  structured so a real Places API can be swapped in later).
- **8-page dashboard**: Dashboard, Live Ride, Ride Analytics, Trip History, Nearby
  Services, Ride Journal, Emergency/SOS, About/Architecture.
- **Runs fully offline** (except live map tiles) — Chart.js and Leaflet are vendored
  locally, so the demo does not depend on a CDN being reachable during your viva.

---

## 2. Architecture

```
Sensor Layer (GPS, Accelerometer, Gyroscope, Tilt)
        │
        ▼
Data Collection  (ride simulator generates a tick every 1.5s)
        │
        ▼
Preprocessing    (rolling windows, unit handling)
        │
        ▼
Feature Extraction  (10 engineered features)
        │
        ▼
ML Model  (RandomForestClassifier — SAFE / AGGRESSIVE + confidence)
        │
        ▼
Risk Analysis  (rule thresholds cross-checked with the ML label)
        │
        ▼
Safe Rider Score  (explainable 0–100 score with per-factor breakdown)
        │
        ▼
Dashboard / Alerts  (WebSocket live UI, SQLite persistence, SOS pipeline)
```

**Stack**: Python 3 · FastAPI · Uvicorn · WebSockets · scikit-learn · pandas · numpy ·
joblib · SQLite · HTML5/CSS3/vanilla JavaScript · Chart.js · Leaflet.js + OpenStreetMap.

The frontend is a single static page (no build step) served directly by FastAPI, so
**one command starts the whole system** — backend, ML model and UI together.

---

## 3. ML Pipeline

**Features extracted from every sensor window** (`backend/config.py::FEATURE_NAMES`):

| Feature | Meaning |
|---|---|
| `speed_kmh` | current speed |
| `longitudinal_accel` | forward/backward acceleration (braking is negative) |
| `lateral_accel` | sideways acceleration (cornering force) |
| `gyro_magnitude` | combined rotation rate across all 3 axes |
| `tilt_angle` | lean angle from vertical |
| `braking_intensity` | `max(0, -longitudinal_accel)` |
| `sudden_accel_magnitude` | `max(0, longitudinal_accel)` |
| `turning_intensity` | `|lateral_accel| × |gyro_magnitude| / 10` |
| `speed_deviation` | deviation from the recent rolling-average speed |
| `accel_variance` | variance of longitudinal acceleration over a short window |

**Model**: `RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=3)`,
trained in `backend/ml/train_model.py` on a synthetic dataset (~4,200 samples across
safe riding, overspeeding, harsh braking, sudden acceleration and aggressive-turn
sub-populations, with injected label noise and feature jitter so classes overlap
realistically instead of being perfectly separable). Typical test accuracy: **~97%**.
Model + feature list + class names are saved together with `joblib` to
`backend/ml/model.joblib`, and metrics/feature importances are saved to
`backend/ml/feature_info.json`.

> **Data disclaimer** (shown on the About page too): for prototype demonstration,
> synthetic sensor data is used because real-world labeled motorcycle sensor data was
> not available. Predictions at runtime are genuine model inference — the backend loads
> `model.joblib` and calls `model.predict_proba()` on every sensor tick — not hard-coded
> responses.

**Hybrid risk layer**: alongside the ML label, `backend/config.py` defines explicit
thresholds (overspeed > 80 km/h, harsh braking ≤ −4.5 m/s², sudden acceleration ≥ 3.5
m/s², aggressive turn when ≥2 of {lateral accel, gyro, tilt} cross their thresholds).
This mirrors how the project brief describes crash detection ("hybrid approach based on
sudden impact and motion inactivity") and extends the same idea to everyday risk
detection — rules give instantly explainable, guaranteed detections for the demo buttons,
while the ML model provides a probabilistic, generalizing second opinion.

**Safe Rider Score**: starts at 100 per trip; each detected event subtracts a fixed,
documented amount (Overspeeding −8, Harsh braking −6, Aggressive turning −4, Sudden
acceleration −3, Possible crash −25); the score slowly regenerates while riding safely.
Every change is logged with its cause, so the dashboard can always show *why* the score
is what it is (see `backend/scoring.py`).

---

## 4. Project Structure

```
motopulse/
├── backend/
│   ├── main.py                 # FastAPI app: REST + WebSocket + serves the frontend
│   ├── config.py                # single source of truth for thresholds/scoring/features
│   ├── database.py               # SQLite schema, connection helper, demo seed data
│   ├── scoring.py                # explainable Safe Rider Score engine
│   ├── services_data.py          # Nearby Services demo data
│   ├── models.py                  # Pydantic request/response models
│   ├── ml/
│   │   ├── train_model.py          # synthetic dataset + Random Forest training
│   │   ├── predictor.py             # loads model.joblib, runs inference
│   │   ├── model.joblib              # trained model (generated by train_model.py)
│   │   └── feature_info.json          # accuracy, feature importances, disclaimer
│   ├── simulator/
│   │   ├── ride_simulator.py       # real-time tick generator + crash state machine
│   │   └── route.py                 # precomputed demo GPS route (Bangalore)
│   ├── motopulse.db                # SQLite database (generated on first run)
│   └── requirements.txt
├── frontend/
│   ├── index.html                  # entire UI: HTML + CSS + JS, no build step
│   └── vendor/                      # Chart.js + Leaflet, vendored locally (offline-safe)
├── data/
│   └── sensor_data_sample.csv       # sample of the synthetic training data
├── README.md
├── DEMO_SCRIPT.md
└── VIVA_PREP.md
```

---

## 5. Installation & Running

**Requirements**: Python 3.10+ (no Node.js needed — the frontend has no build step).

```bash
# 1. Go into the backend folder
cd motopulse/backend

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the ML model (creates ml/model.joblib + ml/feature_info.json)
python train_model.py
#    (this also runs automatically on first server startup if the model is missing)

# 4. Start the server — this serves BOTH the API and the dashboard
uvicorn main:app --reload
```

Then open **http://localhost:8000** in your browser. That's it — one server, one URL.

The SQLite database (`motopulse.db`) and a few sample past trips are created
automatically the first time the server starts. To wipe and reseed at any time, use the
**Reset Demo Data** button on the About page (or `POST /api/settings/reset-demo-data`).

### Running the demo without internet
Everything works offline except the OpenStreetMap map tiles (the marker, route and every
other feature still work — the map shows a "map tiles need an internet connection"
notice and continues normally). Chart.js and Leaflet's JS/CSS are vendored locally in
`frontend/vendor/`, so they do **not** require internet access.

---

## 6. API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/rider` | Rider profile |
| GET | `/api/current-trip` | Current simulator state |
| POST | `/api/ride/start` | Start a new simulated ride |
| POST | `/api/ride/pause` / `/api/ride/resume` | Pause / resume |
| POST | `/api/ride/end` | End ride, save trip summary |
| POST | `/api/simulate-event` | Inject a scenario: `overspeed`, `harsh_braking`, `sudden_acceleration`, `aggressive_turn`, `crash` |
| POST | `/api/predict` | Run the ML model directly on a supplied sensor snapshot |
| GET | `/api/safe-score` | Current Safe Rider Score + breakdown |
| GET | `/api/alerts` | List SOS alerts |
| POST | `/api/sos/test` | Immediately trigger a simulated SOS |
| POST | `/api/sos/cancel` | Cancel an active SOS countdown |
| GET | `/api/trip-history` | List completed trips |
| GET | `/api/trip-history/{id}` | Trip detail: readings + events |
| GET | `/api/nearby-services?category=` | Nearby fuel/hospital/service/food demo data |
| POST/GET | `/api/journal` | Create / list ride journal entries |
| GET | `/api/system-info` | ML metrics, thresholds, tech stack (used by the About page) |
| POST | `/api/settings/reset-demo-data` | Wipe and reseed the database |
| WS | `/ws/live` | Live sensor ticks, events, score updates, crash/SOS messages |

Full interactive docs are also available at **http://localhost:8000/docs** (FastAPI's
built-in Swagger UI) — useful if a faculty member wants to see `/api/predict` called
directly with custom values.

---

## 7. Database

SQLite (`backend/motopulse.db`), kept intentionally simple:

- `riders` — rider profile + emergency contact
- `trips` — one row per ride (distance, duration, avg/max speed, score, classification)
- `sensor_readings` — every sensor tick of every trip
- `ride_events` — every detected risk event, with the score delta and evidence text
- `alerts` — SOS alerts (simulated)
- `ride_journal` — journal entries

---

## 8. Future Improvements

- Replace the synthetic training dataset with real labeled motorcycle telemetry.
- Connect a real Places API (Google Places / OSRM) for Nearby Services and routing.
- Move from simulated sensors to a companion mobile app reading real phone sensors.
- Add user accounts / multi-rider support (currently single-rider by design, matching
  the "personal safety companion" framing of the project).
- Real push-notification / SMS integration for the SOS pipeline (currently simulated
  end-to-end, by design, for safety and to avoid contacting real services in a demo).
- Model retraining pipeline with real accident/near-miss data once collected.

---

*For the presentation flow, see **DEMO_SCRIPT.md**. For likely faculty questions and
answers, see **VIVA_PREP.md**.*
