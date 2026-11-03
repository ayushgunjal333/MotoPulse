"""
MotoPulse backend -- FastAPI application.

Run with:  uvicorn main:app --reload
Serves the REST + WebSocket API AND the frontend (single command demo).
"""
import json
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

import config
import database
import services_data
from ml import predictor, train_model
from models import SimulateEventRequest, PredictRequest, JournalCreate
from simulator.ride_simulator import RideSimulator
from simulator import route as route_mod

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
ML_DIR = BASE_DIR / "ml"

app = FastAPI(title="MotoPulse API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# WebSocket connection manager
# --------------------------------------------------------------------------- #
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(json.dumps(message, default=str))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()
simulator = RideSimulator(broadcast_fn=manager.broadcast)


# --------------------------------------------------------------------------- #
# startup
# --------------------------------------------------------------------------- #
@app.on_event("startup")
def on_startup():
    database.init_db(seed=True)
    if not predictor.is_ready():
        print("[MotoPulse] No trained model found -- training now (first run)...")
        train_model.train()


def get_rider_row():
    with database.get_db() as conn:
        row = conn.execute("SELECT * FROM riders LIMIT 1").fetchone()
        return row


# --------------------------------------------------------------------------- #
# rider / trip
# --------------------------------------------------------------------------- #
@app.get("/api/rider")
def get_rider():
    row = get_rider_row()
    if not row:
        return JSONResponse({"error": "No rider found"}, status_code=404)
    return dict(row)


@app.get("/api/current-trip")
def current_trip():
    return simulator.state_snapshot()


@app.post("/api/ride/start")
async def ride_start():
    rider = get_rider_row()
    if not rider:
        return JSONResponse({"error": "No rider configured"}, status_code=400)
    state = await simulator.start(rider)
    return state


@app.post("/api/ride/pause")
def ride_pause():
    return simulator.pause()


@app.post("/api/ride/resume")
def ride_resume():
    return simulator.resume()


@app.post("/api/ride/end")
async def ride_end():
    summary = await simulator.end()
    return {"summary": summary}


# --------------------------------------------------------------------------- #
# simulation controls
# --------------------------------------------------------------------------- #
@app.post("/api/simulate-event")
def simulate_event(req: SimulateEventRequest):
    result = simulator.inject_scenario(req.event_type)
    return result


# --------------------------------------------------------------------------- #
# ML prediction (also directly testable, e.g. via curl, for the viva)
# --------------------------------------------------------------------------- #
@app.post("/api/predict")
def predict(req: PredictRequest):
    import math
    gyro_mag = math.sqrt(req.gyro_x**2 + req.gyro_y**2 + req.gyro_z**2)
    tilt_angle = math.sqrt(req.tilt_roll**2 + req.tilt_pitch**2)
    avg_speed = req.avg_recent_speed if req.avg_recent_speed is not None else req.speed_kmh
    features = {
        "speed_kmh": req.speed_kmh,
        "longitudinal_accel": req.accel_y,
        "lateral_accel": req.accel_x,
        "gyro_magnitude": gyro_mag,
        "tilt_angle": tilt_angle,
        "braking_intensity": max(0.0, -req.accel_y),
        "sudden_accel_magnitude": max(0.0, req.accel_y),
        "turning_intensity": abs(req.accel_x) * abs(gyro_mag) / 10.0,
        "speed_deviation": abs(req.speed_kmh - avg_speed),
        "accel_variance": 0.0,
    }
    try:
        ml_result = predictor.predict_from_features(features)
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=503)

    risk_type = None
    if req.speed_kmh > config.OVERSPEED_KMH:
        risk_type = "OVERSPEEDING"
    elif req.accel_y <= config.HARSH_BRAKING_ACCEL:
        risk_type = "HARSH_BRAKING"
    elif req.accel_y >= config.SUDDEN_ACCEL_THRESHOLD:
        risk_type = "SUDDEN_ACCELERATION"
    elif sum([abs(req.accel_x) >= config.AGGRESSIVE_TURN_LATERAL,
              gyro_mag >= config.AGGRESSIVE_TURN_GYRO,
              tilt_angle >= config.AGGRESSIVE_TURN_TILT]) >= 2:
        risk_type = "AGGRESSIVE_TURN"

    return {
        "riding_status": "RISKY" if risk_type else "SAFE",
        "risk_type": risk_type,
        "safe_rider_score": round(simulator.score_engine.score, 1),
        "confidence": ml_result["confidence"],
        "ml_label": ml_result["label"],
        "probabilities": ml_result["probabilities"],
        "features": features,
        "timestamp": time.time(),
        "location": {"lat": route_mod.START_LAT, "lon": route_mod.START_LON},
    }


# --------------------------------------------------------------------------- #
# safety score / alerts
# --------------------------------------------------------------------------- #
@app.get("/api/safe-score")
def safe_score():
    return simulator.score_engine.breakdown()


@app.get("/api/alerts")
def get_alerts(trip_id: int | None = None):
    with database.get_db() as conn:
        if trip_id:
            rows = conn.execute("SELECT * FROM alerts WHERE trip_id=? ORDER BY ts DESC", (trip_id,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM alerts ORDER BY ts DESC LIMIT 20").fetchall()
        return [dict(r) for r in rows]


@app.post("/api/sos/test")
async def sos_test():
    alert = await simulator.trigger_sos(manual=True)
    return alert


@app.post("/api/sos/cancel")
async def sos_cancel():
    return await simulator.cancel_sos()


# --------------------------------------------------------------------------- #
# trip history
# --------------------------------------------------------------------------- #
@app.get("/api/trip-history")
def trip_history():
    with database.get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM trips WHERE status='completed' ORDER BY start_time DESC"
        ).fetchall()
        return [dict(r) for r in rows]


@app.get("/api/trip-history/{trip_id}")
def trip_detail(trip_id: int):
    with database.get_db() as conn:
        trip = conn.execute("SELECT * FROM trips WHERE id=?", (trip_id,)).fetchone()
        if not trip:
            return JSONResponse({"error": "Trip not found"}, status_code=404)
        events = conn.execute(
            "SELECT * FROM ride_events WHERE trip_id=? ORDER BY ts ASC", (trip_id,)
        ).fetchall()
        readings = conn.execute(
            "SELECT ts, speed_kmh, accel_y, tilt_roll, tilt_pitch, lat, lon, safe_rider_score "
            "FROM sensor_readings WHERE trip_id=? ORDER BY ts ASC", (trip_id,)
        ).fetchall()
        return {
            "trip": dict(trip),
            "events": [dict(e) for e in events],
            "readings": [dict(r) for r in readings],
        }


# --------------------------------------------------------------------------- #
# nearby services
# --------------------------------------------------------------------------- #
@app.get("/api/nearby-services")
def nearby_services(category: str = "fuel"):
    lat, lon = route_mod.position_at_distance(simulator.distance_km if simulator.trip_id else 0)
    return {
        "category": category,
        "center": {"lat": lat, "lon": lon},
        "results": services_data.get_nearby(category, lat, lon),
    }


# --------------------------------------------------------------------------- #
# ride journal
# --------------------------------------------------------------------------- #
@app.post("/api/journal")
def create_journal(entry: JournalCreate):
    with database.get_db() as conn:
        cur = conn.execute(
            "INSERT INTO ride_journal (trip_id, title, entry_date, start_location, destination, "
            "notes, photos_json, distance_km, duration_sec, safe_rider_score, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (entry.trip_id, entry.title, entry.entry_date, entry.start_location, entry.destination,
             entry.notes, json.dumps(entry.photos), entry.distance_km, entry.duration_sec,
             entry.safe_rider_score, time.time()),
        )
        new_id = cur.lastrowid
        row = conn.execute("SELECT * FROM ride_journal WHERE id=?", (new_id,)).fetchone()
        return dict(row)


@app.get("/api/journal")
def list_journal():
    with database.get_db() as conn:
        rows = conn.execute("SELECT * FROM ride_journal ORDER BY created_at DESC").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["photos"] = json.loads(d.pop("photos_json") or "[]")
            except Exception:
                d["photos"] = []
            out.append(d)
        return out


# --------------------------------------------------------------------------- #
# system info (About / Architecture page)
# --------------------------------------------------------------------------- #
@app.get("/api/system-info")
def system_info():
    feature_info = {}
    fi_path = ML_DIR / "feature_info.json"
    if fi_path.exists():
        feature_info = json.loads(fi_path.read_text())
    return {
        "ml": feature_info,
        "thresholds": {
            "overspeed_kmh": config.OVERSPEED_KMH,
            "harsh_braking_accel": config.HARSH_BRAKING_ACCEL,
            "sudden_accel_threshold": config.SUDDEN_ACCEL_THRESHOLD,
            "aggressive_turn_lateral": config.AGGRESSIVE_TURN_LATERAL,
            "aggressive_turn_gyro": config.AGGRESSIVE_TURN_GYRO,
            "aggressive_turn_tilt": config.AGGRESSIVE_TURN_TILT,
            "crash_impact_accel": config.CRASH_IMPACT_ACCEL,
            "crash_inactivity_ticks": config.CRASH_INACTIVITY_TICKS,
        },
        "score_deltas": config.SCORE_DELTAS,
        "score_bands": config.SCORE_BANDS,
        "tech_stack": {
            "backend": ["Python 3", "FastAPI", "Uvicorn", "WebSockets"],
            "ml": ["scikit-learn (RandomForestClassifier)", "pandas", "numpy", "joblib"],
            "frontend": ["HTML5 / CSS3 / JavaScript", "Chart.js", "Leaflet.js + OpenStreetMap"],
            "database": ["SQLite"],
            "sensors_modeled": ["GPS", "Accelerometer", "Gyroscope", "Tilt (roll/pitch)"],
        },
    }


@app.post("/api/settings/reset-demo-data")
def reset_demo_data():
    database.reset_demo_data()
    simulator.reset()
    return {"ok": True}


# --------------------------------------------------------------------------- #
# WebSocket live feed
# --------------------------------------------------------------------------- #
@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await websocket.send_text(json.dumps({"type": "snapshot", "state": simulator.state_snapshot()}, default=str))
        while True:
            # We don't need incoming messages, but must await to detect disconnects
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)


# --------------------------------------------------------------------------- #
# frontend (served last so it doesn't shadow /api or /ws routes)
# --------------------------------------------------------------------------- #
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
