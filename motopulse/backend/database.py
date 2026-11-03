"""
MotoPulse — SQLite database layer.

Kept deliberately simple (per project brief: "do not over-engineer the
database, the objective is a reliable academic demo"): plain sqlite3, no ORM,
one connection per request via a context manager.
"""
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "motopulse.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS riders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    rider_code TEXT NOT NULL,
    motorcycle TEXT,
    emergency_contact_name TEXT,
    emergency_contact_phone TEXT
);

CREATE TABLE IF NOT EXISTS trips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rider_id INTEGER NOT NULL,
    start_time REAL,
    end_time REAL,
    distance_km REAL DEFAULT 0,
    duration_sec REAL DEFAULT 0,
    avg_speed REAL DEFAULT 0,
    max_speed REAL DEFAULT 0,
    safe_rider_score REAL DEFAULT 100,
    risky_events_count INTEGER DEFAULT 0,
    status TEXT DEFAULT 'ongoing',
    start_location TEXT,
    destination TEXT,
    classification TEXT,
    FOREIGN KEY (rider_id) REFERENCES riders (id)
);

CREATE TABLE IF NOT EXISTS sensor_readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trip_id INTEGER NOT NULL,
    ts REAL,
    speed_kmh REAL,
    accel_x REAL, accel_y REAL, accel_z REAL,
    gyro_x REAL, gyro_y REAL, gyro_z REAL,
    tilt_roll REAL, tilt_pitch REAL,
    lat REAL, lon REAL,
    riding_status TEXT,
    safe_rider_score REAL,
    FOREIGN KEY (trip_id) REFERENCES trips (id)
);

CREATE TABLE IF NOT EXISTS ride_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trip_id INTEGER NOT NULL,
    ts REAL,
    event_type TEXT,
    severity TEXT,
    description TEXT,
    score_delta REAL,
    score_after REAL,
    sensor_evidence TEXT,
    FOREIGN KEY (trip_id) REFERENCES trips (id)
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trip_id INTEGER,
    alert_type TEXT,
    ts REAL,
    status TEXT,
    rider_name TEXT,
    lat REAL,
    lon REAL,
    emergency_contact TEXT,
    details TEXT
);

CREATE TABLE IF NOT EXISTS ride_journal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trip_id INTEGER,
    title TEXT,
    entry_date TEXT,
    start_location TEXT,
    destination TEXT,
    notes TEXT,
    photos_json TEXT,
    distance_km REAL,
    duration_sec REAL,
    safe_rider_score REAL,
    created_at REAL
);
"""


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(seed: bool = True):
    with get_db() as conn:
        conn.executescript(SCHEMA)

        rider = conn.execute("SELECT * FROM riders LIMIT 1").fetchone()
        if rider is None and seed:
            conn.execute(
                "INSERT INTO riders (name, rider_code, motorcycle, "
                "emergency_contact_name, emergency_contact_phone) VALUES (?,?,?,?,?)",
                ("G G Rajesh", "MP-RIDER-0418", "Royal Enfield Classic 350",
                 "Ayush Gunjal (Emergency Contact)", "+91 98450 12345"),
            )
            rider_id = conn.execute("SELECT id FROM riders LIMIT 1").fetchone()["id"]
            _seed_demo_history(conn, rider_id)


def _seed_demo_history(conn, rider_id):
    """Pre-populate a few past trips (with real synthetic telemetry, not just
    summary rows) + a journal entry, so Trip History / Ride Journal / Ride
    Analytics all have something to show on a fresh install."""
    now = time.time()
    demo_trips = [
        dict(days_ago=6, distance=18.7, duration=32 * 60, avg=42, mx=71, score=87, events=2,
             start="BIT College, V.V. Puram", dest="Koramangala", cls="Good",
             event_types=["OVERSPEEDING", "HARSH_BRAKING"]),
        dict(days_ago=4, distance=84.0, duration=137 * 60, avg=48, mx=96, score=91, events=1,
             start="Basavanagudi", dest="Nandi Hills", cls="Very Safe",
             event_types=["OVERSPEEDING"]),
        dict(days_ago=2, distance=11.2, duration=21 * 60, avg=38, mx=88, score=63, events=5,
             start="Jayanagar", dest="Electronic City", cls="Needs Attention",
             event_types=["OVERSPEEDING", "HARSH_BRAKING", "AGGRESSIVE_TURN", "OVERSPEEDING", "SUDDEN_ACCELERATION"]),
        dict(days_ago=1, distance=25.4, duration=44 * 60, avg=45, mx=68, score=96, events=0,
             start="Home", dest="MG Road", cls="Very Safe", event_types=[]),
    ]
    for t in demo_trips:
        start_time = now - t["days_ago"] * 86400
        end_time = start_time + t["duration"]
        cur = conn.execute(
            "INSERT INTO trips (rider_id, start_time, end_time, distance_km, duration_sec, "
            "avg_speed, max_speed, safe_rider_score, risky_events_count, status, "
            "start_location, destination, classification) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (rider_id, start_time, end_time, t["distance"], t["duration"], t["avg"], t["mx"],
             t["score"], t["events"], "completed", t["start"], t["dest"], t["cls"]),
        )
        trip_id = cur.lastrowid
        _generate_trip_telemetry(conn, trip_id, start_time, end_time, t["avg"], t["mx"],
                                  t["score"], t["event_types"])
        if t["dest"] == "Nandi Hills":
            conn.execute(
                "INSERT INTO ride_journal (trip_id, title, entry_date, start_location, "
                "destination, notes, photos_json, distance_km, duration_sec, safe_rider_score, "
                "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (trip_id, "Weekend Ride to Nandi Hills",
                 time.strftime("%Y-%m-%d", time.localtime(start_time)),
                 t["start"], t["dest"], "Smooth morning ride with minimal traffic.",
                 json.dumps([]), t["distance"], t["duration"], t["score"], start_time),
            )


def _generate_trip_telemetry(conn, trip_id, start_time, end_time, avg_speed, max_speed,
                              final_score, event_types):
    """Synthetic sensor_readings + ride_events for a seeded historical trip so
    the Ride Analytics page has real charts to show, not just a summary row."""
    import math
    import random as _r
    rnd = _r.Random(trip_id * 97 + 13)

    duration = max(60, end_time - start_time)
    n_points = max(20, min(60, int(duration / 45)))
    event_ticks = sorted(rnd.sample(range(2, n_points - 2), min(len(event_types), max(0, n_points - 4)))) \
        if event_types else []

    score = 100.0
    score_path = {}
    for idx, tick in enumerate(event_ticks):
        etype = event_types[idx % len(event_types)] if event_types else None
        delta = {"OVERSPEEDING": -8, "HARSH_BRAKING": -6, "AGGRESSIVE_TURN": -4,
                 "SUDDEN_ACCELERATION": -3}.get(etype, -5)
        score = max(0, score + delta)
        score_path[tick] = (etype, delta, score)

    for i in range(n_points):
        frac = i / max(1, n_points - 1)
        ts = start_time + frac * duration
        speed = max(8, avg_speed + 10 * math.sin(frac * 6.0) + rnd.uniform(-4, 4))
        accel_y = rnd.gauss(0, 0.35)
        accel_x = rnd.gauss(0, 0.25)
        accel_z = 9.8 + rnd.gauss(0, 0.15)
        gyro_x, gyro_y, gyro_z = rnd.gauss(0, 1.5), rnd.gauss(0, 1.5), rnd.gauss(0, 3)
        tilt_roll, tilt_pitch = rnd.gauss(0, 2), rnd.gauss(0, 1.5)
        riding_status = "SAFE"

        if i in score_path:
            etype, delta, sc = score_path[i]
            if etype == "OVERSPEEDING":
                speed = max_speed - rnd.uniform(0, 4)
            elif etype == "HARSH_BRAKING":
                accel_y = rnd.uniform(-8.0, -5.2)
            elif etype == "SUDDEN_ACCELERATION":
                accel_y = rnd.uniform(4.5, 6.8)
            elif etype == "AGGRESSIVE_TURN":
                sign = rnd.choice([-1, 1])
                accel_x = sign * rnd.uniform(3.5, 5.5)
                gyro_z = sign * rnd.uniform(35, 55)
                tilt_roll = sign * rnd.uniform(24, 34)
            riding_status = "RISKY"
            score = sc
            evidence = {
                "OVERSPEEDING": f"Speed {speed:.0f} km/h (limit 80 km/h)",
                "HARSH_BRAKING": f"Longitudinal accel {accel_y:.1f} m/s^2 (threshold -4.5)",
                "SUDDEN_ACCELERATION": f"Longitudinal accel {accel_y:.1f} m/s^2 (threshold 3.5)",
                "AGGRESSIVE_TURN": f"Lateral accel {accel_x:.1f} m/s^2, gyro {gyro_z:.0f} deg/s, tilt {tilt_roll:.0f} deg",
            }.get(etype, "")
            conn.execute(
                "INSERT INTO ride_events (trip_id, ts, event_type, severity, description, "
                "score_delta, score_after, sensor_evidence) VALUES (?,?,?,?,?,?,?,?)",
                (trip_id, ts, etype,
                 "HIGH" if etype == "OVERSPEEDING" else "MEDIUM",
                 {"OVERSPEEDING": "Overspeeding detected", "HARSH_BRAKING": "Harsh braking detected",
                  "SUDDEN_ACCELERATION": "Sudden acceleration detected",
                  "AGGRESSIVE_TURN": "Aggressive turn detected"}.get(etype, etype),
                 delta, round(score, 1), evidence),
            )

        lat = 12.9432 + frac * 0.04 + rnd.uniform(-0.002, 0.002)
        lon = 77.5722 + frac * 0.05 + rnd.uniform(-0.002, 0.002)
        conn.execute(
            "INSERT INTO sensor_readings (trip_id, ts, speed_kmh, accel_x, accel_y, accel_z, "
            "gyro_x, gyro_y, gyro_z, tilt_roll, tilt_pitch, lat, lon, riding_status, "
            "safe_rider_score) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (trip_id, ts, speed, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z,
             tilt_roll, tilt_pitch, lat, lon, riding_status, round(score, 1)),
        )


def reset_demo_data():
    """Wipe everything and reseed -- used by Settings 'Reset Demo Data'."""
    with get_db() as conn:
        for table in ["sensor_readings", "ride_events", "alerts", "ride_journal", "trips", "riders"]:
            conn.execute(f"DELETE FROM {table}")
    init_db(seed=True)
