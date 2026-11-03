"""
MotoPulse — Real-Time Ride Simulator & Telemetry Replay Engine.

Replays unseen real/simulated telemetry from CSV row-by-row, feeds each tick into
feature extraction -> ML prediction -> rule thresholds -> Safe Rider Score,
and streams the live snapshot over WebSockets.
"""
import asyncio
import csv
import math
import os
import random
import time
from collections import deque
from statistics import pvariance
from pathlib import Path

import config
import database
from ml import predictor
from scoring import ScoreEngine
from . import route as route_mod

EVENT_DESCRIPTIONS = {
    "OVERSPEEDING": "Speed exceeded the {limit} km/h safe threshold.",
    "HARSH_BRAKING": "Sudden deceleration detected (longitudinal accel {val} m/s^2).",
    "SUDDEN_ACCELERATION": "Rapid forward acceleration detected ({val} m/s^2).",
    "AGGRESSIVE_TURN": "High lateral force and lean angle detected during a turn.",
    "POSSIBLE_CRASH": "Impact spike followed by motion inactivity -- possible crash.",
}

CSV_DATA_PATH = Path(__file__).resolve().parent / "raw_unseen_ride.csv"


class RideSimulator:
    def __init__(self, broadcast_fn):
        self._broadcast = broadcast_fn
        self._task = None
        self._countdown_task = None
        self.raw_data_stream = []
        self._load_telemetry_dataset()
        self.reset()

    def _load_telemetry_dataset(self):
        """Loads pre-recorded / unseen telemetry CSV for deterministic testing."""
        self.raw_data_stream = []
        if CSV_DATA_PATH.exists():
            with open(CSV_DATA_PATH, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.raw_data_stream.append({
                        "speed_kmh": float(row["speed_kmh"]),
                        "accel_x": float(row["accel_x"]),
                        "accel_y": float(row["accel_y"]),
                        "accel_z": float(row["accel_z"]),
                        "gyro_x": float(row["gyro_x"]),
                        "gyro_y": float(row["gyro_y"]),
                        "gyro_z": float(row["gyro_z"]),
                        "tilt_roll": float(row["tilt_roll"]),
                        "tilt_pitch": float(row["tilt_pitch"]),
                    })
            print(f"[Simulator] Loaded {len(self.raw_data_stream)} telemetry ticks from {CSV_DATA_PATH.name}")
        else:
            print(f"[Simulator] {CSV_DATA_PATH.name} not found. Defaulting to live procedural generation.")

    def reset(self):
        self.status = "IDLE"          # IDLE | RUNNING | PAUSED | ENDED
        self.crash_phase = None       # None | impact | inactivity | countdown | triggered | cancelled
        self.trip_id = None
        self.rider = None
        self.start_time = None
        self.elapsed_sec = 0.0
        self.distance_km = 0.0
        self.baseline_speed = 42.0
        self.speed = 0.0
        self.speed_history = deque(maxlen=8)
        self.long_accel_history = deque(maxlen=6)
        self.speed_sum = 0.0
        self.speed_samples = 0
        self.max_speed = 0.0
        self.risky_events_count = 0
        self.scenario = None
        self.scenario_ticks_left = 0
        self.inactivity_ticks = 0
        self.countdown_seconds_left = 0
        self.score_engine = ScoreEngine()
        self.recent_events = []
        self.last_reading = None
        self.last_prediction = None
        self.last_alert = None
        self.csv_index = 0

    async def start(self, rider_row):
        if self.status == "RUNNING":
            return self.state_snapshot()
        self.reset()
        self.rider = dict(rider_row)
        self.status = "RUNNING"
        self.start_time = time.time()
        with database.get_db() as conn:
            cur = conn.execute(
                "INSERT INTO trips (rider_id, start_time, distance_km, duration_sec, "
                "safe_rider_score, status, start_location, destination) VALUES (?,?,?,?,?,?,?,?)",
                (self.rider["id"], self.start_time, 0, 0, config.STARTING_SCORE, "ongoing",
                 "BIT College, V.V. Puram", "Nandi Hills Road (demo route)"),
            )
            self.trip_id = cur.lastrowid
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run_loop())
        return self.state_snapshot()

    def pause(self):
        if self.status == "RUNNING":
            self.status = "PAUSED"
        return self.state_snapshot()

    def resume(self):
        if self.status == "PAUSED":
            self.status = "RUNNING"
        return self.state_snapshot()

    async def end(self):
        was_active = self.status in ("RUNNING", "PAUSED") and self.trip_id is not None
        self.status = "ENDED"
        if self._countdown_task:
            self._countdown_task.cancel()
            self._countdown_task = None
        summary = None
        if was_active:
            avg_speed = self.speed_sum / self.speed_samples if self.speed_samples else 0
            duration = time.time() - self.start_time
            classification = self.score_engine.breakdown()["band"]
            with database.get_db() as conn:
                conn.execute(
                    "UPDATE trips SET end_time=?, distance_km=?, duration_sec=?, avg_speed=?, "
                    "max_speed=?, safe_rider_score=?, risky_events_count=?, status=?, "
                    "classification=? WHERE id=?",
                    (time.time(), round(self.distance_km, 2), round(duration, 1),
                     round(avg_speed, 1), round(self.max_speed, 1),
                     round(self.score_engine.score, 1), self.risky_events_count,
                     "completed", classification, self.trip_id),
                )
            summary = {
                "trip_id": self.trip_id,
                "distance_km": round(self.distance_km, 2),
                "duration_sec": round(duration, 1),
                "avg_speed": round(avg_speed, 1),
                "max_speed": round(self.max_speed, 1),
                "safe_rider_score": round(self.score_engine.score, 1),
                "risky_events_count": self.risky_events_count,
                "classification": classification,
            }
        await self._broadcast({"type": "trip_ended", "summary": summary})
        return summary

    def inject_scenario(self, name):
        if self.status != "RUNNING" or self.crash_phase is not None:
            return {"ok": False, "reason": "Ride not active or a crash sequence is in progress."}
        durations = {
            "overspeed": 5, "harsh_braking": 2, "sudden_acceleration": 2,
            "aggressive_turn": 3, "crash": 1,
        }
        if name not in durations:
            return {"ok": False, "reason": f"Unknown scenario '{name}'"}
        self.scenario = name
        self.scenario_ticks_left = durations[name]
        if name == "crash":
            self.crash_phase = "impact"
        return {"ok": True}

    async def cancel_sos(self):
        if self.crash_phase == "countdown":
            if self._countdown_task:
                self._countdown_task.cancel()
            self.crash_phase = "cancelled"
            await self._log_event("SOS_CANCELLED", "INFO", "Rider cancelled the SOS alert.", 0)
            await self._broadcast({"type": "sos_cancelled"})
            self._resume_after_crash()
        return self.state_snapshot()

    async def trigger_sos(self, manual=False):
        loc = route_mod.position_at_distance(self.distance_km) if self.trip_id else route_mod.ROUTE[0]
        alert = {
            "trip_id": self.trip_id,
            "alert_type": "SOS",
            "ts": time.time(),
            "status": "SENT",
            "rider_name": self.rider["name"] if self.rider else "Demo Rider",
            "lat": loc[0], "lon": loc[1],
            "emergency_contact": self.rider["emergency_contact_name"] if self.rider else "N/A",
            "details": "Manual test" if manual else "Auto-triggered after crash countdown",
        }
        with database.get_db() as conn:
            conn.execute(
                "INSERT INTO alerts (trip_id, alert_type, ts, status, rider_name, lat, lon, "
                "emergency_contact, details) VALUES (?,?,?,?,?,?,?,?,?)",
                (alert["trip_id"], alert["alert_type"], alert["ts"], alert["status"],
                 alert["rider_name"], alert["lat"], alert["lon"], alert["emergency_contact"],
                 alert["details"]),
            )
        self.last_alert = alert
        if self.crash_phase == "countdown":
            self.crash_phase = "triggered"
            if self._countdown_task:
                self._countdown_task.cancel()
        await self._broadcast({"type": "sos_triggered", "alert": alert})
        if not manual:
            self._resume_after_crash()
        return alert

    def _resume_after_crash(self):
        self.crash_phase = None
        self.scenario = None
        self.baseline_speed = 28.0
        self.speed = 20.0
        self.inactivity_ticks = 0

    async def _run_loop(self):
        while self.status in ("RUNNING", "PAUSED"):
            if self.status == "PAUSED":
                await asyncio.sleep(0.4)
                continue
            await self._tick()
            await asyncio.sleep(config.TICK_SECONDS)

    async def _tick(self):
        ts = time.time()
        self.elapsed_sec = ts - self.start_time

        raw = self._generate_raw_reading()
        features = self._extract_features(raw)

        try:
            ml_result = predictor.predict_from_features(features)
        except RuntimeError:
            ml_result = {"label": "SAFE", "label_id": 0, "confidence": 0.0, "probabilities": {}}

        risk_type = self._rule_based_risk_type(features)
        riding_status = "RISKY" if risk_type else "SAFE"

        d_km = (raw["speed_kmh"] / 3600.0) * config.TICK_SECONDS
        self.distance_km += d_km
        self.speed = raw["speed_kmh"]
        self.speed_sum += raw["speed_kmh"]
        self.speed_samples += 1
        self.max_speed = max(self.max_speed, raw["speed_kmh"])
        lat, lon = route_mod.position_at_distance(self.distance_km)

        score_delta = 0
        if risk_type:
            score_delta, _ = self.score_engine.apply_event(risk_type, ts)
            self.risky_events_count += 1
            await self._log_event(risk_type, self._severity(risk_type), self._evidence_text(risk_type, features),
                                   score_delta, features=features, ml_result=ml_result)
        else:
            self.score_engine.apply_safe_tick(ts)

        reading = {
            "ts": ts,
            "gps": {"lat": lat, "lon": lon, "speed_kmh": round(raw["speed_kmh"], 1)},
            "accelerometer": {
                "x": round(raw["accel_x"], 2), "y": round(raw["accel_y"], 2), "z": round(raw["accel_z"], 2),
                "magnitude": round(math.sqrt(raw["accel_x"]**2 + raw["accel_y"]**2 + raw["accel_z"]**2), 2),
            },
            "gyroscope": {
                "x": round(raw["gyro_x"], 1), "y": round(raw["gyro_y"], 1), "z": round(raw["gyro_z"], 1),
                "magnitude": round(features["gyro_magnitude"], 1),
            },
            "tilt": {"roll": round(raw["tilt_roll"], 1), "pitch": round(raw["tilt_pitch"], 1)},
            "distance_km": round(self.distance_km, 3),
            "duration_sec": round(self.elapsed_sec, 1),
            "riding_status": "CRASH_SUSPECTED" if self.crash_phase in ("impact", "inactivity") else riding_status,
            "risk_type": risk_type,
        }
        self.last_reading = reading
        self.last_prediction = ml_result

        with database.get_db() as conn:
            conn.execute(
                "INSERT INTO sensor_readings (trip_id, ts, speed_kmh, accel_x, accel_y, accel_z, "
                "gyro_x, gyro_y, gyro_z, tilt_roll, tilt_pitch, lat, lon, riding_status, "
                "safe_rider_score) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (self.trip_id, ts, raw["speed_kmh"], raw["accel_x"], raw["accel_y"], raw["accel_z"],
                 raw["gyro_x"], raw["gyro_y"], raw["gyro_z"], raw["tilt_roll"], raw["tilt_pitch"],
                 lat, lon, reading["riding_status"], self.score_engine.score),
            )
            conn.execute("UPDATE trips SET distance_km=?, duration_sec=?, safe_rider_score=?, "
                         "max_speed=?, risky_events_count=? WHERE id=?",
                         (round(self.distance_km, 3), round(self.elapsed_sec, 1),
                          round(self.score_engine.score, 1), round(self.max_speed, 1),
                          self.risky_events_count, self.trip_id))

        await self._broadcast({
            "type": "tick",
            "reading": reading,
            "prediction": ml_result,
            "score": self.score_engine.breakdown(),
            "features": {k: round(v, 3) for k, v in features.items()},
        })

        self._advance_crash_state_machine(features)

    def _generate_raw_reading(self):
        if self.crash_phase == "impact":
            return self._crash_impact_reading()
        if self.crash_phase in ("inactivity", "countdown"):
            return self._crash_inactivity_reading()

        if self.scenario and self.scenario_ticks_left > 0:
            self.scenario_ticks_left -= 1
            raw = self._scenario_reading(self.scenario)
            if self.scenario_ticks_left == 0:
                self.scenario = None
            return raw

        if self.raw_data_stream:
            row = self.raw_data_stream[self.csv_index]
            self.csv_index = (self.csv_index + 1) % len(self.raw_data_stream)
            self.baseline_speed = row["speed_kmh"]
            return dict(row)

        return self._baseline_reading()

    def _baseline_reading(self):
        self.baseline_speed = max(15, min(70, self.baseline_speed + random.uniform(-2.5, 2.5)))
        speed = max(8, self.baseline_speed + random.uniform(-3, 3))
        accel_y = random.gauss(0, 0.3)
        accel_x = random.gauss(0, 0.2)
        accel_z = 9.8 + random.gauss(0, 0.15)
        gyro_z = random.gauss(0, 3)
        gyro_x = random.gauss(0, 1.5)
        gyro_y = random.gauss(0, 1.5)
        tilt_roll = max(-8, min(8, random.gauss(0, 2)))
        tilt_pitch = max(-6, min(6, random.gauss(0, 1.5)))
        return dict(speed_kmh=speed, accel_x=accel_x, accel_y=accel_y, accel_z=accel_z,
                    gyro_x=gyro_x, gyro_y=gyro_y, gyro_z=gyro_z,
                    tilt_roll=tilt_roll, tilt_pitch=tilt_pitch)

    def _scenario_reading(self, name):
        base = self._baseline_reading()
        if name == "overspeed":
            base["speed_kmh"] = random.uniform(90, 106)
            base["accel_y"] = random.uniform(0.8, 1.6)
            self.baseline_speed = base["speed_kmh"] * 0.6
        elif name == "harsh_braking":
            base["accel_y"] = random.uniform(-8.2, -5.0)
            base["speed_kmh"] = max(10, self.speed - random.uniform(15, 28))
            self.baseline_speed = base["speed_kmh"]
        elif name == "sudden_acceleration":
            base["accel_y"] = random.uniform(4.2, 7.2)
            base["speed_kmh"] = min(115, self.speed + random.uniform(18, 30))
            self.baseline_speed = base["speed_kmh"]
        elif name == "aggressive_turn":
            sign = random.choice([-1, 1])
            base["accel_x"] = sign * random.uniform(3.2, 6.0)
            base["gyro_z"] = sign * random.uniform(32, 62)
            base["tilt_roll"] = sign * random.uniform(22, 38)
            base["speed_kmh"] = max(15, self.speed - random.uniform(3, 9))
        return base

    def _crash_impact_reading(self):
        return dict(
            speed_kmh=max(0, self.speed - random.uniform(20, 35)),
            accel_x=random.uniform(-6, 6),
            accel_y=random.uniform(-9.5, -6.5),
            accel_z=random.uniform(15, 21),
            gyro_x=random.uniform(-70, 70), gyro_y=random.uniform(-70, 70), gyro_z=random.uniform(-90, 90),
            tilt_roll=random.uniform(45, 85) * random.choice([-1, 1]),
            tilt_pitch=random.uniform(20, 45) * random.choice([-1, 1]),
        )

    def _crash_inactivity_reading(self):
        return dict(
            speed_kmh=max(0, random.uniform(0, 1.5)),
            accel_x=random.gauss(0, 0.1), accel_y=random.gauss(0, 0.1), accel_z=9.8 + random.gauss(0, 0.1),
            gyro_x=random.gauss(0, 0.5), gyro_y=random.gauss(0, 0.5), gyro_z=random.gauss(0, 0.5),
            tilt_roll=self.last_reading["tilt"]["roll"] if self.last_reading else 70.0,
            tilt_pitch=self.last_reading["tilt"]["pitch"] if self.last_reading else 30.0,
        )

    def _extract_features(self, raw):
        self.speed_history.append(raw["speed_kmh"])
        self.long_accel_history.append(raw["accel_y"])
        gyro_mag = math.sqrt(raw["gyro_x"] ** 2 + raw["gyro_y"] ** 2 + raw["gyro_z"] ** 2)
        tilt_angle = math.sqrt(raw["tilt_roll"] ** 2 + raw["tilt_pitch"] ** 2)
        avg_speed = sum(self.speed_history) / len(self.speed_history)
        speed_deviation = abs(raw["speed_kmh"] - avg_speed)
        accel_var = pvariance(self.long_accel_history) if len(self.long_accel_history) > 1 else 0.0
        braking_intensity = max(0.0, -raw["accel_y"])
        sudden_accel_magnitude = max(0.0, raw["accel_y"])
        turning_intensity = abs(raw["accel_x"]) * abs(gyro_mag) / 10.0
        return {
            "speed_kmh": raw["speed_kmh"],
            "longitudinal_accel": raw["accel_y"],
            "lateral_accel": raw["accel_x"],
            "gyro_magnitude": gyro_mag,
            "tilt_angle": tilt_angle,
            "braking_intensity": braking_intensity,
            "sudden_accel_magnitude": sudden_accel_magnitude,
            "turning_intensity": turning_intensity,
            "speed_deviation": speed_deviation,
            "accel_variance": accel_var,
        }

    def _rule_based_risk_type(self, f):
        if self.crash_phase in ("impact", "inactivity", "countdown"):
            return None
        if f["speed_kmh"] > config.OVERSPEED_KMH:
            return "OVERSPEEDING"
        if f["longitudinal_accel"] <= config.HARSH_BRAKING_ACCEL:
            return "HARSH_BRAKING"
        if f["longitudinal_accel"] >= config.SUDDEN_ACCEL_THRESHOLD:
            return "SUDDEN_ACCELERATION"
        turn_signals = [
            abs(f["lateral_accel"]) >= config.AGGRESSIVE_TURN_LATERAL,
            f["gyro_magnitude"] >= config.AGGRESSIVE_TURN_GYRO,
            f["tilt_angle"] >= config.AGGRESSIVE_TURN_TILT,
        ]
        if sum(turn_signals) >= 2:
            return "AGGRESSIVE_TURN"
        return None

    def _severity(self, risk_type):
        return {
            "OVERSPEEDING": "HIGH", "HARSH_BRAKING": "MEDIUM", "SUDDEN_ACCELERATION": "MEDIUM",
            "AGGRESSIVE_TURN": "MEDIUM", "POSSIBLE_CRASH": "CRITICAL",
        }.get(risk_type, "LOW")

    def _evidence_text(self, risk_type, f):
        if risk_type == "OVERSPEEDING":
            return f"Speed {f['speed_kmh']:.0f} km/h (limit {config.OVERSPEED_KMH:.0f} km/h)"
        if risk_type == "HARSH_BRAKING":
            return f"Longitudinal accel {f['longitudinal_accel']:.1f} m/s^2 (threshold {config.HARSH_BRAKING_ACCEL})"
        if risk_type == "SUDDEN_ACCELERATION":
            return f"Longitudinal accel {f['longitudinal_accel']:.1f} m/s^2 (threshold {config.SUDDEN_ACCEL_THRESHOLD})"
        if risk_type == "AGGRESSIVE_TURN":
            return f"Lateral accel {f['lateral_accel']:.1f} m/s^2, gyro {f['gyro_magnitude']:.0f} deg/s, tilt {f['tilt_angle']:.0f} deg"
        if risk_type == "POSSIBLE_CRASH":
            return f"Impact spike then {config.CRASH_INACTIVITY_TICKS}+ ticks of near-zero motion"
        return ""

    async def _log_event(self, event_type, severity, description, score_delta, features=None, ml_result=None):
        ts = time.time()
        evidence = description
        event = {
            "ts": ts, "event_type": event_type, "severity": severity,
            "description": self._friendly_label(event_type), "sensor_evidence": evidence,
            "score_delta": score_delta, "score_after": round(self.score_engine.score, 1),
        }
        self.recent_events.insert(0, event)
        self.recent_events = self.recent_events[:50]
        with database.get_db() as conn:
            conn.execute(
                "INSERT INTO ride_events (trip_id, ts, event_type, severity, description, "
                "score_delta, score_after, sensor_evidence) VALUES (?,?,?,?,?,?,?,?)",
                (self.trip_id, ts, event_type, severity, event["description"], score_delta,
                 event["score_after"], evidence),
            )
        await self._broadcast({"type": "event", "event": event})

    def _friendly_label(self, event_type):
        return {
            "OVERSPEEDING": "Overspeeding detected",
            "HARSH_BRAKING": "Harsh braking detected",
            "SUDDEN_ACCELERATION": "Sudden acceleration detected",
            "AGGRESSIVE_TURN": "Aggressive turn detected",
            "POSSIBLE_CRASH": "POSSIBLE CRASH DETECTED",
            "SOS_CANCELLED": "SOS alert cancelled by rider",
        }.get(event_type, event_type)

    def _advance_crash_state_machine(self, features):
        if self.crash_phase == "impact":
            asyncio.create_task(self._log_event(
                "POSSIBLE_CRASH", "CRITICAL",
                "Impact spike detected -- checking for motion inactivity...", 0))
            self.crash_phase = "inactivity"
            self.inactivity_ticks = 0
        elif self.crash_phase == "inactivity":
            self.inactivity_ticks += 1
            if self.inactivity_ticks >= config.CRASH_INACTIVITY_TICKS:
                score_delta, _ = self.score_engine.apply_event("POSSIBLE_CRASH", time.time())
                self.risky_events_count += 1
                asyncio.create_task(self._log_event(
                    "POSSIBLE_CRASH", "CRITICAL",
                    self._evidence_text("POSSIBLE_CRASH", features), score_delta))
                self.crash_phase = "countdown"
                self.countdown_seconds_left = config.CRASH_COUNTDOWN_SECONDS
                self._countdown_task = asyncio.create_task(self._run_countdown())

    async def _run_countdown(self):
        try:
            await self._broadcast({
                "type": "crash_detected",
                "seconds": config.CRASH_COUNTDOWN_SECONDS,
                "location": {"lat": route_mod.position_at_distance(self.distance_km)[0],
                             "lon": route_mod.position_at_distance(self.distance_km)[1]},
            })
            for remaining in range(config.CRASH_COUNTDOWN_SECONDS, 0, -1):
                self.countdown_seconds_left = remaining
                await self._broadcast({"type": "crash_countdown", "seconds_left": remaining})
                await asyncio.sleep(1.0)
            self.countdown_seconds_left = 0
            await self.trigger_sos(manual=False)
        except asyncio.CancelledError:
            pass

    def state_snapshot(self):
        return {
            "status": self.status,
            "crash_phase": self.crash_phase,
            "trip_id": self.trip_id,
            "distance_km": round(self.distance_km, 3),
            "duration_sec": round(self.elapsed_sec, 1),
            "speed_kmh": round(self.speed, 1),
            "score": self.score_engine.breakdown(),
            "risky_events_count": self.risky_events_count,
            "last_reading": self.last_reading,
            "last_prediction": self.last_prediction,
            "recent_events": self.recent_events[:15],
            "countdown_seconds_left": self.countdown_seconds_left,
        }