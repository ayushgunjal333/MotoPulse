from typing import List, Optional

from pydantic import BaseModel, Field


class SimulateEventRequest(BaseModel):
    event_type: str = Field(..., description="overspeed | harsh_braking | sudden_acceleration | aggressive_turn | crash | safe")


class PredictRequest(BaseModel):
    """Raw sensor snapshot -- lets a faculty member test /api/predict directly
    (e.g. with curl) without going through the simulator."""
    speed_kmh: float = 45.0
    accel_x: float = 0.0     # lateral
    accel_y: float = 0.0     # longitudinal
    accel_z: float = 9.8     # vertical (gravity baseline)
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0
    tilt_roll: float = 0.0
    tilt_pitch: float = 0.0
    avg_recent_speed: Optional[float] = None  # for speed_deviation; defaults to speed_kmh


class JournalCreate(BaseModel):
    title: str
    entry_date: str
    start_location: str
    destination: str
    notes: str = ""
    photos: List[str] = []
    trip_id: Optional[int] = None
    distance_km: Optional[float] = None
    duration_sec: Optional[float] = None
    safe_rider_score: Optional[float] = None
