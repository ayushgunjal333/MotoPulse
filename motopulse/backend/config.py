"""
MotoPulse shared configuration.

Every threshold that defines "what counts as risky riding" lives here so the
synthetic training data generator, the rule-based event detector, and the
Safe Rider Score engine all agree on the same numbers. This is the single
source of truth referenced throughout the backend and in the viva prep notes.
"""

# ---------------------------------------------------------------------------
# Feature thresholds (rule-based detection layer, runs alongside the ML model)
# ---------------------------------------------------------------------------
OVERSPEED_KMH = 80.0                 # speed above this -> overspeeding
HARSH_BRAKING_ACCEL = -4.5           # longitudinal accel (m/s^2) below this -> harsh braking
SUDDEN_ACCEL_THRESHOLD = 3.5         # longitudinal accel (m/s^2) above this -> sudden acceleration
AGGRESSIVE_TURN_LATERAL = 2.5        # lateral accel (m/s^2) above this -> aggressive turn
AGGRESSIVE_TURN_GYRO = 25.0          # gyro magnitude (deg/s) above this -> aggressive turn
AGGRESSIVE_TURN_TILT = 18.0          # tilt angle (deg) above this -> aggressive turn
CRASH_IMPACT_ACCEL = 9.0             # combined accel magnitude (m/s^2) above this -> possible impact
CRASH_INACTIVITY_SPEED = 3.0         # speed (km/h) below this after impact -> inactivity
CRASH_INACTIVITY_TICKS = 2           # consecutive low-motion ticks after impact to confirm crash

# ---------------------------------------------------------------------------
# Safe Rider Score
# ---------------------------------------------------------------------------
STARTING_SCORE = 100
SCORE_DELTAS = {
    "OVERSPEEDING": -8,
    "HARSH_BRAKING": -6,
    "AGGRESSIVE_TURN": -4,
    "SUDDEN_ACCELERATION": -3,
    "POSSIBLE_CRASH": -25,
}
SCORE_RECOVERY_PER_SAFE_TICK = 0.4   # slow regen while riding safely
SCORE_MIN = 0
SCORE_MAX = 100

SCORE_BANDS = [
    (100, 100, "Excellent"),
    (90, 99, "Very Safe"),
    (75, 89, "Good"),
    (60, 74, "Needs Attention"),
    (0, 59, "Risky"),
]

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
TICK_SECONDS = 1.5           # simulated seconds between sensor ticks (also wall-clock cadence)
CRASH_COUNTDOWN_SECONDS = 10

# ML feature order — every module that builds a feature vector MUST use this order
FEATURE_NAMES = [
    "speed_kmh",
    "longitudinal_accel",
    "lateral_accel",
    "gyro_magnitude",
    "tilt_angle",
    "braking_intensity",
    "sudden_accel_magnitude",
    "turning_intensity",
    "speed_deviation",
    "accel_variance",
]

CLASS_NAMES = {0: "SAFE", 1: "AGGRESSIVE"}
