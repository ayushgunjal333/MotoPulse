# MotoPulse — Viva Preparation

Likely faculty questions with strong, simple answers.

**Q: Why Random Forest?**
It handles tabular sensor features well without heavy preprocessing, is robust to noisy
readings, resists overfitting better than a single decision tree (it averages many
trees), and — importantly for a viva — its feature-importance output is easy to explain:
you can literally show which sensor mattered most to the decision.

**Q: Why accelerometer?**
It directly measures forces on the rider: braking (negative longitudinal), acceleration
(positive longitudinal), and cornering force (lateral). It's the single most informative
sensor for detecting harsh events and, combined with gravity on the Z-axis, can also hint
at impacts (a sudden large-magnitude spike).

**Q: Why gyroscope?**
Accelerometer alone can't distinguish "turning safely" from "turning aggressively" —
gyroscope gives the rate of rotation (deg/s), so a fast yaw/roll rate combined with high
lateral force and lean angle is what actually signals an aggressive turn, not just one
signal in isolation.

**Q: Why GPS?**
For speed (cross-checked against accelerometer-derived speed changes), location for the
map/trip route, and distance/duration calculation for trip summaries. It's also what
powers Nearby Services and the SOS location share.

**Q: How do you detect harsh braking?**
Rule-based: longitudinal acceleration below −4.5 m/s² (`config.HARSH_BRAKING_ACCEL`).
This threshold is cross-checked by the ML model's prediction on the full feature vector,
so a single noisy spike that doesn't fit the broader "braking" pattern the model learned
is less likely to be treated as high-confidence.

**Q: How do you detect aggressive turning?**
At least **two of three** signals must cross their thresholds together: lateral
acceleration ≥ 2.5 m/s², gyroscope magnitude ≥ 25°/s, and tilt angle ≥ 18°. Requiring 2-of-3
avoids false positives from a single noisy sensor reading — a real design decision, not
an accident.

**Q: How does crash detection work?**
A hybrid rule, matching the project proposal: (1) an impact spike — combined acceleration
magnitude far above normal riding forces, then (2) two or more consecutive ticks of
near-zero motion (speed and acceleration both near zero) immediately after. Only when
both conditions are met in sequence does the system raise "POSSIBLE CRASH DETECTED" and
start the 10-second SOS countdown — a single spike (like a pothole) without the following
inactivity does *not* trigger a crash alert.

**Q: How is Safe Rider Score calculated?**
Starts at 100 for every trip. Each detected risk event subtracts a fixed, documented
amount (Overspeeding −8, Harsh braking −6, Aggressive turning −4, Sudden acceleration −3,
Possible crash −25), and the score slowly regenerates during sustained safe riding. Every
change is logged with a timestamp, the exact sensor evidence, and the delta — so the
score is always explainable, not a mystery number.

**Q: What happens if GPS is unavailable?**
The system is designed to degrade gracefully: the map falls back to a "no signal" state
without crashing, and risk detection continues to work off accelerometer/gyroscope/tilt
alone, since those don't depend on GPS. Only distance/route-based features would be
unavailable.

**Q: What happens if the phone is placed differently (e.g. in a pocket vs. mounted)?**
This is a real limitation of phone-based sensing in general — orientation changes the
mapping between phone axes and vehicle axes. A production version would need a
calibration step (e.g. asking the rider to keep the phone in a fixed mount, or using
sensor fusion to estimate device orientation and re-project axes). It's listed under
Future Improvements.

**Q: What are false positives, and how do you reduce them?**
A false positive is the system flagging safe riding as risky (e.g. a pothole read as
harsh braking, or a speed bump read as a crash impact). We reduce them two ways: (1) the
2-of-3 rule for aggressive turns, and (2) requiring *sustained* inactivity (not just one
low-motion tick) after an impact spike before declaring a crash — a single jolt alone
never triggers the SOS countdown.

**Q: What dataset is used?**
A synthetic dataset generated to match realistic riding-behaviour distributions (safe
riding, overspeeding, harsh braking, sudden acceleration, aggressive turning), documented
openly in `train_model.py` and on the About page, with light label noise and feature
jitter added so the classes overlap somewhat rather than being perfectly separable —
because real-world sensor data would never separate so cleanly. This is stated honestly
because real-world labeled motorcycle telemetry was not available for the project.

**Q: How can this system be improved?**
Real labeled telemetry data to retrain the model; a real Places/routing API; a companion
mobile app for actual sensor access instead of simulation; sensor-fusion-based
orientation calibration; real SMS/push notification integration for SOS; multi-rider
accounts; and richer crash-severity classification (e.g. distinguishing a fall from a
full collision).

**Q: Why FastAPI?**
It's async-native (needed for the WebSocket live feed and the background ride-simulator
task running concurrently with request handling), has automatic request/response
validation via Pydantic, and ships interactive API docs for free — useful for
demonstrating `/api/predict` directly.

**Q: Why SQLite?**
Zero-configuration, file-based, and more than sufficient for a single-rider academic
prototype — no separate database server to install or manage before a demo. The schema
(riders, trips, sensor_readings, ride_events, alerts, ride_journal) is designed so it
could be swapped for PostgreSQL with minimal changes if scaled up.

**Q: What is the role of machine learning here, specifically?**
It classifies the *overall pattern* in a feature window as SAFE or AGGRESSIVE with a
confidence score — generalizing across combinations of features rather than requiring a
hand-written rule for every possible risky combination. The rule layer catches the
clear-cut, individually-explainable cases (a single threshold crossed); the ML layer
provides a second, probabilistic opinion that can, in principle, catch patterns the rules
don't explicitly enumerate.

**Q: What is the difference between rule-based detection and ML classification here?**
A rule is a fixed, human-written threshold on one or a few features (e.g. "speed > 80")
— fully transparent, but has to be manually authored for every case. The ML classifier
looks at *all ten features together* and has learned the decision boundary from data — it
can pick up on combinations that would be tedious to hand-encode as rules, at the cost of
being a little less directly inspectable than a single threshold. MotoPulse uses both
deliberately: rules for guaranteed, explainable, demo-reliable detection, ML for the
probabilistic corroboration a real product would lean on more heavily as more real data
becomes available.
