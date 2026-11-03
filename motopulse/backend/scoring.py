"""
MotoPulse — Safe Rider Score engine.

The score starts at 100 for every trip and moves according to explainable
rules (see config.SCORE_DELTAS). Every change is logged with a reason so the
dashboard can always answer "why did the score change?".
"""
import config


def score_band(score: float) -> str:
    for lo, hi, label in config.SCORE_BANDS:
        if lo <= score <= hi:
            return label
    return "Risky"


class ScoreEngine:
    """One instance per active trip."""

    def __init__(self):
        self.score = float(config.STARTING_SCORE)
        self.contributions = {k: 0.0 for k in config.SCORE_DELTAS}  # cumulative deduction per factor
        self.safe_ticks_streak = 0
        self.history = []  # list of (timestamp, score)

    def apply_event(self, event_type: str, ts: float):
        delta = config.SCORE_DELTAS.get(event_type, 0)
        self.score = max(config.SCORE_MIN, min(config.SCORE_MAX, self.score + delta))
        if event_type in self.contributions:
            self.contributions[event_type] += abs(delta)
        self.safe_ticks_streak = 0
        self.history.append((ts, self.score))
        return delta, self.score

    def apply_safe_tick(self, ts: float):
        self.safe_ticks_streak += 1
        if self.score < config.SCORE_MAX:
            self.score = min(config.SCORE_MAX, self.score + config.SCORE_RECOVERY_PER_SAFE_TICK)
        self.history.append((ts, self.score))
        return self.score

    def breakdown(self):
        total = sum(self.contributions.values())
        factors = []
        for k, v in self.contributions.items():
            if v > 0:
                pct = round(100 * v / total, 1) if total > 0 else 0
                factors.append({"factor": k, "deduction": round(v, 1), "percent": pct})
        factors.sort(key=lambda f: -f["deduction"])
        return {
            "score": round(self.score, 1),
            "band": score_band(self.score),
            "risk_factor_breakdown": factors,
            "positive_factors": self._positive_factors(),
        }

    def _positive_factors(self):
        positives = []
        if self.safe_ticks_streak >= 2:
            positives.append("Stable speed")
        if self.contributions.get("HARSH_BRAKING", 0) == 0:
            positives.append("Smooth braking")
        if self.contributions.get("AGGRESSIVE_TURN", 0) == 0:
            positives.append("Controlled turning")
        if self.contributions.get("SUDDEN_ACCELERATION", 0) == 0:
            positives.append("Smooth acceleration")
        return positives
