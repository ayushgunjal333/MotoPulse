"""
MotoPulse — demo ride route.

Generates a smooth, realistic-looking path (no external routing API needed)
starting near Bangalore Institute of Technology, V.V. Puram and winding
generally north-east -- the same direction as the Nandi Hills ride mentioned
in the project's Ride Journal example. The simulator maps "distance
travelled" onto this path so the map marker moves continuously.
"""
import math
import random

START_LAT, START_LON = 12.9432, 77.5722   # BIT College, V.V. Puram, Bangalore

_rng = random.Random(7)


def _generate_path(n_points=1400, step_km=0.015):
    """Random-walk path with heading momentum so turns look like real roads,
    not noise. Returns list of (lat, lon)."""
    lat, lon = START_LAT, START_LON
    heading = math.radians(48)  # roughly north-east, toward Nandi Hills
    path = [(lat, lon)]
    for _ in range(n_points):
        heading += math.radians(_rng.uniform(-9, 9))
        dlat = (step_km / 111.0) * math.cos(heading)
        dlon = (step_km / (111.0 * math.cos(math.radians(lat)))) * math.sin(heading)
        lat += dlat
        lon += dlon
        path.append((lat, lon))
    return path


ROUTE = _generate_path()

_cum_dist = [0.0]
for i in range(1, len(ROUTE)):
    lat1, lon1 = ROUTE[i - 1]
    lat2, lon2 = ROUTE[i]
    d = math.hypot((lat2 - lat1) * 111.0,
                    (lon2 - lon1) * 111.0 * math.cos(math.radians(lat1)))
    _cum_dist.append(_cum_dist[-1] + d)

ROUTE_TOTAL_KM = _cum_dist[-1]


def position_at_distance(distance_km: float):
    """Interpolate a lat/lon along ROUTE for a given cumulative distance.
    Clamps at the end of the precomputed path (~21 km, well beyond any demo
    ride length)."""
    if distance_km <= 0:
        return ROUTE[0]
    if distance_km >= ROUTE_TOTAL_KM:
        return ROUTE[-1]
    # binary search
    lo, hi = 0, len(_cum_dist) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if _cum_dist[mid] < distance_km:
            lo = mid + 1
        else:
            hi = mid
    i = max(1, lo)
    d0, d1 = _cum_dist[i - 1], _cum_dist[i]
    lat0, lon0 = ROUTE[i - 1]
    lat1, lon1 = ROUTE[i]
    t = 0 if d1 == d0 else (distance_km - d0) / (d1 - d0)
    return (lat0 + (lat1 - lat0) * t, lon0 + (lon1 - lon0) * t)
