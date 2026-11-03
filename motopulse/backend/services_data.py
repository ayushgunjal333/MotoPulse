"""
MotoPulse — Nearby Services demo data.

No paid map/places API key is required for the demo. This module returns
realistic, hand-placed points of interest around the simulated ride route
(Bangalore). Each function returns coordinates as small offsets from the
rider's current location so results always look "nearby" during the demo.

To connect a real Places API later: replace the body of `get_nearby()` with
a call to that API and keep the same return shape.
"""
import random

CATEGORY_META = {
    "fuel": {"label": "Fuel Stations", "icon": "fuel"},
    "hospital": {"label": "Hospitals", "icon": "hospital"},
    "service": {"label": "Service Centers", "icon": "service"},
    "food": {"label": "Food Stops", "icon": "food"},
}

_NAMES = {
    "fuel": [
        "Indian Oil Fuel Station", "Bharat Petroleum Pump", "HP Petrol Bunk",
        "Shell Fuel Station", "Reliance Petrol Pump",
    ],
    "hospital": [
        "City General Hospital", "St. Martha's Hospital", "Apollo Clinic",
        "Fortis Care Centre", "Manipal Hospital Annex",
    ],
    "service": [
        "Royal Enfield Service Center", "TVS Authorised Workshop",
        "Bike Doctor Multi-Brand Service", "Quick Fix Two-Wheeler Garage",
        "Highway Motorcycle Service Point",
    ],
    "food": [
        "Adyar Ananda Bhavan", "Highway Dhaba", "CTR Darshini",
        "Cafe Coffee Day", "Empire Restaurant",
    ],
}


def get_nearby(category: str, lat: float, lon: float, count: int = 5):
    if category not in CATEGORY_META:
        return []
    rng = random.Random(f"{category}-{round(lat, 3)}-{round(lon, 3)}")
    results = []
    names = _NAMES[category]
    for i in range(min(count, len(names))):
        d_lat = rng.uniform(-0.02, 0.02)
        d_lon = rng.uniform(-0.02, 0.02)
        distance_km = round(((d_lat ** 2 + d_lon ** 2) ** 0.5) * 111, 1)
        results.append({
            "name": names[i],
            "category": category,
            "lat": round(lat + d_lat, 6),
            "lon": round(lon + d_lon, 6),
            "distance_km": distance_km,
            "eta_min": max(1, round(distance_km / 0.5)),
            "rating": round(rng.uniform(3.6, 4.8), 1),
        })
    results.sort(key=lambda r: r["distance_km"])
    return results
