import os
from geopy.distance import geodesic
from typing import Tuple, Dict, Optional
from database import get_school_settings

DEFAULT_MAX_DISTANCE = float(os.getenv("DEFAULT_MAX_DISTANCE_METRES", "100"))

# Subscription plans that support custom geofence distance
CONFIGURABLE_PLANS = {"pro", "enterprise"}


def calculate_distance(
    point_a: Tuple[float, float],
    point_b: Tuple[float, float]
) -> float:
    """
    Calculate distance in metres between two GPS coordinates.

    Args:
        point_a: (latitude, longitude) tuple
        point_b: (latitude, longitude) tuple

    Returns:
        Distance in metres (float)
    """
    return geodesic(point_a, point_b).meters


def get_max_distance_for_school(school_id: str) -> float:
    """
    Determine the maximum allowed attendance distance for a school
    based on their subscription plan and custom settings.

    Rules:
    - Basic plan: always 100 metres (not configurable)
    - Pro/Enterprise: use school's custom setting if set,
                      otherwise fall back to 100 metres
    """
    school = get_school_settings(school_id)

    if not school:
        # School not found — use safe default
        return DEFAULT_MAX_DISTANCE

    plan = school.get("subscription", {}).get("plan", "basic")

    if plan in CONFIGURABLE_PLANS:
        # Check if school has a custom distance configured
        custom_distance = (
            school
            .get("attendanceSettings", {})
            .get("maxDistanceMetres")
        )
        if custom_distance and isinstance(custom_distance, (int, float)):
            # Clamp between 50 and 500 metres for safety
            return max(50.0, min(500.0, float(custom_distance)))

    # Basic plan or no custom setting — use default 100 metres
    return DEFAULT_MAX_DISTANCE


def validate_student_location(
    lecturer_lat: float,
    lecturer_lng: float,
    student_lat: float,
    student_lng: float,
    school_id: str
) -> Dict:
    """
    Check whether a student is close enough to the lecturer
    for their attendance to be considered valid.

    Returns a dict with:
    - is_valid (bool): whether student is within allowed range
    - distance_metres (float): actual distance calculated
    - max_allowed_metres (float): the threshold used
    - plan (str): the school's subscription plan
    """
    lecturer_coords = (lecturer_lat, lecturer_lng)
    student_coords = (student_lat, student_lng)

    distance = calculate_distance(lecturer_coords, student_coords)
    max_distance = get_max_distance_for_school(school_id)

    is_valid = distance <= max_distance

    return {
        "is_valid": is_valid,
        "distance_metres": round(distance, 2),
        "max_allowed_metres": max_distance,
    }


def is_valid_coordinate(lat: float, lng: float) -> bool:
    """Sanity check that coordinates are within valid GPS range."""
    return -90 <= lat <= 90 and -180 <= lng <= 180
