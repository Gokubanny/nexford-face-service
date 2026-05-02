from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class FaceRegisterResponse(BaseModel):
    success: bool
    message: str
    student_id: str
    face_encoding: Optional[List[float]] = None
    image_url: Optional[str] = None


class SelfieAttendanceRequest(BaseModel):
    student_id: str
    session_id: str
    student_latitude: float
    student_longitude: float


class SelfieAttendanceResponse(BaseModel):
    success: bool
    message: str
    student_id: str
    face_matched: bool
    location_valid: bool
    distance_metres: Optional[float] = None
    marked_present: bool


class ClassPhotoResponse(BaseModel):
    success: bool
    message: str
    total_faces_detected: int
    recognized_students: List[str]
    unrecognized_count: int


class HealthResponse(BaseModel):
    success: bool
    message: str
    face_model: str
    detector: str
