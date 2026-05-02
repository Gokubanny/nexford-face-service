import os
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from bson import ObjectId
from typing import Optional, Dict, Any, List
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

_client: Optional[MongoClient] = None
_db: Optional[Database] = None


def get_db() -> Database:
    """Get database connection (singleton pattern)."""
    global _client, _db

    if _db is None:
        uri = os.getenv("MONGODB_URI")
        if not uri:
            raise ValueError("MONGODB_URI not set in environment variables")

        _client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        _db = _client.get_default_database()
        print("✅ Connected to MongoDB")

    return _db


def close_db():
    """Close the database connection."""
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db = None


# ── STUDENT QUERIES ────────────────────────────────────────────────────────────

def get_student_by_id(student_id: str) -> Optional[Dict]:
    """Get a student document by their MongoDB ObjectId."""
    db = get_db()
    try:
        return db.students.find_one({"_id": ObjectId(student_id)})
    except Exception:
        return None


def get_student_by_user(user_id: str) -> Optional[Dict]:
    """Get a student document by their user ObjectId."""
    db = get_db()
    try:
        return db.students.find_one({"user": ObjectId(user_id)})
    except Exception:
        return None


def save_face_encoding(student_id: str, encoding: List[float], image_url: str) -> bool:
    """Save a student's face encoding and image URL to MongoDB."""
    db = get_db()
    try:
        result = db.students.update_one(
            {"_id": ObjectId(student_id)},
            {"$set": {
                "faceEncoding": encoding,
                "faceImageUrl": image_url,
                "updatedAt": datetime.utcnow()
            }}
        )
        return result.modified_count > 0
    except Exception as e:
        print(f"Error saving face encoding: {e}")
        return False


def get_all_students_with_encodings(school_id: str) -> List[Dict]:
    """
    Get all students from a school who have registered face encodings.
    Used during class photo recognition to compare against.
    """
    db = get_db()
    try:
        students = db.students.find(
            {
                "school": ObjectId(school_id),
                "faceEncoding": {"$exists": True, "$ne": [], "$ne": None},
                "approvalStatus": "approved"
            },
            {
                "_id": 1,
                "matricNumber": 1,
                "faceEncoding": 1,
                "faceImageUrl": 1,
                "user": 1
            }
        )
        return list(students)
    except Exception as e:
        print(f"Error fetching students with encodings: {e}")
        return []


# ── ATTENDANCE SESSION QUERIES ─────────────────────────────────────────────────

def get_active_session(session_id: str) -> Optional[Dict]:
    """Get an attendance session that is currently open."""
    db = get_db()
    try:
        session = db.attendancesessions.find_one({
            "_id": ObjectId(session_id),
            "status": "open",
            "expiresAt": {"$gt": datetime.utcnow()}
        })
        return session
    except Exception as e:
        print(f"Error fetching session: {e}")
        return None


def mark_student_submitted(session_id: str, student_id: str) -> bool:
    """
    Record that a student has already submitted a selfie for this session.
    Prevents duplicate submissions.
    """
    db = get_db()
    try:
        result = db.attendancesessions.update_one(
            {"_id": ObjectId(session_id)},
            {"$addToSet": {"submittedStudents": ObjectId(student_id)}}
        )
        return result.modified_count > 0
    except Exception as e:
        print(f"Error recording submission: {e}")
        return False


def has_student_submitted(session_id: str, student_id: str) -> bool:
    """Check if a student has already submitted a selfie for this session."""
    db = get_db()
    try:
        session = db.attendancesessions.find_one({
            "_id": ObjectId(session_id),
            "submittedStudents": ObjectId(student_id)
        })
        return session is not None
    except Exception:
        return False


# ── SCHOOL QUERIES ─────────────────────────────────────────────────────────────

def get_school_settings(school_id: str) -> Optional[Dict]:
    """
    Get school document including subscription plan and geofence settings.
    Used to determine max allowed distance for attendance.
    """
    db = get_db()
    try:
        return db.schools.find_one(
            {"_id": ObjectId(school_id)},
            {
                "subscription.plan": 1,
                "attendanceSettings.maxDistanceMetres": 1,
                "name": 1
            }
        )
    except Exception as e:
        print(f"Error fetching school settings: {e}")
        return None
