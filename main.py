import os
import io
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from typing import Optional

from models import (
    FaceRegisterResponse,
    SelfieAttendanceResponse,
    ClassPhotoResponse,
    HealthResponse,
)
from face_service import (
    generate_face_encoding,
    upload_face_to_cloudinary,
    verify_single_face,
    recognize_faces_in_class_photo,
)
from geo_service import validate_student_location, is_valid_coordinate
from database import (
    get_db,
    close_db,
    get_student_by_id,
    save_face_encoding,
    get_all_students_with_encodings,
    get_active_session,
    has_student_submitted,
    mark_student_submitted,
)

load_dotenv()

INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "")
FACE_MODEL = os.getenv("FACE_MODEL", "ArcFace")
FACE_DETECTOR = os.getenv("FACE_DETECTOR", "retinaface")


# ── LIFESPAN ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: connect to DB
    get_db()
    print("✅ NexFord Face Recognition Service started")
    yield
    # Shutdown: close DB
    close_db()
    print("🛑 Service shutting down")


# ── APP SETUP ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NexFord Face Recognition Service",
    description="AI-powered facial recognition microservice for NexFord Edu",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── AUTH GUARD ─────────────────────────────────────────────────────────────────
def verify_api_key(x_api_key: str = Header(...)):
    """
    Simple API key check so only the Node.js backend
    can call this service — not the public internet.
    """
    if INTERNAL_API_KEY and x_api_key != INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key.")
    return x_api_key


# ── HEALTH CHECK ───────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "success": True,
        "message": "NexFord Face Recognition Service is running.",
        "face_model": FACE_MODEL,
        "detector": FACE_DETECTOR,
    }


# ── REGISTER STUDENT FACE ──────────────────────────────────────────────────────
@app.post("/register", response_model=FaceRegisterResponse)
async def register_face(
    student_id: str = Form(...),
    school_id: str = Form(...),
    image: UploadFile = File(...),
    _: str = Depends(verify_api_key),
):
    """
    Register a student's face encoding in the database.
    Called when a student uploads their profile face photo.

    Steps:
    1. Read uploaded image bytes
    2. Validate student exists in DB
    3. Generate face encoding using DeepFace
    4. Upload image to Cloudinary
    5. Save encoding + image URL to student's MongoDB document
    """
    # Read image bytes
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file uploaded.")

    # Verify student exists
    student = get_student_by_id(student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found.")

    # Generate face encoding
    try:
        encoding = generate_face_encoding(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Encoding error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Face encoding failed. Please try again with a clearer image."
        )

    # Upload image to Cloudinary
    try:
        image_url = upload_face_to_cloudinary(image_bytes, student_id)
    except Exception as e:
        print(f"Cloudinary upload error: {e}")
        image_url = ""

    # Save to MongoDB
    saved = save_face_encoding(student_id, encoding, image_url)
    if not saved:
        raise HTTPException(status_code=500, detail="Failed to save face data.")

    return {
        "success": True,
        "message": "Face registered successfully.",
        "student_id": student_id,
        "face_encoding": encoding,
        "image_url": image_url,
    }


# ── STUDENT SELFIE ATTENDANCE ──────────────────────────────────────────────────
@app.post("/recognize/selfie", response_model=SelfieAttendanceResponse)
async def selfie_attendance(
    student_id: str = Form(...),
    session_id: str = Form(...),
    student_latitude: float = Form(...),
    student_longitude: float = Form(...),
    image: UploadFile = File(...),
    _: str = Depends(verify_api_key),
):
    """
    Student takes a selfie to mark their own attendance.
    Two checks must pass:
    1. Geofence — student must be within the allowed distance of lecturer
    2. Face match — selfie must match student's registered face encoding

    Both must pass for attendance to be marked.
    """

    # ── Step 1: Validate GPS coordinates ──────────────────────────────────────
    if not is_valid_coordinate(student_latitude, student_longitude):
        raise HTTPException(status_code=400, detail="Invalid GPS coordinates provided.")

    # ── Step 2: Get active session ─────────────────────────────────────────────
    session = get_active_session(session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Attendance session not found, expired, or already closed."
        )

    school_id = str(session.get("school", ""))
    lecturer_lat = session.get("lecturerLatitude")
    lecturer_lng = session.get("lecturerLongitude")

    if lecturer_lat is None or lecturer_lng is None:
        raise HTTPException(
            status_code=400,
            detail="Lecturer location not recorded for this session."
        )

    # ── Step 3: Check if student already submitted ─────────────────────────────
    if has_student_submitted(session_id, student_id):
        raise HTTPException(
            status_code=409,
            detail="You have already submitted attendance for this session."
        )

    # ── Step 4: Geofence check ─────────────────────────────────────────────────
    geo_result = validate_student_location(
        lecturer_lat=float(lecturer_lat),
        lecturer_lng=float(lecturer_lng),
        student_lat=student_latitude,
        student_lng=student_longitude,
        school_id=school_id,
    )

    location_valid = geo_result["is_valid"]
    distance_metres = geo_result["distance_metres"]

    if not location_valid:
        return {
            "success": False,
            "message": (
                f"You are {distance_metres:.0f} metres from the class location. "
                f"Maximum allowed is {geo_result['max_allowed_metres']:.0f} metres. "
                "Please ensure you are physically present in the classroom."
            ),
            "student_id": student_id,
            "face_matched": False,
            "location_valid": False,
            "distance_metres": distance_metres,
            "marked_present": False,
        }

    # ── Step 5: Get student's registered face encoding ─────────────────────────
    student = get_student_by_id(student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found.")

    stored_encoding = student.get("faceEncoding")
    if not stored_encoding:
        raise HTTPException(
            status_code=400,
            detail=(
                "Your face has not been registered yet. "
                "Please go to your profile and register your face first."
            )
        )

    # ── Step 6: Read selfie and verify face ────────────────────────────────────
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file.")

    try:
        face_result = verify_single_face(image_bytes, stored_encoding)
    except Exception as e:
        print(f"Face verification error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Face verification failed. Please try again."
        )

    face_matched = face_result.get("matched", False)
    error = face_result.get("error")

    if error:
        return {
            "success": False,
            "message": error,
            "student_id": student_id,
            "face_matched": False,
            "location_valid": True,
            "distance_metres": distance_metres,
            "marked_present": False,
        }

    if not face_matched:
        return {
            "success": False,
            "message": (
                "Face verification failed. Your selfie did not match your "
                "registered face. Please ensure your face is clearly visible "
                "and try again."
            ),
            "student_id": student_id,
            "face_matched": False,
            "location_valid": True,
            "distance_metres": distance_metres,
            "marked_present": False,
        }

    # ── Step 7: Both checks passed — record submission ─────────────────────────
    mark_student_submitted(session_id, student_id)

    return {
        "success": True,
        "message": "Attendance marked successfully. You have been recorded as present.",
        "student_id": student_id,
        "face_matched": True,
        "location_valid": True,
        "distance_metres": distance_metres,
        "marked_present": True,
    }


# ── LECTURER CLASS PHOTO ATTENDANCE ───────────────────────────────────────────
@app.post("/recognize/class", response_model=ClassPhotoResponse)
async def class_photo_attendance(
    school_id: str = Form(...),
    image: UploadFile = File(...),
    _: str = Depends(verify_api_key),
):
    """
    Lecturer takes a group photo of the class.
    All faces in the photo are detected and matched against
    registered student encodings for this school.

    No GPS check here — the lecturer is trusted.
    """

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file.")

    # Get all students with registered face encodings for this school
    students = get_all_students_with_encodings(school_id)

    if not students:
        raise HTTPException(
            status_code=404,
            detail=(
                "No students with registered face encodings found for this school. "
                "Students must register their faces before attendance can be taken."
            )
        )

    # Run face recognition on the class photo
    try:
        recognized_ids, total_detected, unrecognized = recognize_faces_in_class_photo(
            image_bytes, students
        )
    except Exception as e:
        print(f"Class recognition error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Face recognition failed. Please try again with a clearer photo."
        )

    if total_detected == 0:
        return {
            "success": False,
            "message": "No faces were detected in the photo. Please take a clearer photo of the class.",
            "total_faces_detected": 0,
            "recognized_students": [],
            "unrecognized_count": 0,
        }

    return {
        "success": True,
        "message": (
            f"{len(recognized_ids)} of {total_detected} detected "
            f"face(s) were recognized."
        ),
        "total_faces_detected": total_detected,
        "recognized_students": recognized_ids,
        "unrecognized_count": unrecognized,
    }
