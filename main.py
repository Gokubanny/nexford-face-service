import os
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
    FACE_MODEL,
    FACE_DETECTOR,
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_db()
    print(f"✅ NexFord Face Recognition Service started")
    print(f"   Model    : {FACE_MODEL}")
    print(f"   Detector : {FACE_DETECTOR}")
    yield
    close_db()
    print("🛑 Service shutting down")


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


def verify_api_key(x_api_key: str = Header(...)):
    if INTERNAL_API_KEY and x_api_key != INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key.")
    return x_api_key


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "success": True,
        "message": "NexFord Face Recognition Service is running.",
        "face_model": FACE_MODEL,
        "detector": FACE_DETECTOR,
    }


@app.post("/register", response_model=FaceRegisterResponse)
async def register_face(
    student_id: str = Form(...),
    school_id: str = Form(...),
    image: UploadFile = File(...),
    _: str = Depends(verify_api_key),
):
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file uploaded.")

    student = get_student_by_id(student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found.")

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

    try:
        image_url = upload_face_to_cloudinary(image_bytes, student_id)
    except Exception as e:
        print(f"Cloudinary upload error: {e}")
        image_url = ""

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


@app.post("/recognize/selfie", response_model=SelfieAttendanceResponse)
async def selfie_attendance(
    student_id: str = Form(...),
    session_id: str = Form(...),
    student_latitude: float = Form(...),
    student_longitude: float = Form(...),
    image: UploadFile = File(...),
    _: str = Depends(verify_api_key),
):
    if not is_valid_coordinate(student_latitude, student_longitude):
        raise HTTPException(status_code=400, detail="Invalid GPS coordinates provided.")

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

    if has_student_submitted(session_id, student_id):
        raise HTTPException(
            status_code=409,
            detail="You have already submitted attendance for this session."
        )

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


@app.post("/recognize/class", response_model=ClassPhotoResponse)
async def class_photo_attendance(
    school_id: str = Form(...),
    image: UploadFile = File(...),
    _: str = Depends(verify_api_key),
):
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file.")

    students = get_all_students_with_encodings(school_id)

    if not students:
        raise HTTPException(
            status_code=404,
            detail=(
                "No students with registered face encodings found for this school. "
                "Students must register their faces before attendance can be taken."
            )
        )

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
            "message": "No faces were detected in the photo. Please take a clearer photo.",
            "total_faces_detected": 0,
            "recognized_students": [],
            "unrecognized_count": 0,
        }

    return {
        "success": True,
        "message": (
            f"{len(recognized_ids)} of {total_detected} detected face(s) were recognized."
        ),
        "total_faces_detected": total_detected,
        "recognized_students": recognized_ids,
        "unrecognized_count": unrecognized,
    }