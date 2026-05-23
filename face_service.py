import os
import cv2
import numpy as np
import tempfile
import cloudinary
import cloudinary.uploader
from deepface import DeepFace
from typing import List, Optional, Dict, Tuple
from dotenv import load_dotenv

load_dotenv()

# ── CLOUDINARY SETUP ───────────────────────────────────────────────────────────
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# ── CONFIG ─────────────────────────────────────────────────────────────────────
# Defaults are tuned for Render free plan (512 MB RAM):
#   Facenet  — 128-dim embeddings, ~22 MB model file
#   opencv   — Haar cascade detector, no neural net, near-zero RAM overhead
#   0.40     — cosine distance threshold for Facenet (stricter than ArcFace's 0.68)
#
# To upgrade to higher accuracy on Render Starter ($7/mo, 1 GB RAM):
#   FACE_MODEL=ArcFace  FACE_DETECTOR=retinaface  FACE_THRESHOLD=0.68
FACE_MODEL      = os.getenv("FACE_MODEL",           "Facenet")
FACE_DETECTOR   = os.getenv("FACE_DETECTOR",         "opencv")
DISTANCE_METRIC = os.getenv("FACE_DISTANCE_METRIC",  "cosine")
THRESHOLD       = float(os.getenv("FACE_THRESHOLD",  "0.40"))


def read_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Convert raw bytes to an OpenCV image array."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Please upload a valid image file.")
    return img


def save_temp_image(image_bytes: bytes, suffix: str = ".jpg") -> str:
    """Save image bytes to a temporary file and return the path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(image_bytes)
    tmp.close()
    return tmp.name


def upload_face_to_cloudinary(image_bytes: bytes, student_id: str) -> str:
    """
    Upload a student's face image to Cloudinary.
    Returns the secure URL of the uploaded image.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        result = cloudinary.uploader.upload(
            tmp_path,
            folder=f"nexford/faces/{student_id}",
            public_id=f"face_{student_id}",
            overwrite=True,
            resource_type="image",
            transformation=[
                {"width": 400, "height": 400, "crop": "fill", "gravity": "face"}
            ]
        )
        return result["secure_url"]
    finally:
        os.unlink(tmp_path)


def generate_face_encoding(image_bytes: bytes) -> List[float]:
    """
    Generate a face encoding (embedding vector) from an image.
    This is the mathematical representation of a face stored in MongoDB.

    With Facenet the returned vector is 128-dimensional.
    With ArcFace  the returned vector is 512-dimensional.
    Vectors from different models are NOT interchangeable — if you switch
    models, wipe existing faceEncoding fields and have students re-register.

    Raises:
        ValueError: If no face is detected or multiple faces are found.
    """
    tmp_path = save_temp_image(image_bytes)

    try:
        # DeepFace.represent returns a list of detected face embeddings
        embeddings = DeepFace.represent(
            img_path=tmp_path,
            model_name=FACE_MODEL,
            detector_backend=FACE_DETECTOR,
            enforce_detection=True,
            align=True
        )

        if not embeddings:
            raise ValueError(
                "No face detected in the image. "
                "Please ensure your face is clearly visible and well-lit."
            )

        if len(embeddings) > 1:
            raise ValueError(
                f"Multiple faces detected ({len(embeddings)}). "
                "Please upload a photo with only your face."
            )

        return embeddings[0]["embedding"]

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def verify_single_face(
    selfie_bytes: bytes,
    stored_encoding: List[float]
) -> Dict:
    """
    Compare a selfie against a stored face encoding.
    Used for student self-attendance (selfie flow).

    Returns:
        dict with keys: matched (bool), distance (float), threshold (float)

    Important: the stored_encoding dimension must match the current FACE_MODEL.
    Facenet  → 128 floats   (threshold ~0.40)
    ArcFace  → 512 floats   (threshold ~0.68)
    If dimensions mismatch the numpy dot product raises a shape error which
    the caller catches and surfaces as a 500.
    """
    tmp_path = save_temp_image(selfie_bytes)

    try:
        # Generate encoding for the incoming selfie
        selfie_embeddings = DeepFace.represent(
            img_path=tmp_path,
            model_name=FACE_MODEL,
            detector_backend=FACE_DETECTOR,
            enforce_detection=True,
            align=True
        )

        if not selfie_embeddings:
            return {
                "matched": False,
                "distance": None,
                "error": "No face detected in selfie. Please face the camera directly."
            }

        if len(selfie_embeddings) > 1:
            return {
                "matched": False,
                "distance": None,
                "error": "Multiple faces detected in selfie. Only one face should be visible."
            }

        selfie_vector = np.array(selfie_embeddings[0]["embedding"])
        stored_vector = np.array(stored_encoding)

        # Guard against dimension mismatch (old ArcFace encoding vs new Facenet model)
        if selfie_vector.shape != stored_vector.shape:
            return {
                "matched": False,
                "distance": None,
                "error": (
                    "Your stored face data is incompatible with the current model. "
                    "Please go to your Profile and re-register your face."
                )
            }

        # Cosine distance — lower means more similar (0 = identical, 1 = opposite)
        # Facenet + cosine threshold: 0.40
        # ArcFace + cosine threshold: 0.68
        if DISTANCE_METRIC == "cosine":
            dot    = np.dot(selfie_vector, stored_vector)
            norm_a = np.linalg.norm(selfie_vector)
            norm_b = np.linalg.norm(stored_vector)
            if norm_a == 0 or norm_b == 0:
                distance = 1.0
            else:
                distance = 1 - (dot / (norm_a * norm_b))
        else:
            # Euclidean distance
            distance = float(np.linalg.norm(selfie_vector - stored_vector))

        matched = distance <= THRESHOLD

        return {
            "matched":   matched,
            "distance":  round(float(distance), 4),
            "threshold": THRESHOLD
        }

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def recognize_faces_in_class_photo(
    image_bytes: bytes,
    students_with_encodings: List[Dict]
) -> Tuple[List[str], int, int]:
    """
    Recognize multiple student faces in a single classroom photo.
    Used for the lecturer group-photo attendance method.

    Args:
        image_bytes: Raw bytes of the classroom photo.
        students_with_encodings: List of student dicts from MongoDB,
                                  each with _id and faceEncoding fields.

    Returns:
        Tuple of:
        - recognized_student_ids (List[str])
        - total_faces_detected   (int)
        - unrecognized_count     (int)
    """
    tmp_path = save_temp_image(image_bytes)
    recognized_ids: List[str] = []

    try:
        img = cv2.imread(tmp_path)
        if img is None:
            raise ValueError("Could not read classroom image.")

        # Extract all face embeddings from the classroom image
        try:
            class_embeddings = DeepFace.represent(
                img_path=tmp_path,
                model_name=FACE_MODEL,
                detector_backend=FACE_DETECTOR,
                enforce_detection=False,   # Don't fail if some faces are unclear
                align=True
            )
        except Exception as e:
            print(f"Face detection error in class photo: {e}")
            return [], 0, 0

        if not class_embeddings:
            return [], 0, 0

        total_detected = len(class_embeddings)
        unrecognized   = 0

        # For each detected face find the closest matching registered student
        for face_data in class_embeddings:
            face_vector     = np.array(face_data["embedding"])
            best_match_id: Optional[str] = None
            best_distance   = float("inf")

            for student in students_with_encodings:
                stored_encoding = student.get("faceEncoding")
                if not stored_encoding:
                    continue

                stored_vector = np.array(stored_encoding)

                # Skip students whose encoding dimension doesn't match
                # (leftover ArcFace encodings when running Facenet, or vice-versa)
                if face_vector.shape != stored_vector.shape:
                    continue

                # Cosine distance
                dot    = np.dot(face_vector, stored_vector)
                norm_a = np.linalg.norm(face_vector)
                norm_b = np.linalg.norm(stored_vector)

                if norm_a == 0 or norm_b == 0:
                    distance = 1.0
                else:
                    distance = 1 - (dot / (norm_a * norm_b))

                if distance < best_distance:
                    best_distance = distance
                    best_match_id = str(student["_id"])

            if best_match_id and best_distance <= THRESHOLD:
                if best_match_id not in recognized_ids:
                    recognized_ids.append(best_match_id)
            else:
                unrecognized += 1

        return recognized_ids, total_detected, unrecognized

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)