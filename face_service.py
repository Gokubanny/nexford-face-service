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
FACE_MODEL = os.getenv("FACE_MODEL", "ArcFace")
FACE_DETECTOR = os.getenv("FACE_DETECTOR", "retinaface")
DISTANCE_METRIC = os.getenv("FACE_DISTANCE_METRIC", "cosine")
THRESHOLD = float(os.getenv("FACE_THRESHOLD", "0.68"))


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

    Raises:
        ValueError: If no face is detected or multiple faces are found
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
            raise ValueError("No face detected in the image. Please ensure your face is clearly visible.")

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
    Used for student self-attendance.

    Returns:
        dict with keys: matched (bool), distance (float), threshold (float)
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
            return {"matched": False, "distance": None, "error": "No face detected in selfie."}

        if len(selfie_embeddings) > 1:
            return {"matched": False, "distance": None, "error": "Multiple faces detected in selfie."}

        selfie_vector = np.array(selfie_embeddings[0]["embedding"])
        stored_vector = np.array(stored_encoding)

        # Calculate cosine distance (lower = more similar)
        if DISTANCE_METRIC == "cosine":
            dot = np.dot(selfie_vector, stored_vector)
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
            "matched": matched,
            "distance": round(float(distance), 4),
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
        image_bytes: Raw bytes of the classroom photo
        students_with_encodings: List of student dicts from MongoDB,
                                  each with _id and faceEncoding fields

    Returns:
        Tuple of:
        - recognized_student_ids (List[str])
        - total_faces_detected (int)
        - unrecognized_count (int)
    """
    tmp_path = save_temp_image(image_bytes)
    recognized_ids: List[str] = []

    try:
        # Step 1: Detect all faces in the classroom photo
        img = cv2.imread(tmp_path)
        if img is None:
            raise ValueError("Could not read classroom image.")

        # Extract all face embeddings from the classroom image
        try:
            class_embeddings = DeepFace.represent(
                img_path=tmp_path,
                model_name=FACE_MODEL,
                detector_backend=FACE_DETECTOR,
                enforce_detection=False,  # Don't fail if some faces unclear
                align=True
            )
        except Exception as e:
            print(f"Face detection error: {e}")
            return [], 0, 0

        if not class_embeddings:
            return [], 0, 0

        total_detected = len(class_embeddings)
        unrecognized = 0

        # Step 2: For each detected face, find the best matching student
        for face_data in class_embeddings:
            face_vector = np.array(face_data["embedding"])
            best_match_id: Optional[str] = None
            best_distance = float("inf")

            # Compare this face against every registered student
            for student in students_with_encodings:
                stored_encoding = student.get("faceEncoding")
                if not stored_encoding:
                    continue

                stored_vector = np.array(stored_encoding)

                # Calculate cosine distance
                dot = np.dot(face_vector, stored_vector)
                norm_a = np.linalg.norm(face_vector)
                norm_b = np.linalg.norm(stored_vector)

                if norm_a == 0 or norm_b == 0:
                    distance = 1.0
                else:
                    distance = 1 - (dot / (norm_a * norm_b))

                if distance < best_distance:
                    best_distance = distance
                    best_match_id = str(student["_id"])

            # Only count as recognized if distance is below threshold
            if best_match_id and best_distance <= THRESHOLD:
                if best_match_id not in recognized_ids:
                    recognized_ids.append(best_match_id)
            else:
                unrecognized += 1

        return recognized_ids, total_detected, unrecognized

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
