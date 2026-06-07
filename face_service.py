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

# ── CLOUDINARY ────────────────────────────────────────────────────────────────
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
FACE_MODEL      = os.getenv("FACE_MODEL",           "Facenet")
FACE_DETECTOR   = os.getenv("FACE_DETECTOR",         "ssd")        # better than opencv
DISTANCE_METRIC = os.getenv("FACE_DISTANCE_METRIC",  "cosine")
THRESHOLD       = float(os.getenv("FACE_THRESHOLD",  "0.22"))      # strict for attendance


def save_temp_image(image_bytes: bytes, suffix: str = ".jpg") -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(image_bytes)
    tmp.close()
    return tmp.name


def upload_face_to_cloudinary(image_bytes: bytes, student_id: str) -> str:
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
    Generate face embedding for registration.
    STRICT — rejects image if no clear face is found.
    This prevents storing garbage embeddings that would later match anything.
    """
    tmp_path = save_temp_image(image_bytes)
    try:
        # enforce_detection=True during registration:
        # if no face found we raise an error rather than storing a useless embedding
        embeddings = DeepFace.represent(
            img_path=tmp_path,
            model_name=FACE_MODEL,
            detector_backend=FACE_DETECTOR,
            enforce_detection=True,
            align=True
        )

        if not embeddings:
            raise ValueError(
                "No face detected. Please ensure your face is clearly visible "
                "and well-lit, looking directly at the camera."
            )

        if len(embeddings) > 1:
            raise ValueError(
                f"{len(embeddings)} faces detected. Please take a photo "
                "with only your face visible."
            )

        embedding = embeddings[0]["embedding"]
        print(f"✅ Face encoding generated — {len(embedding)} dimensions")
        return embedding

    except ValueError:
        raise  # re-raise our own clear messages
    except Exception as e:
        # DeepFace raises a generic exception when no face is found
        # with enforce_detection=True — turn it into a clear message
        msg = str(e).lower()
        if "face" in msg or "detect" in msg or "extract" in msg:
            raise ValueError(
                "No face detected in the image. Please ensure your face is clearly "
                "visible, well-lit, and not obscured."
            )
        raise ValueError(f"Face encoding failed: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def verify_single_face(
    selfie_bytes: bytes,
    stored_encoding: List[float]
) -> Dict:
    """
    Compare selfie against stored embedding.
    Returns matched, distance, threshold — or an error message.
    """
    tmp_path = save_temp_image(selfie_bytes)
    try:
        # Also strict for verification — if selfie has no detectable face, reject it.
        # This is what prevents "any photo" from passing: without a real face
        # in the selfie, we return an error instead of comparing garbage data.
        try:
            selfie_embeddings = DeepFace.represent(
                img_path=tmp_path,
                model_name=FACE_MODEL,
                detector_backend=FACE_DETECTOR,
                enforce_detection=True,
                align=True
            )
        except Exception as e:
            msg = str(e).lower()
            if "face" in msg or "detect" in msg or "extract" in msg:
                return {
                    "matched": False,
                    "distance": None,
                    "error": (
                        "No face detected in your selfie. "
                        "Please face the camera directly in good lighting."
                    ),
                }
            return {
                "matched": False,
                "distance": None,
                "error": "Selfie processing failed. Please try again.",
            }

        if not selfie_embeddings:
            return {
                "matched": False,
                "distance": None,
                "error": "No face detected in selfie. Please face the camera directly.",
            }

        if len(selfie_embeddings) > 1:
            return {
                "matched": False,
                "distance": None,
                "error": "Multiple faces detected in selfie. Only your face should be visible.",
            }

        selfie_vector = np.array(selfie_embeddings[0]["embedding"])
        stored_vector = np.array(stored_encoding)

        # Dimension mismatch means the stored encoding used a different model
        if selfie_vector.shape != stored_vector.shape:
            return {
                "matched": False,
                "distance": None,
                "error": (
                    "Your stored face data is from a different model. "
                    "Please re-register your face in your profile."
                ),
            }

        # Cosine distance
        if DISTANCE_METRIC == "cosine":
            dot    = np.dot(selfie_vector, stored_vector)
            norm_a = np.linalg.norm(selfie_vector)
            norm_b = np.linalg.norm(stored_vector)
            if norm_a == 0 or norm_b == 0:
                distance = 1.0
            else:
                distance = float(1 - (dot / (norm_a * norm_b)))
        else:
            distance = float(np.linalg.norm(selfie_vector - stored_vector))

        matched = distance <= THRESHOLD

        print(
            f"Face verification | distance={distance:.4f} | "
            f"threshold={THRESHOLD} | matched={matched}"
        )

        return {
            "matched":   matched,
            "distance":  round(distance, 4),
            "threshold": THRESHOLD,
        }

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def recognize_faces_in_class_photo(
    image_bytes: bytes,
    students_with_encodings: List[Dict]
) -> Tuple[List[str], int, int]:
    """Recognize multiple students in a class photo."""
    tmp_path = save_temp_image(image_bytes)
    recognized_ids: List[str] = []

    try:
        try:
            class_embeddings = DeepFace.represent(
                img_path=tmp_path,
                model_name=FACE_MODEL,
                detector_backend=FACE_DETECTOR,
                enforce_detection=False,  # OK here — we expect multiple faces
                align=True
            )
        except Exception as e:
            print(f"Class photo face detection error: {e}")
            return [], 0, 0

        if not class_embeddings:
            return [], 0, 0

        total_detected = len(class_embeddings)
        unrecognized   = 0

        for face_data in class_embeddings:
            face_vector   = np.array(face_data["embedding"])
            best_match_id: Optional[str] = None
            best_distance = float("inf")

            for student in students_with_encodings:
                stored_encoding = student.get("faceEncoding")
                if not stored_encoding:
                    continue

                stored_vector = np.array(stored_encoding)
                if face_vector.shape != stored_vector.shape:
                    continue

                dot    = np.dot(face_vector, stored_vector)
                norm_a = np.linalg.norm(face_vector)
                norm_b = np.linalg.norm(stored_vector)
                distance = float(1 - (dot / (norm_a * norm_b))) if norm_a and norm_b else 1.0

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