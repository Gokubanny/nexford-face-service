# NexFord Face Recognition Service

Python FastAPI microservice for facial recognition and geofenced attendance.
Part of the NexFord Edu platform.

---

## How It Works

### Two Attendance Methods

**Method 1 — Student Selfie (Geofenced)**
1. Lecturer opens an attendance session (sets their GPS + duration)
2. Students open the attendance page on their phones
3. Browser captures student GPS location
4. System checks student is within allowed distance of lecturer
5. Student takes a selfie
6. Face is verified against their registered face encoding
7. If both checks pass → marked PRESENT

**Method 2 — Lecturer Class Photo**
1. Lecturer takes a group photo of the class
2. Service detects all faces in the photo
3. Each face is matched against all registered student encodings
4. All matched students → marked PRESENT

---

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Service health check |
| POST | /register | Register a student's face |
| POST | /recognize/selfie | Student selfie attendance |
| POST | /recognize/class | Lecturer class photo attendance |

All endpoints require header: `x-api-key: <INTERNAL_API_KEY>`

---

## Setup — Local Development

### 1. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
```bash
cp .env.example .env
# Fill in all values — use same MONGODB_URI as Node.js backend
```

### 4. Run the service
```bash
uvicorn main:app --reload --port 8000
```

Service runs at: `http://localhost:8000`
API docs at: `http://localhost:8000/docs`

---

## Deployment — Railway

1. Create a new Railway project
2. Connect your GitHub repo
3. Set environment variables from `.env.example`
4. Railway auto-detects `Procfile` and deploys

Set `FACE_RECOGNITION_SERVICE_URL` in your Node.js backend `.env`
to the Railway deployment URL.

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MONGODB_URI` | Same as Node.js backend | Required |
| `INTERNAL_API_KEY` | Shared secret with Node.js | Required |
| `FACE_MODEL` | DeepFace model to use | ArcFace |
| `FACE_DETECTOR` | Face detector backend | retinaface |
| `FACE_DISTANCE_METRIC` | Distance metric | cosine |
| `FACE_THRESHOLD` | Match threshold (lower = stricter) | 0.68 |
| `DEFAULT_MAX_DISTANCE_METRES` | Geofence for Basic plan | 100 |
| `CLOUDINARY_CLOUD_NAME` | Cloudinary config | Required |
| `CLOUDINARY_API_KEY` | Cloudinary config | Required |
| `CLOUDINARY_API_SECRET` | Cloudinary config | Required |

---

## Geofence Rules

| Plan | Max Distance | Configurable? |
|------|-------------|---------------|
| Basic | 100 metres | ❌ Fixed |
| Pro | 50–500 metres | ✅ School admin sets it |
| Enterprise | 50–500 metres | ✅ School admin sets it |

---

## Face Recognition Models

| Model | Accuracy | Speed | Notes |
|-------|----------|-------|-------|
| ArcFace | ⭐⭐⭐⭐⭐ | Medium | Recommended |
| Facenet512 | ⭐⭐⭐⭐⭐ | Medium | Also excellent |
| VGG-Face | ⭐⭐⭐⭐ | Slow | Good backup |

---

## Node.js Backend Integration

The Node.js backend calls this service in two places:

**1. Student face registration** (`/api/attendance/register-face`):
```
POST http://face-service/register
Form: student_id, school_id, image (file)
Header: x-api-key
```

**2. Selfie attendance** (`/api/attendance/selfie`):
```
POST http://face-service/recognize/selfie
Form: student_id, session_id, student_latitude,
      student_longitude, image (file)
Header: x-api-key
```

**3. Class photo attendance** (`/api/attendance/courses/:id/facial`):
```
POST http://face-service/recognize/class
Form: school_id, image (file)
Header: x-api-key
```

---

## MongoDB Collections Used

- `students` — reads/writes `faceEncoding` and `faceImageUrl`
- `schools` — reads `subscription.plan` and `attendanceSettings`
- `attendancesessions` — reads session details, writes `submittedStudents`
