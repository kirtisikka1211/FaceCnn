# from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# from fastapi.responses import JSONResponse
# from facenet_pytorch import MTCNN, InceptionResnetV1
# from PIL import Image
# import numpy as np
# import io
# import cv2
# import base64
# import torch

# app = FastAPI()

# embedder = InceptionResnetV1(pretrained="vggface2").eval()

# # Initialize models
# detector = MTCNN(keep_all=True)
# # embedder = FaceNet()


# @app.post("/face/detect")
# async def detect_face(image: UploadFile = File(...)):
#     # Validate file type
#     if not image.filename.lower().endswith((".jpg", ".jpeg", ".png")):
#         raise HTTPException(status_code=400, detail="Invalid image format")

#     file_name = image.filename

#     # Read image bytes
#     content = await image.read()
#     pil_image = Image.open(io.BytesIO(content)).convert("RGB")

#     np_image = np.array(pil_image)

#     # Run MTCNN detection
#     boxes, probs = detector.detect(np_image)

#     if boxes is None:
#         return JSONResponse({
#             "file_name": file_name,
#             "faces_detected": 0,
#             "message": "No face found"
#         })

#     detections = []
#     for box, confidence in zip(boxes, probs):
#         x1, y1, x2, y2 = [float(v) for v in box]
#         detections.append({
#             "bounding_box": {
#                 "x1": x1,
#                 "y1": y1,
#                 "x2": x2,
#                 "y2": y2,
#                 "width": x2 - x1,
#                 "height": y2 - y1
#             },
#             "confidence": float(confidence)
#         })

#     return {
#         "file_name": file_name,
#         "faces_detected": len(detections),
#         "detections": detections
#     }

# @app.post("/face/embed")
# async def generate_embedding(
#     image: UploadFile = File(...),
#     x1: float = Form(...),
#     y1: float = Form(...),
#     x2: float = Form(...),
#     y2: float = Form(...),
#     file_name: str = Form(...)
# ):

#     if not image.filename.lower().endswith((".jpg", ".jpeg", ".png")):
#         raise HTTPException(status_code=400, detail="Invalid image format")

#     # Load image
#     content = await image.read()
#     pil_image = Image.open(io.BytesIO(content)).convert("RGB")
#     np_image = np.array(pil_image)

#     cropped_face = np_image[int(y1):int(y2), int(x1):int(x2)]
#     if cropped_face.size == 0:
#         raise HTTPException(status_code=400, detail="Invalid bounding box")

#     # Resize to 160x160 for InceptionResnetV1 (standard)
#     face_img = Image.fromarray(cropped_face).resize((160, 160))
#     face_tensor = torch.tensor(np.array(face_img)).permute(2, 0, 1).float() / 255.0
#     face_tensor = face_tensor.unsqueeze(0)

#     # Generate embedding
#     with torch.no_grad():
#         embedding = embedder(face_tensor).squeeze().tolist()

#     # Encode cropped face
#     _, buffer = cv2.imencode(".jpg", cropped_face)
#     cropped_b64 = base64.b64encode(buffer).decode()

#     return {
#         "file_name": file_name,
#         "bounding_box_used": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
#         "embedding": embedding,
#         "cropped_face_b64": cropped_b64
#     }

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import io
import uuid
import os
import cv2
import base64
import torch

app = FastAPI()

# Initialize detector + embedder
detector = MTCNN(keep_all=True)
embedder = InceptionResnetV1(pretrained="vggface2").eval()

TEMP_DIR = "./temp_images"
os.makedirs(TEMP_DIR, exist_ok=True)

# -----------------------------
# 1. FACE DETECTION ENDPOINT
# -----------------------------
@app.post("/face/detect")
async def detect_face(image: UploadFile = File(...)):

    if not image.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Invalid image format")

    file_name = image.filename

    # Read image bytes
    content = await image.read()
    pil_image = Image.open(io.BytesIO(content)).convert("RGB")
    np_image = np.array(pil_image)

    # Save temp file
    temp_id = str(uuid.uuid4()) + "_" + file_name
    temp_path = os.path.join(TEMP_DIR, temp_id)
    pil_image.save(temp_path)

    # Detect faces
    boxes, probs = detector.detect(np_image)

    if boxes is None:
        return JSONResponse({
            "file_name": file_name,
            "temp_path": temp_path,
            "faces_detected": 0,
            "message": "No face found"
        })

    detections = []
    for box, confidence in zip(boxes, probs):
        x1, y1, x2, y2 = [float(v) for v in box]
        detections.append({
            "bounding_box": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": x2 - x1,
                "height": y2 - y1
            },
            "confidence": float(confidence)
        })

    return {
        "file_name": file_name,
        "temp_path": temp_path,
        "faces_detected": len(detections),
        "detections": detections
    }


# -----------------------------
# 2. EMBEDDING GENERATION ENDPOINT
# -----------------------------
@app.post("/face/embed")
async def generate_embedding(
    x1: float = Form(...),
    y1: float = Form(...),
    x2: float = Form(...),
    y2: float = Form(...),
    temp_path: str = Form(...)
):

    if not os.path.exists(temp_path):
        raise HTTPException(status_code=400, detail="Temp file not found")

    # Load image again from disk
    pil_image = Image.open(temp_path).convert("RGB")
    np_image = np.array(pil_image)

    # Crop using the bounding box
    cropped_face = np_image[int(y1):int(y2), int(x1):int(x2)]
    if cropped_face.size == 0:
        raise HTTPException(status_code=400, detail="Invalid bounding box")

    # Resize for FaceNet
    face_img = Image.fromarray(cropped_face).resize((160, 160))
    face_tensor = torch.tensor(np.array(face_img)).permute(2, 0, 1).float() / 255.0
    face_tensor = face_tensor.unsqueeze(0)

    # Generate embedding
    with torch.no_grad():
        embedding = embedder(face_tensor).squeeze().tolist()

    # Convert cropped face to base64 for preview
    _, buffer = cv2.imencode(".jpg", cropped_face)
    cropped_b64 = base64.b64encode(buffer).decode()

    return {
        "temp_path": temp_path,
        "bounding_box_used": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "embedding": embedding,
        "cropped_face_b64": cropped_b64
    }
