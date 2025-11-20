from fastapi import FastAPI, UploadFile, File, HTTPException
from facenet_pytorch import MTCNN, InceptionResnetV1
from pydantic import BaseModel
from typing import List
from PIL import Image
import numpy as np
import io
import cv2
import torch
import chromadb
import uuid

app = FastAPI()

# ---------------------------
# ML MODELS
# ---------------------------
detector = MTCNN(keep_all=True)
embedder = InceptionResnetV1(pretrained="vggface2").eval()

# ---------------------------
# CHROMADB DATABASE 
# ---------------------------
EMBEDDING_DIM = 512
COLLECTION_NAME = "face_collection"
PERSIST_DIRECTORY = "./chroma_db"

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

# Get or create collection (uses cosine similarity by default)
try:
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    print(f"Loaded existing collection: {COLLECTION_NAME}")
except:
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Face embeddings with cosine similarity"}
    )
    print(f"Created new collection: {COLLECTION_NAME}")



SIMILARITY_THRESHOLD = 0.40   
class SearchPayload(BaseModel):
    embedding: List[float]
    file_name: str          
    top_k: int = 5
class StorePayload(BaseModel):
    file_name: str
    embedding: List[float]




# ---------------------------
# FACE EMBEDDING
# ---------------------------
@app.post("/face/process")
async def detect_and_embed(image: UploadFile = File(...)):
    if not image.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(400, "Invalid image format")

    content = await image.read()
    pil_image = Image.open(io.BytesIO(content)).convert("RGB")
    arr = np.array(pil_image)

    boxes, probs = detector.detect(arr)
    if boxes is None:
        return {"message": "No face found"}

    x1, y1, x2, y2 = map(int, boxes[0])
    face = arr[y1:y2, x1:x2]

    face_img = Image.fromarray(face).resize((160, 160))
    face_tensor = torch.tensor(np.array(face_img)).permute(2, 0, 1).float() / 255.0
    face_tensor = face_tensor.unsqueeze(0)

    with torch.no_grad():
        embedding = embedder(face_tensor).squeeze().tolist()

    return {"file_name": image.filename, "embedding": embedding}


# ---------------------------
# STORE FACE
# ---------------------------
@app.post("/face/store")
async def store_face(payload: StorePayload):
    if len(payload.embedding) != 512:
        raise HTTPException(400, "Embedding must be 512 dims")

    # Convert embedding to list (ChromaDB expects list of floats)
    embedding = payload.embedding
    
    # Generate unique ID
    face_id = str(uuid.uuid4())
    
    # Add to ChromaDB (automatically handles cosine similarity)
    collection.add(
        embeddings=[embedding],
        metadatas=[{"file_name": payload.file_name}],
        ids=[face_id]
    )

    return {
        "stored": True,
        "face_id": face_id,
        "file_name": payload.file_name
    }


# ---------------------------
# SEARCH FACE 
# ---------------------------
@app.post("/face/search")
async def search_face(payload: SearchPayload):
    if len(payload.embedding) != 512:
        raise HTTPException(400, "Embedding must be 512 dims")

    # Query ChromaDB (cosine distance)
    results = collection.query(
        query_embeddings=[payload.embedding],
        n_results=payload.top_k,
        include=["metadatas", "distances"]
    )

    # If no results at all → Save directly
    if not results["ids"] or len(results["ids"][0]) == 0:
        face_id = str(uuid.uuid4())
        collection.add(
            embeddings=[payload.embedding],
            metadatas=[{"file_name": payload.file_name}],
            ids=[face_id]
        )
        return {
            "exists": False,
            "stored": True,
            "face_id": face_id,
            "file_name": payload.file_name,
            "message": "No similar face found — new face stored."
        }

    # Evaluate top result
    best_id = results["ids"][0][0]
    best_distance = float(results["distances"][0][0])
    best_metadata = results["metadatas"][0][0]

    similarity = 1 - best_distance   # convert distance → similarity

    # If similarity above threshold → return matched
    if similarity >= SIMILARITY_THRESHOLD:
        return {
            "exists": True,
            "stored": False,
            "face_id": best_id,
            "similarity": similarity,
            "file_name": best_metadata.get("file_name", ""),
            "message": "Face already exists."
        }

    # Otherwise → Store new face
    new_id = str(uuid.uuid4())
    collection.add(
        embeddings=[payload.embedding],
        metadatas=[{"file_name": payload.file_name}],
        ids=[new_id]
    )

    return {
        "exists": False,
        "stored": True,
        "face_id": new_id,
        "similarity": similarity,
        "file_name": payload.file_name,
        "message": "No match — stored as new face."
    }# ---------------------------
# HEALTH CHECK
# ---------------------------
@app.get("/")
async def health_check():
    count = collection.count()
    return {
        "status": "running",
        "database": "ChromaDB (Cosine Similarity)",
        "total_faces": count,
        "embedding_dim": EMBEDDING_DIM,
        "collection": COLLECTION_NAME
    }
