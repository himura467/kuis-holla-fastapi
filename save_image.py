import os
from uuid import uuid4
import shutil
from fastapi import UploadFile

UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def save_image_locally(file: UploadFile, user_id: int) -> str:
    ext = file.filename.split(".")[-1]
    unique_name = f"user_{user_id}_{uuid4().hex[:8]}.{ext}"
    path = os.path.join(UPLOAD_DIR, unique_name)
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return path