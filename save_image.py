import os
import shutil
from uuid import uuid4

from fastapi import HTTPException, UploadFile  # Updated import

UPLOAD_DIR = "uploaded_images"

# If UPLOAD_DIR doesn't exist, create it
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)


def save_image_locally(file: UploadFile, user_id: int) -> str:

    # Check if filename is None
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename cannot be empty.")

    # Validate file extension (ensure it is an image)
    ext = file.filename.split(".")[-1].lower()
    allowed_extensions = ["jpg", "jpeg", "png", "gif"]
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only JPG, JPEG, PNG, and GIF are allowed.",
        )

    # Create a unique filename
    unique_name = f"user_{user_id}_{uuid4().hex[:8]}.{ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)

    # Save the file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return file_path
