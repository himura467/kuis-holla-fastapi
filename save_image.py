from fastapi import UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import os
import shutil
from uuid import uuid4

UPLOAD_DIR = "uploaded_images"
#今はローカル環境に保存する設定

# 保存先ディレクトリが存在しない場合は作成
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

def save_image_locally(file: UploadFile, user_id: int) -> str:
    # 拡張子の確認（画像ファイルかどうか）
    ext = file.filename.split(".")[-1].lower()
    allowed_extensions = ["jpg", "jpeg", "png", "gif"]
    user_id = 0
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPG, JPEG, PNG, and GIF are allowed.")
    
    # ユニークなファイル名を作成
    unique_name = f"user_{user_id}_{uuid4().hex[:8]}.{ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)

    # ファイル保存
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return file_path