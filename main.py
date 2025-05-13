# FastAPI本体とセキュリティ関連
import os
from datetime import datetime, timedelta
from typing import List, Optional

from databases import Database

# secret key の環境変数から読み取り################################################
from dotenv import load_dotenv  # .envファイルを読み取るためのimport
from fastapi import Body, Depends, FastAPI, File, HTTPException, UploadFile, Response, Request
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt

# 認証・ハッシュ・トークン生成
from passlib.context import CryptContext

# モデル・DB関連
from pydantic import BaseModel
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    and_,
    create_engine,
    select,
)
from starlette.middleware.cors import CORSMiddleware

from prompt import generate_dummy_topic  # ← 追加
from save_image import save_image_locally

load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY") or "dummy-secret"
if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY is not set in environment variables!")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
###############################################################################


oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/login"
)  # Swagger UIが認識するログイン先
pwd_context = CryptContext(
    schemes=["bcrypt"], deprecated="auto"
)  # パスワードハッシュ用

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # ← "*" は NG
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = "sqlite:///./database.db"  # 同じディレクトリ内のtest2.dbファイル
database = Database(DATABASE_URL)
metadata = MetaData()  # metadataを生成

"""
テーブル：

ユーザーテーブル：
　id（自動生成）、ユーザー名（username）、パスワード（password）、性別（gender）、学部・研究科（department）、趣味（hobby）、出身地（hometown）、言語（language）、ステータス（status）

イベントテーブル：
　id（自動生成）、名前（name）、場所（place）、時間（time）、登録ユーザー（registered users）

ログインページ >>
ユーザー情報：ユーザー名、パスワード

アカウント作成ページ >>
ユーザー登録情報：id（自動生成）、ユーザー名、パスワード、性別、学部、趣味、出身地、言語

イベント情報ページ >>
イベント情報：id（自動生成）、名前、場所、時間、登録ユーザー

おすすめユーザーページ >>
ユーザー一覧：ユーザー名、ステータス（フィルター：趣味、学部、出身地、言語；ソート可能）

生成AIページ >>
入力情報：ユーザー名、性別、学部、趣味、出身地、言語
（生成AIが必要な情報：性別、学部、趣味、出身地、言語）
（生成後、ステータスが変更される）
"""

# usersテーブル定義（id, name, hashed_password）適宜追加可能
users = Table(
    "users",
    metadata,  # metadataとは、tableの情報を保持するための変数。ここに、Columnの情報等が入っている。ここでは、usersというTableをmetadataに格納している
    Column("id", Integer, primary_key=True),
    Column("name", String, nullable=False),
    Column("hashed_password", String, nullable=False),
    Column("gender", String, nullable=True),
    Column("department", String, nullable=True),
    Column("hobbies", JSON, nullable=True),
    Column("hometown", String, nullable=True),
    Column("languages", JSON, nullable=True),
    Column("status", Integer, nullable=True),
    Column("talked_count", Integer, nullable=True),
    Column("role", String, nullable=False),
    Column("image_path", String, nullable=True),
)

events = Table(
    "events",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("event_name", String, nullable=False),
    Column("place", String, nullable=True),
    Column("start_time", DateTime, nullable=True),
    Column("end_time", DateTime, nullable=True),
    Column("registered_users", JSON, nullable=True),
    Column("creater", String, nullable=False),
)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
# 実際にDBと通信するための接続機
# SQliteからPostgreSQLに切り替える時は、（connect_argsは不要）
# engine = create_engine("postgresql://user:pass@localhost/dbname")に変える


metadata.create_all(
    engine
)  # ここで、metadataに格納されているすべてのTableを読み取り、DBを構築する


####fastAPIが受け取り、返す型を定義している
class UserCreate(BaseModel):  # 登録用
    name: str
    password: str
    gender: str
    department: str
    hobbies: List[str]  # not sure about this one
    hometown: str
    languages: List[str]


class UserLogin(BaseModel):  # 未使用（今はOAuth2Formに依存）
    name: str  # <=clientの送ってくる[name]は、str型出なくてはならない


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# サーバー起動中にcurl http://localhost:8000/users でuserリスト確認


# イベント相関
class EventCreate(BaseModel):
    event_name: str
    place: str
    start_time: datetime
    end_time: datetime
    registered_users: List[str]


# Pydanticモデル（入力と出力）
class UserIn(BaseModel):
    name: str  # <=clientの送ってくる[name]は、str型出なくてはならない


class UserOut(BaseModel):
    id: int  # APIの返すidはintでなければならない
    name: str  # ,,,はstrでなければならない


class UserNameOut(BaseModel):
    name: str  # ,,,はstrでなければならない


class EventIn(BaseModel):
    event_name: Optional[str] = None
    place: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    registered_users: Optional[List[str]] = None


class EventOut(BaseModel):
    id: int
    event_name: str


class EventInfoOut(BaseModel):
    id: int
    event_name: str
    place: str
    start_time: datetime
    end_time: datetime
    registered_users: List[str]
    creater: str


class UserChange(BaseModel):
    name: Optional[str] = None
    password: Optional[str] = None
    gender: Optional[str] = None
    department: Optional[str] = None
    hobbies: Optional[List[str]] = None
    hometown: Optional[str] = None
    languages: Optional[List[str]] = None


class UserInfoOut(BaseModel):
    id: int
    name: str
    gender: str
    department: str
    hobbies: List[str]
    hometown: str
    languages: List[str]
    status: int


# トークン検証用の関数
async def get_current_user(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    query = users.select().where(users.c.name == username)
    user = await database.fetch_one(query)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

##############エンドポイント###############
# エンドポイントとは、DBにアクセスするための窓口のようなもの


# 接続処理
@app.on_event("startup")
async def startup():
    await database.connect()


@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

@app.post("/login")
async def login(response: Response, form_data: OAuth2PasswordRequestForm = Depends()):
    query = users.select().where(users.c.name == form_data.username)
    db_user = await database.fetch_one(query)
    if db_user is None or not verify_password(form_data.password, db_user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = create_access_token(data={"sub": form_data.username})

    # Cookie にセット（secure, samesite は環境に応じて）
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        secure=False,       # ローカル開発なら False、本番では True
        samesite="Lax",     # cross-site の場合は "None" + secure=True
        max_age=1800,
        expires=1800,
        path="/",
    )

    return {"message": "Login successful"}

# GET: ユーザー一覧取得
@app.get("/users", response_model=list[UserOut])  # (ユーザ全員の情報)
async def get_users():
    query = select(users.c.id, users.c.name)
    return await database.fetch_all(query)


# [リクエスト：URLにユーザーIDを入れる（例：/users/1）
# {レスポンス
#  "id": 1,
# "name": "たくみ"
# },
# {
# "id": 2,
#  "name": "あやの"
# }
# ]
# {エラー時
# "detail": "User not found"
# }
# 認証つきエンドポイント
@app.get("/users/me", response_model=UserOut)
async def read_current_user(request: Request):
    token = request.cookies.get("access_token")
    print(">>> Cookie access_token:", token)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    query = users.select().where(users.c.name == username)
    user = await database.fetch_one(query)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    return user


@app.get("/users/{user_id}", response_model=UserInfoOut)  # (ユーザ個別情報)
async def get_user_by_id(user_id: int):
    query = users.select().where(users.c.id == user_id)
    user = await database.fetch_one(query)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


# {
#  "id": 1,
#  "name": "たくみ"
# }


@app.get("/")
def root():
    return {"message": "Welcome to HOLLA backend!"}


# POST: ユーザー登録
@app.post("/users", response_model=UserOut)
async def create_user(user: UserIn):
    query = users.insert().values(name=user.name)
    user_id = await database.execute(query)
    return {**user.dict(), "id": user_id}


# {リクエスト
#  "name": "たくみ"
# }

# {レスポンス：idと一緒に返される
#  "id": 3,
#  "name": "たくみ"
# }


@app.post("/register", response_model=UserOut)  ##登録用POST
async def register_user(user: UserCreate):
    hashed_pw = hash_password(user.password)

    query = users.insert().values(
        name=user.name,
        hashed_password=hashed_pw,
        gender=user.gender,
        department=user.department,
        hobbies=user.hobbies,
        hometown=user.hometown,
        languages=user.languages,
        status=0,
        role="participants",
    )

    user_id = await database.execute(query)
    return {**user.dict(exclude={"password"}), "id": user_id}


# {リクエスト
#  "name": "たくみ"
# "password": "secretpass"
# }
# {レスポンス：idと一緒に返される
#  "id": 3,
#  "name": "たくみ"
# }


@app.post("/register/admin", response_model=UserOut)  ##登録用POST
async def register_user_admin(user: UserCreate):
    hashed_pw = hash_password(user.password)

    query = users.insert().values(
        name=user.name,
        hashed_password=hashed_pw,
        role="admin",
    )

    user_id = await database.execute(query)
    return {**user.dict(exclude={"password"}), "id": user_id}


@app.put("/users/{user_id}", response_model=UserInfoOut)
async def update_user(user_id: int, user: UserChange = Body(...)):
    # 対象のユーザーが存在するか確認
    query = users.select().where(users.c.id == user_id)
    existing_user = await database.fetch_one(query)
    if existing_user is None:
        raise HTTPException(status_code=404, detail="User not found")

    # 更新クエリを発行
    update_data = user.dict(exclude_unset=True)
    if "password" in update_data:
        update_data["hashed_password"] = hash_password(update_data.pop("password"))
    if update_data:
        update_query = users.update().where(users.c.id == user_id).values(**update_data)
        await database.execute(update_query)

    await database.execute(update_query)

    # 更新後の情報を取得して返す
    updated_user = await database.fetch_one(users.select().where(users.c.id == user_id))
    return updated_user


# リクエスト
# URLにID, ボディに新しい名前
# PUT /users/3
# {
# "name": "たくみ（改）"
# }

# レスポンス
# {
# "id": 3,
# "name": "たくみ（改）"
# }
#
#
#


@app.delete("/users/{user_id}", response_model=UserOut)
async def delete_user(user_id: int):
    # 対象のユーザーが存在するか確認
    query = users.select().where(users.c.id == user_id)
    existing_user = await database.fetch_one(query)
    if existing_user is None:
        raise HTTPException(status_code=404, detail="User not found")

    # 削除クエリを実行
    delete_query = users.delete().where(users.c.id == user_id)
    await database.execute(delete_query)

    # 削除前のデータを返す（確認用）
    return existing_user


# リクエスト：URLに削除対象のIDを指定
# レスポンス
# {
# "id": 3,
# "name": "たくみ（改）"
# }


# POST: イベント登録
@app.post("/events/register", response_model=EventOut)
async def register_event(
    event: EventCreate, current_user: dict = Depends(get_current_user)
):
    query = events.insert().values(
        event_name=event.event_name,
        place=event.place,
        start_time=event.start_time,
        end_time=event.end_time,
        registered_users=event.registered_users,
        creater=current_user["name"],
    )
    event_id = await database.execute(query)
    return {**event.dict(), "id": event_id}


# GET: 現在進行中のイベントを獲得
@app.get("/events/active", response_model=List[EventOut])
async def get_active_events():
    now = datetime.utcnow()
    query = events.select().where(
        (events.c.start_time <= now) & (events.c.end_time >= now)
    )
    active_events = await database.fetch_all(query)
    if active_events is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return active_events


# GET: UserNameで参加しているイベントを獲得
@app.get("/events/user/{user_name}", response_model=List[EventOut])
async def get_user_events(user_name: str):
    like_pattern = f'%"{user_name}"%'
    query = events.select().where(events.c.registered_users.like(like_pattern))
    user_events = await database.fetch_all(query)
    if not user_events:
        raise HTTPException(status_code=404, detail="No events found")
    return user_events


# GET: 現在のユーザーが参加しているイベントを獲得
@app.get("/events/me", response_model=List[EventOut])
async def get_my_event(current_user: dict = Depends(get_current_user)):
    like_pattern = f'%{current_user["name"]}%'
    query = events.select().where(events.c.registered_users.like(like_pattern))
    user_events = await database.fetch_all(query)
    if not user_events:
        raise HTTPException(status_code=404, detail="No events found")
    return user_events


# GET: 現在のユーザーがcreateしたイベントを獲得
@app.get("/events/creater", response_model=List[EventOut])
async def get_creater_event(current_user: dict = Depends(get_current_user)):
    like_pattern = f'%{current_user["name"]}%'
    query = events.select().where(events.c.creater.like(like_pattern))
    user_events = await database.fetch_all(query)
    if not user_events:
        raise HTTPException(status_code=404, detail="No events found")
    return user_events


# PUT: イベント更新
@app.put("/events/{event_id}", response_model=EventInfoOut)
async def update_event(event_id: int, event: EventIn = Body(...)):
    query = events.select().where(events.c.id == event_id)
    existing_event = await database.fetch_one(query)
    if existing_event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    update_data = event.dict(exclude_unset=True)
    if update_data:
        update_query = (
            events.update().where(events.c.id == event_id).values(**update_data)
        )
        await database.execute(update_query)
    await database.execute(update_query)
    updated_event = await database.fetch_one(
        events.select().where(events.c.id == event_id)
    )
    return updated_event


# DELETE: イベントをIDで削除
@app.delete("/events/{event_id}", response_model=EventOut)
async def delete_event(event_id: int):
    query = events.select().where(events.c.id == event_id)
    existing_event = await database.fetch_one(query)
    if existing_event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    delete_query = events.delete().where(events.c.id == event_id)
    await database.execute(delete_query)
    return existing_event


# イベント参加
@app.post("/events/{event_id}/join", response_model=EventInfoOut)
async def join_event(event_id: int, current_user: dict = Depends(get_current_user)):
    query = events.select().where(events.c.id == event_id)
    event = await database.fetch_one(query)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")

    registered_users = event["registered_users"] or []
    if current_user["name"] not in registered_users:
        registered_users.append(str(current_user["name"]))
        update_query = (
            events.update()
            .where(events.c.id == event_id)
            .values(registered_users=registered_users)
        )
        await database.execute(update_query)

    updated_event = await database.fetch_one(
        events.select().where(events.c.id == event_id)
    )
    return updated_event


#####話しかけられた回数を更新するためのエンドポイントを追加
@app.post("/users/{user_id}/increment_talk_count")
async def increment_talked_count(user_id: int):
    query = users.select().where(users.c.id == user_id)
    user = await database.fetch_one(query)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    new_count = (user["talked_count"] or 0) + 1
    update_query = (
        users.update().where(users.c.id == user_id).values(talked_count=new_count)
    )
    await database.execute(update_query)
    return {"id": user_id, "talked_count": new_count}


@app.get("/get_recommended_list", response_model=List[int])
async def get_recommended_list(current_user: dict = Depends(get_current_user)):
    query = (
        users.select()
        .where(and_(users.c.status == 1, users.c.id != current_user["id"]))
        .order_by(users.c.talked_count.asc())
    )
    user_list = await database.fetch_all(query)
    user_ids = [user["id"] for user in user_list]
    return user_ids


@app.get("/users/{user_id}/get_status", response_model=int)
async def get_status(user_id: int):
    query = select(users.c.status).where(users.c.id == user_id)
    result = await database.fetch_one(query)

    if result is None:
        raise HTTPException(status_code=404, detail="User not found")

    return result["status"]


@app.put("/users/{user_id}/status", response_model=UserOut)
async def update_status(user_id: int):
    await database.connect()
    query = users.select().where(users.c.id == user_id)
    existing_user = await database.fetch_one(query)
    if existing_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    update_query = users.update().where(users.c.id == user_id).values(status=1)
    await database.execute(update_query)
    return existing_user


####################################chatGPTによるサジェスト


@app.post("/topic/generate")
async def generate_topic(current_user: dict = Depends(get_current_user)):
    name = current_user["name"]
    # gender = current_user["gender"] or "不明"
    department = current_user["department"] or "未設定"
    hobbies = current_user["hobbies"] or []
    hometown = current_user["hometown"] or "不明"
    # languages = current_user["languages"] or "不明"

    # 外部に分離された関数を使って話題生成
    generated_topic = generate_dummy_topic(name, department, hobbies, hometown)

    return {"suggested_topic": generated_topic}


@app.post("/users/{user_id}/upload_image")
async def upload_image(user_id: int, file: UploadFile = File(...)):
    # ユーザーの存在確認
    query = users.select().where(users.c.id == user_id)
    user = await database.fetch_one(query)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    # ユーザーがいない場合のエラー処理
    # 画像保存
    image_path = save_image_locally(file, user_id)

    # 画像パスをデータベースに保存
    update_query = (
        users.update().where(users.c.id == user_id).values(image_path=image_path)
    )
    await database.execute(update_query)

    return {"message": "Image uploaded successfully", "image_path": image_path}


# アップロード後、画像のパスが users テーブルの image_path カラムに保存される。
@app.get("/users/{user_id}/image")
async def get_user_image(user_id: int):
    # ユーザーの画像パスを取得
    query = users.select().where(users.c.id == user_id)
    user = await database.fetch_one(query)
    if user is None or not user["image_path"]:
        raise HTTPException(status_code=404, detail="Image not found")

    # 画像ファイルが存在するか確認
    image_path = user["image_path"]
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image file not found")

    # 画像を返す
    return FileResponse(image_path)


# とりあえず画像をUploaded_imagesに追加。


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # ファイルの保存
    file_path = save_image_locally(
        file, 0
    )  # File format specified by save_image_locally

    return {"info": f"file '{file.filename}' saved at '{file_path}'"}
