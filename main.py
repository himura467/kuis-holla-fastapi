# FastAPI本体とセキュリティ関連
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

# モデル・DB関連
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, DateTime, JSON, create_engine, MetaData, Table, text
from databases import Database
from typing import List

# 認証・ハッシュ・トークン生成
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta

#secret key の環境変数から読み取り################################################
from dotenv import load_dotenv #.envファイルを読み取るためのimport
import os
load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY") or "dummy-secret"
if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY is not set in environment variables!")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
###############################################################################


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")  # Swagger UIが認識するログイン先
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")  # パスワードハッシュ用

app = FastAPI()

DATABASE_URL = "sqlite:///./test2.db" #同じディレクトリ内のtest2.dbファイル
database = Database(DATABASE_URL)
metadata = MetaData() #metadataを生成

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
    metadata,#metadataとは、tableの情報を保持するための変数。ここに、Columnの情報等が入っている。ここでは、usersというTableをmetadataに格納している
    Column("id", Integer, primary_key=True),
    Column("name", String, nullable=False),
    Column("hashed_password", String, nullable=False),
    Column("gender", String, nullable=True),
    Column("department", String, nullable=True),
    Column("hobby", JSON, nullable=True),
    Column("hometown", String, nullable=True),
    Column("language", String, nullable=True),
    Column("status", Integer, nullable=True)
)

events = Table(
    "events",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("event_name", String, nullable=False),
    Column("place", String, nullable=True),
    Column("start_time", DateTime, nullable=True),
    Column("end_time", DateTime, nullable=True),
    Column("registered_users", JSON, nullable=True)
)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
#実際にDBと通信するための接続機
#SQliteからPostgreSQLに切り替える時は、（connect_argsは不要）
#engine = create_engine("postgresql://user:pass@localhost/dbname")に変える


metadata.create_all(engine)#ここで、metadataに格納されているすべてのTableを読み取り、DBを構築する



####fastAPIが受け取り、返す型を定義している
class UserCreate(BaseModel):  # 登録用
    name: str
    password: str
    gender: str
    department: str
    hobby: List[str] # not sure about this one
    hometown: str
    language: str

class UserLogin(BaseModel):  # 未使用（今はOAuth2Formに依存）
      name: str #<=clientの送ってくる[name]は、str型出なくてはならない 

class UserIn(BaseModel):     # 更新時など

    name: str


class UserOut(BaseModel):    # 出力用（パスワード非表示）
    id: int
    name: str

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

#サーバー起動中にcurl http://localhost:8000/users でuserリスト確認


# イベント相関
class EventCreate(BaseModel):
    event_name: str
    place: str
    start_time: datetime
    end_time: datetime
    registered_users: List[str]

# Pydanticモデル（入力と出力）
class UserIn(BaseModel):
    name: str #<=clientの送ってくる[name]は、str型出なくてはならない

class UserOut(BaseModel):
    id: int #APIの返すidはintでなければならない
    name: str #,,,はstrでなければならない

class EventOut(BaseModel):
    id: int
    event_name: str

#トークン検証用の関数
async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    # ユーザー検索
    query = users.select().where(users.c.name == username)
    user = await database.fetch_one(query)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


##############エンドポイント###############
#エンドポイントとは、DBにアクセスするための窓口のようなもの

# 接続処理
@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

@app.post("/login") #ログイン.成功すると、アクセストークンをレスポンスとして返す。
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    query = users.select().where(users.c.name == form_data.username)
    db_user = await database.fetch_one(query)
    if db_user is None or not verify_password(form_data.password, db_user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}

# GET: ユーザー一覧取得
@app.get("/users", response_model=list[UserOut])#(ユーザ全員の情報)
async def get_users():
    query = users.select()
    return await database.fetch_all(query)
#[リクエスト：URLにユーザーIDを入れる（例：/users/1）
 # {レスポンス
  #  "id": 1,
   # "name": "たくみ"
  #},
  #{
   # "id": 2,
  #  "name": "あやの"
  #}
#]
#{エラー時
 # "detail": "User not found"
#}
#認証つきエンドポイント
@app.get("/users/me", response_model=UserOut)
async def read_current_user(current_user: dict = Depends(get_current_user)):
    return current_user

@app.get("/users/{user_id}", response_model=UserOut)#(ユーザ個別情報)
async def get_user_by_id(user_id: int):
    query = users.select().where(users.c.id == user_id)
    user = await database.fetch_one(query)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user
#{
#  "id": 1,
#  "name": "たくみ"
#}

@app.get("/")
def root():
    return {"message": "Welcome to HOLLA backend!"}


# POST: ユーザー登録
@app.post("/users", response_model=UserOut)
async def create_user(user: UserIn):
    query = users.insert().values(name=user.name)
    user_id = await database.execute(query)
    return {**user.dict(), "id": user_id}
#{リクエスト
#  "name": "たくみ"
#}

#{レスポンス：idと一緒に返される
#  "id": 3,
#  "name": "たくみ"
#}


@app.post("/register", response_model=UserOut)##登録用POST
async def register_user(user: UserCreate):
    hashed_pw = hash_password(user.password)
    query = users.insert().values(name=user.name, hashed_password=hashed_pw, gender=user.gender, department=user.department, hobby=user.hobby, hometown=user.hometown, language=user.language, status=0)
    user_id = await database.execute(query)
    return {**user.dict(exclude={"password"}), "id": user_id}
#{リクエスト
#  "name": "たくみ"
#"password": "secretpass"
#}
#{レスポンス：idと一緒に返される
#  "id": 3,
#  "name": "たくみ"
#}


@app.put("/users/{user_id}", response_model=UserOut)
async def update_user(user_id: int, user: UserIn):
    # 対象のユーザーが存在するか確認
    query = users.select().where(users.c.id == user_id)
    existing_user = await database.fetch_one(query)
    if existing_user is None:
        raise HTTPException(status_code=404, detail="User not found")

    # 更新クエリを発行
    update_query = users.update().where(users.c.id == user_id).values(name=user.name)
    await database.execute(update_query)

    # 更新後の情報を取得して返す
    return {**user.dict(), "id": user_id}
#リクエスト
#URLにID, ボディに新しい名前
# PUT /users/3
#{
 # "name": "たくみ（改）"
#}

#レスポンス
#{
 # "id": 3,
 # "name": "たくみ（改）"
#}
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
#リクエスト：URLに削除対象のIDを指定
#レスポンス
#{
 # "id": 3,
  #"name": "たくみ（改）"
#}


# POST: イベント登録
@app.post("/events/register", response_model=EventOut)
async def register_event(event: EventCreate): 
    query = events.insert().values(event_name=event.event_name, place=event.place, start_time=event.start_time, end_time=event.end_time, registered_users=event.registered_users)
    event_id = await database.execute(query)
    return {**event.dict(), "id": event_id}

# GET: 現在進行中のイベントを獲得
@app.get("/events/active", response_model=List[EventOut])
async def get_active_events():
    now = datetime.utcnow()
    query = events.select().where(
        (events.c.start_time <= now) &
        (events.c.end_time >= now)
    )
    active_events = await database.fetch_all(query)
    if active_events is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return active_events

# GET: UserIDで参加しているイベントを獲得
@app.get("/events/user/{user_name}", response_model=List[EventOut])
async def get_user_events(user_name: str):
    like_pattern = f'%"{user_name}"%'
    query = events.select().where(
        events.c.registered_users.like(like_pattern)
    )
    user_events = await database.fetch_all(query)
    if not user_events:
        raise HTTPException(status_code=404, detail="No active events found for user")
    return user_events

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