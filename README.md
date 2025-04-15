# HOLLA FastAPI Backend

このリポジトリは、FastAPI によって構築された HOLLA アプリケーションのバックエンドです。  
JWT認証、SQLiteによるユーザー管理、Dockerによる環境構築などをサポートしています。

---

## ✅ 機能一覧

- ユーザー登録（パスワードはハッシュ化）
- JWTによるログイン認証
- 認証付きエンドポイント（ユーザー自身の取得）
- ユーザーのCRUD（作成・取得・更新・削除）
- Swagger UIでAPIの操作が可能
- `.env` によるシークレット管理
- Docker / Docker Compose 対応

---

## 🏗️ 環境構築手順

### ① Poetry のインストール

```bash
pip install poetry
```

### ２ pythonのバージョン切り替え
```bash
pyenv install 3.12.7
pyenv local 3.12.7
```
### 3 仮想環境の作成と依存関係のインストール
```bash
poetry install
```
### 4 Gitにあげるな！！秘密鍵のファイルを生成。.gitignoreに.envを追加
```bash
SECRET_KEY=your_super_secret_key_here
```
### 5 サーバーの起動
```bash
poetry run uvicorn main:app --reload
```
---

## 🐳 Docker 対応

FastAPIをコンテナで動かす

---

### ⑥ Docker Desktop をインストール（Macの場合）

[https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)

---

### ⑦ Dockerfile の作成

プロジェクトルートに `Dockerfile` を作成：

```dockerfile
FROM python:3.12-slim

WORKDIR /app

ENV POETRY_VIRTUALENVS_CREATE=false

RUN pip install --upgrade pip && \
    pip install poetry

COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-root

COPY . .

CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

---

### ⑧ Dockerfile の作成

プロジェクトルートに `Dockerfile` を作成し、以下の内容を記述：

```dockerfile
FROM python:3.12-slim

WORKDIR /app

ENV POETRY_VIRTUALENVS_CREATE=false

RUN pip install --upgrade pip && \
    pip install poetry

COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-root

COPY . .

CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

### ⑨ docker-compose.yml の作成
プロジェクトルートに Dockerfile を作成：
```dockerfile
FROM python:3.12.7-slim

WORKDIR /app

ENV POETRY_VIRTUALENVS_CREATE=false

RUN pip install --upgrade pip && \
    pip install poetry

COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-root

COPY . .

CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

### ➓Dockerで起動
以下のコマンドでFastAPIコンテナをビルド＆起動
```bash
docker compose up --build
```
起動後、ブラウザで http://localhost:8000/docs にアクセスして確認。

## 🔐 JWT 認証の使い方

FastAPIはOAuth2 + JWTを使ったログイン認証をサポートしています。  
以下の手順でトークンの取得と認証付きエンドポイントの利用ができます。

---

### ① ユーザー登録（`/register`）

エンドポイント：`POST /register`  
リクエストボディの例：

```json
{
  "name": "takumi",
  "password": "secretpass"
}
```
## 適宜記述を追加します、、、、