FROM python:3.12.7-slim

WORKDIR /app

ENV POETRY_VIRTUALENVS_CREATE=false

RUN pip install --upgrade pip && \
    pip install poetry

COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-root

COPY . .

CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]