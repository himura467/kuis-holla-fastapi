## Local Setup

We will be using `pyenv`, `poetry` and `docker`.

### Python installation using pyenv

https://github.com/pyenv/pyenv

```sh
pyenv install `cat .python-version`
```

### Dependencies installation using Poetry

https://python-poetry.org

```sh
poetry config virtualenvs.in-project true
./dev/poetry-all.sh install
```

### Run an ASGI web server using uvicorn

```sh
poetry run uvicorn main:app --reload
```
