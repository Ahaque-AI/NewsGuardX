FROM python:3.10

WORKDIR /app

COPY . /app

RUN pip install -r exact-requirements.txt

CMD uvicorn app:app --reload --port=8000 --host=0.0.0.0