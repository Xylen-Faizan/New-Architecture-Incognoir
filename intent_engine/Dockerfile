FROM python:3.11
WORKDIR /app
COPY . .
RUN pip install -r intent_engine/requirements.txt
CMD ["uvicorn", "intent_engine.app:app", "--host", "0.0.0.0", "--port", "8000"]

