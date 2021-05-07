FROM python:3.6

WORKDIR /app
COPY *.py ./
COPY *.pkl ./
COPY pretrained/*.pickle ./pretrained/
COPY requirements-api.txt ./requirements.txt
RUN pip install -r requirements.txt
EXPOSE 10000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]