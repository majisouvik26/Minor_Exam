FROM python:3.11-slim

WORKDIR /app

COPY question_2/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY question_1/ .

CMD ["python", "train.py"]
