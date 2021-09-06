FROM python:3.9

RUN pip install -r requirements.txt

COPY . .

RUN ["python3", "main.py"]