FROM python:3.9

RUN pip install numpy

RUN pip install time

COPY . .

RUN ["python3", "main.py"]