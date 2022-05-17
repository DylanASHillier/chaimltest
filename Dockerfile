# syntax=docker/dockerfile:1

FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime
# FROM python:3.8-slim-buster
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
