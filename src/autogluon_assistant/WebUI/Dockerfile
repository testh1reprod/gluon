FROM python:3.10-slim

WORKDIR /app


RUN pip install -U pip \
    && pip install -U setuptools wheel

RUN  apt-get -y update && \
    apt-get install -y git

COPY . /app

RUN pip install .


EXPOSE 8501

WORKDIR /app/src

CMD ["streamlit", "run", "app.py"]
