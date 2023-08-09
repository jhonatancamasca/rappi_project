FROM python:3.8

COPY src /app/src
COPY requirements.txt /app

RUN mkdir /app/models
RUN mkdir -p /app/data/raw

COPY models/model.pkl /app/models

WORKDIR /app

RUN pip install -r requirements.txt

WORKDIR /app/src

CMD streamlit run app.py