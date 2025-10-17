FROM ghcr.io/mlflow/mlflow:latest

RUN pip install --no-cache-dir psycopg2-binary

ENV TZ=Asia/Tokyo
ENV ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone