FROM ghcr.io/mlflow/mlflow:latest

RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir psycopg2-binary boto3

EXPOSE 5000

ENV TZ=Asia/Tokyo
ENV ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--allowed-hosts", "*"]