# CUDA-compatible Python 3.9 environment
FROM --platform=linux/amd64 python:3.9

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

RUN git clone https://github.com/yucrim01/bis-transformer.git /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install --no-cache-dir -e /app

CMD ["sh", "-c", \
    "aws configure set aws_access_key_id ${AWS_ACCESS_KEY_ID} && \
    aws configure set aws_secret_access_key ${AWS_SECRET_ACCESS_KEY} && \
    aws configure set region ${AWS_REGION} && \
    aws configure set output json && \
    echo 'PGPASSWORD=${PGPASSWORD}' >> app/.env && \
    echo 'PGUSER=${PGUSER}' >> app/.env && \
    echo 'PGHOST=${PGHOST}' >> app/.env && \
    echo 'PGPORT=${PGPORT}' >> app/.env && \
    echo 'PGDB=${PGDB}' >> app/.env && \
    echo 'PGPASSWORD_SECRET_NAME=${PGPASSWORD_SECRET_NAME}' >> app/.env && \
    echo 'MLFLOW_PORT=${MLFLOW_PORT}' >> app/.env && \
    echo 'AWS_PROFILE=${AWS_PROFILE}' >> app/.env && \
    echo 'AWS_REGION=${AWS_REGION}' >> app/.env && \
    echo 'AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}' >> app/.env && \
    echo 'S3_BUCKET=${S3_BUCKET}' >> app/.env && \
    echo 'ECS_TASK_CPU=${ECS_TASK_CPU}' >> app/.env && \
    echo 'ECS_TASK_MEMORY=${ECS_TASK_MEMORY}' >> app/.env && \
    echo 'S3_ARTIFACTS_KEY=${S3_ARTIFACTS_KEY}' >> app/.env && \
    echo 'ECS_TASK_CPU=${ECS_TASK_CPU}' >> app/.env && \
    echo 'ECS_TASK_MEMORY=${ECS_TASK_MEMORY}'"]
