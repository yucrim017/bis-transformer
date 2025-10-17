# CUDA-compatible Python 3.9 environment
FROM --platform=linux/amd64 python:3.9

WORKDIR /opt/ml/code

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt /opt/ml/code/requirements.txt
RUN pip install --no-cache-dir -r /opt/ml/code/requirements.txt

COPY bis-transformer/src /opt/ml/code/src
COPY bis-transformer/configs /opt/ml/code/configs
COPY bis-transformer/pyproject.toml /opt/ml/code/pyproject.toml

RUN pip install --no-cache-dir -e /opt/ml/code

COPY train_sagemaker.py /opt/ml/code/train_sagemaker.py

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/opt/ml/code:$PYTHONPATH

ENV SAGEMAKER_PROGRAM=train_sagemaker.py
