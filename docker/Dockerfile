# M1 Pro MPS対応 Python 3.10環境
FROM --platform=linux/arm64 python:3.10-slim

# 作業ディレクトリの設定
WORKDIR /app

# システムパッケージの更新とインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Pythonパッケージのインストール
RUN pip install --no-cache-dir --upgrade pip

# 基本的な科学計算ライブラリ
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scipy \
    matplotlib \
    seaborn \
    tqdm \
    scikit-learn \
    jupyter \
    ipython

# PyTorch (MPS対応版 - Apple Siliconで動作)
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio

# その他の機械学習・深層学習ライブラリ
RUN pip install --no-cache-dir \
    transformers \
    datasets \
    tensorboard \
    wandb

# 作業ディレクトリをマウントポイントとして設定
VOLUME /app

# デフォルトコマンド
CMD ["/bin/bash"]
