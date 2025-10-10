# BIS Transformer

BIS（Bispectral Index）値を予測するTransformerベースの深層学習モデル

## プロジェクト構造

```
bis-transformer/
├── bin/                    # シェルスクリプト
│   ├── train.sh           # 新規学習開始
│   ├── resume_train.sh    # 学習再開
│   └── run_pipeline.sh    # 完全パイプライン実行
│
├── scripts/               # Pythonスクリプト
│   ├── prepare_data.py    # データ前処理
│   ├── train.py           # 学習
│   ├── evaluate.py        # 評価
│   ├── predict.py         # 予測
│   ├── explain.py         # 説明性分析
│   ├── pipeline.py        # 統合パイプライン
│   └── cleanup_mlflow_runs.py  # MLflow runクリーンアップ
│
├── src/bistransformer/    # ソースコード
│   ├── data/              # データローダー
│   ├── models/            # モデル定義
│   ├── training/          # 学習ループ・最適化
│   └── utils/             # ユーティリティ
│
├── configs/               # Hydra設定ファイル
├── data/                  # データ
├── outputs/               # 学習ログ・チェックポイント
└── mlruns/               # MLflow tracking
```

## クイックスタート

### 1. 環境構築

```bash
# Conda環境作成
conda create -n bit python=3.10 -y
conda activate bit

# 依存関係インストール
pip install -e .
```

### 2. データ準備

```bash
python scripts/prepare_data.py
```

### 3. 学習実行

**新規学習:**
```bash
# シェルスクリプト経由
./bin/train.sh

# または直接Python
python scripts/train.py
```

**学習再開（中断から）:**
```bash
./bin/resume_train.sh
```

**完全パイプライン（Train → Evaluate → Predict → Explain）:**
```bash
./bin/run_pipeline.sh
```

### 4. MLflow UIで結果確認

```bash
mlflow ui
# ブラウザで http://localhost:5000 を開く
```

## 主要コマンド

### Pythonスクリプト

```bash
# データ前処理
python scripts/prepare_data.py

# 学習
python scripts/train.py

# 評価
python scripts/evaluate.py

# 予測
python scripts/predict.py

# 説明性分析
python scripts/explain.py

# 統合パイプライン
python scripts/pipeline.py
python scripts/pipeline.py --skip-train
python scripts/pipeline.py --steps=train,evaluate

# MLflow runクリーンアップ
python scripts/cleanup_mlflow_runs.py
python scripts/cleanup_mlflow_runs.py --status KILLED
```

### シェルスクリプト

```bash
# 新規学習
./bin/train.sh

# 学習再開
./bin/resume_train.sh
./bin/resume_train.sh [run_id]
./bin/resume_train.sh --new-exp

# 完全パイプライン
./bin/run_pipeline.sh
```

## 設定

設定ファイルは`configs/`ディレクトリに配置：

- `configs/config.yaml` - メイン設定
- `configs/data/base.yaml` - データ設定
- `configs/model/transformer.yaml` - モデル設定
- `configs/train/base.yaml` - 学習設定
- `configs/mlflow/local.yaml` - MLflow設定

Hydraのオーバーライド：
```bash
python scripts/train.py train.epochs=50 train.optimizer.lr=1e-4
```

## チェックポイント機能

- **`last.pt`**: 毎エポック保存（中断再開用）
- **`best.pt`**: 最良モデル（推論用）

学習が中断されても自動的に再開可能。

## AWS実行

詳細は `docs/AWS_SETUP.md` を参照

## ライセンス

MIT
