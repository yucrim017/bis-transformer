"""
Prepare data from VitalDB and log with MLflow
"""
from __future__ import annotations

import sys
import os
from dotenv import load_dotenv
import logging
from pathlib import Path
from typing import List, Dict, Any

import hydra
from omegaconf import DictConfig, OmegaConf

from bistransformer.preparation import VitalDBProcessor
from bistransformer.tracking import (
    MLflowLogger,
    TagManager,
    PipelineStage
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def build_tracks(cfg: DictConfig) -> List[str]:
    """build tracks list from config columns"""
    all_tracks = []
    for k in ["bis", "sqi", "emg", "eeg", "vital", "drug"]:
        val = cfg.get(k, [])
        if val is None:
            continue
        if isinstance(val, str):
            all_tracks.append(val)
        else:
            try:
                all_tracks.extend(list(val))
            except (TypeError, ValueError):
                pass
    return sorted(set(all_tracks))

def build_cols_dict(cfg: DictConfig) -> Dict[str, Any]:
    """build columns dictionary from config"""
    cols_dict = {}
    for k in ["bis", "sqi", "emg", "eeg", "vital", "drug"]:
        val = cfg.get(k, [])
        if val is None:
            continue
        if isinstance(val, str):
            cols_dict[k] = val
        else:
            cols_dict[k] = list(val)
    return cols_dict
    
@hydra.main(version_base=None, config_path="../configs/", config_name="config.yaml")
def main(cfg: DictConfig):
    load_dotenv("aws.env")

    # --- Get environment variables ---
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    logger.info("=" * 80)
    logger.info("Data Preparation Pipeline")
    logger.info("=" * 80)

    mllogger = MLflowLogger(
        experiment_name=cfg.mlflow.experiment.name,
        tracking_uri=mlflow_tracking_uri,
        tags=TagManager.get_common_tags(
            stage=PipelineStage.PREPARE,
            pipeline_version=cfg.mlflow.run.tags.get("pipeline_version", "v1.0"),
            dataset_version=cfg.mlflow.run.tags.get("dataset_version", "v1"),
            purpose=cfg.mlflow.run.tags.get("purpose", "development")
        ),
        run_name=cfg.mlflow.run.get("name", None)
    )

    # --- Start MLflow run ---
    run_id = mllogger.start_run()

    logger.info(f"Started MLflow run: {run_id}")
    logger.info(f"Run name: {mllogger.run_name}")
    logger.info("="*80)

    try:
        logger.info(f"Tracks:\n  {OmegaConf.to_yaml(cfg.data.cols)}")
        logger.info(f"Output directory: {cfg.data.io.output_dir}")
        logger.info(f"Parameters:\n  {OmegaConf.to_yaml(cfg.data.params)}")
        logger.info("="*80)

        output_dir = Path(cfg.data.io.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if next(output_dir.glob("**/*.npz"), None) is not None:
            logger.warning("Output directory is not empty. May Overwriting existing files.")

        tracks = build_tracks(cfg.data.cols)
        logger.info(f"Tracks: {tracks}")

        processor = VitalDBProcessor(
            tracks,
            output_dir=output_dir,
            fs=cfg.data.params.get("fs", 128.0),
            verbose=cfg.data.runtime.get("verbose", True)
        )

        if cfg.data.id.get("cids"):
            cids = list(cfg.data.id.cids)
        else:
            cids = processor.find_cases(cfg.data.id.get("n_cases"))

        logger.info(f"Processing {len(cids)} cases ...")

        cols_dict = build_cols_dict(cfg.data.cols)

        # --- Process cases ---
        results = processor.process_batch(
            cids,
            cols=cols_dict,
            params=OmegaConf.to_container(cfg.data.params, resolve=True),
            save_npz=cfg.data.runtime.get("save_npz", True),
            save_parquet=cfg.data.runtime.get("save_parquet", False)
        )

        manifest = processor.save_manifest(results)
        logger.info(f"Saved manifest: {manifest}")

        # --- Log to MLflow ---
        mllogger.log_dict(
            OmegaConf.to_container(cfg, resolve=True),
            "config/config.json"
        )
        mllogger.log_artifact(
            str(manifest),
            artifact_path=str(cfg.mlflow.artifact_path)
        )

        if hasattr(results, 'iterrows'):
            for i, (idx, row) in enumerate(results.iterrows()):
                if 'processing_time' in row:
                    mllogger.log_metric(
                        "processing_time",
                        row["processing_time"],
                        step=i+1
                    )
                if 'n_features' in row:
                    mllogger.log_metric(
                        "n_features",
                        row["n_features"],
                        step=i+1
                    )
                if 'seconds' in row:
                    mllogger.log_metric(
                        "seconds",
                        row["seconds"],
                        step=i+1
                    )

        mllogger.log_batch(
            metrics={
                "n_total": len(cids),
                "n_processed": len(results),
                "n_failed": len(cids) - len(results),
                "success_rate": len(results) / len(cids) if len(cids) > 0 else 0,
            },
            params=OmegaConf.to_container(cfg.data.params, resolve=True)
        )

        logger.info(f"Processed {len(results)} / {len(cids)} cases successfully")

        # --- End MLflow run ---
        mllogger.end_run()
        logger.info(f"Pipeline completed successfully")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        mllogger.set_tag("error", str(e))
        mllogger.set_tag("error_type", type(e).__name__)
        mllogger.end_run("FAILED")
        raise


if __name__ == "__main__":
    main()