from __future__ import annotations

import os
from typing import Dict, Optional, Any
from datetime import datetime

import mlflow
from mlflow.entities import RunStatus
from mlflow.tracking.client import MlflowClient

class MLflowLogger:
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str,
        tags: Optional[Dict[str, str]] = None,
        run_name: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        experiment_name: str
            MLflow experiment name
        tracking_uri: str
            MLflow tracking URI
        tags: dict, optional
            common tags for all steps
        run_name: str, optional
            MLflow run name
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.common_tags = tags or {}
        self.run_name = run_name or datetime.now().strftime("%Y%m%d-%H%M%S")

        self._client = mlflow.MlflowClient(tracking_uri=tracking_uri)
        self._current_run_id: Optional[str] = None
        self._active_run = None

    @property
    def client(self) -> MlflowClient:
        """Get the MLflow client"""
        if self._client is None:
            self._client = MlflowClient(tracking_uri=self.tracking_uri)
        return self._client

    def get_experiment_id(self) -> str:
        """Get the experiment id or create a new experiment"""
        exp = self.client.get_experiment_by_name(self.experiment_name)
        if exp is None:
            exp_id = self.client.create_experiment(self.experiment_name)
        else:
            exp_id = exp.experiment_id
        return exp_id
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Start a new MLflow run

        Parameters
        ----------
        run_name: str, optional
            MLflow run name, if not provided, use the timestamp
        tags: dict, optional
            Tags for the run

        Returns
        -------
        str
            Created run id
        """
        if self._current_run_id:
            raise RuntimeError(
                f"Run {self._current_run_id} is already active."
                "Call `end_run` to finish the run first."
                )
        
        exp_id = self.get_experiment_id()
        run_name = run_name or self.run_name
        run = self.client.create_run(
            experiment_id=exp_id,
            run_name=run_name
        )
        self._current_run_id = run.info.run_id

        for k, v in self.common_tags.items():
            self.client.set_tag(self._current_run_id, k, v)
        
        if tags:
            for k, v in tags.items():
                self.client.set_tag(self._current_run_id, k, v)
        
        return self._current_run_id

    
    def end_run(
        self,
        status: str = RunStatus.to_string(RunStatus.FINISHED)
    ):
        """
        End the current run

        Parameters
        ----------
        status: str
            Run status ("FINISHED", "FAILED", "KILLED")
            default is "FINISHED"
        """
        if self._current_run_id is None:
            raise RuntimeError("No active run to end. Call `start_run` first.")

        self.client.set_terminated(self._current_run_id, status)
        self._current_run_id = None

    def get_run_id(self) -> Optional[str]:
        """Get the current run id"""
        return self._current_run_id
    
    def _ensure_active_run(self):
        """Check if a run is active"""
        if self._current_run_id is None:
            raise RuntimeError("No active run. Call `start_run` first.")

    def log_param(self, key: str, value: Any):
        self._ensure_active_run()
        self.client.log_param(self._current_run_id, key, value)

    def log_params(self, params: Dict[str, Any]):
        self._ensure_active_run()
        for k, v in params.items():
            self.client.log_param(self._current_run_id, k, v)
    
    def log_metric(
        self,
        key: str,
        value: float,
        step: Optional[int] = None,
        timestamp: Optional[int] = None
    ):
        self._ensure_active_run()
        self.client.log_metric(
            self._current_run_id,
            key,
            value,
            timestamp=timestamp or int(datetime.now().timestamp()*1000),
            step=step or 0
        )

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        self._ensure_active_run()
        timestamp = int(datetime.now().timestamp()*1000)
        for k, v in metrics.items():
            self.client.log_metric(
                self._current_run_id,
                k,
                v,
                timestamp=timestamp,
                step=step or 0
            )

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        self._ensure_active_run()
        self.client.log_artifact(self._current_run_id, local_path, artifact_path)

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        self._ensure_active_run()
        self.client.log_artifacts(self._current_run_id, local_dir, artifact_path)

    def log_dict(self, dict: Dict[str, Any], artifact_file: str):
        self._ensure_active_run()
        self.client.log_dict(self._current_run_id, dict, artifact_file)

    def set_tag(self, key: str, value: Any):
        self._ensure_active_run()
        self.client.set_tag(self._current_run_id, key, value)

    def set_tags(self, tags: Dict[str, Any]):
        self._ensure_active_run()
        for k, v in tags.items():
            self.client.set_tag(self._current_run_id, k, v)

    def log_batch(
        self,
        metrics: Dict[str, float],
        params: Dict[str, Any],
        tags: Dict[str, Any]
    ):
        self._ensure_active_run()
        if metrics:
            self.log_metrics(metrics)
        if params:
            self.log_params(params)
        if tags:
            self.set_tags(tags)

class TagManager:
    """Tag managing utility"""
    
    @staticmethod
    def get_common_tags(
        stage: str,
        pipeline_version: str = "v1.0",
        dataset_version: str = "v1",
        purpose: str = "development",
    ) -> Dict[str, Any]:
        """
        Get common tags
        
        Parameters
        ----------
        stage: str
            stage of the run, e.g. "preparation", "training", "evaluation"
        dataset_version: str, optional
            dataset version
        purpose: str, optional
            purpose of the run, e.g. "research", "production"
        """
        import os
        import subprocess

        tags = {
            "pipeline.stage": stage,
            "pipeline.version": pipeline_version,
            "dataset.name": os.environ.get("DATASET_NAME", ""),
            "dataset.version": dataset_version,
            "purpose": purpose,
            "user": os.environ.get("USER", "unknown"),
            "project": os.environ.get("PROJECT", "unknown"),
            "timestamp": datetime.now().strftime("%Y-%m-%d-%H%M%S")
        }

        # get git commit hash
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()[:7]
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()

            tags["git.branch"] = branch
            tags["git.commit"] = commit
        except:
            pass
        
        return tags

    @staticmethod
    def set_pipeline_tags(stage: str, **kwargs):
        tags = TagManager.get_common_tags(stage, **kwargs)
        mlflow.set_tags(tags)
    
class PipelineStage:
    """Stage constants"""
    PREPARE = "prepare"
    TRAIN = "train"
    EVALUATE = "evaluate"
    PREDICT = "predict"
    EXPLAIN = "explain"
    DEPLOY = "deploy"

class Purpose:
    """Purpose constants"""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    EXPERIMENT = "experiment"
    DEBUG = "debug"