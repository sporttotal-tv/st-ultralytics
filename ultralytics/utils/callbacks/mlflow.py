# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
MLflow Logging for Ultralytics YOLO.

This module enables MLflow logging for Ultralytics YOLO. It logs metrics, parameters, and model artifacts.
For setting up, a tracking URI should be specified. The logging can be customized using environment variables.

Commands:
    1. To set a project name:
        `export MLFLOW_EXPERIMENT_NAME=<your_experiment_name>` or use the project=<project> argument

    2. To set a run name:
        `export MLFLOW_RUN=<your_run_name>` or use the name=<name> argument

    3. To start a local MLflow server:
        mlflow server --backend-store-uri runs/mlflow
       It will by default start a local server at http://127.0.0.1:5000.
       To specify a different URI, set the MLFLOW_TRACKING_URI environment variable.

    4. To kill all running MLflow server instances:
        ps aux | grep 'mlflow' | grep -v 'grep' | awk '{print $2}' | xargs kill -9
"""

from ultralytics.utils import LOGGER, RUNS_DIR, SETTINGS, TESTS_RUNNING, colorstr
import subprocess

try:
    import os

    assert not TESTS_RUNNING or 'test_mlflow' in os.environ.get('PYTEST_CURRENT_TEST', '')  # do not log pytest
    assert SETTINGS['mlflow'] is True  # verify integration is enabled
    import mlflow

    assert hasattr(mlflow, '__version__')  # verify package is not directory
    from pathlib import Path
    PREFIX = colorstr('MLflow: ')

    from st_commons.tools.convert_annotation_files import convert_ultralytics_prediction_to_bbox_disk

    from datetime import datetime
    import re


except (ImportError, AssertionError):
    mlflow = None


def get_git_commit_hash(data_path):
    original_dir = os.getcwd()
    os.chdir(data_path)

    commit_hash = subprocess.check_output(['git', 'log', '-n', '1', '--pretty=format:%H']).strip().decode()
    os.chdir(original_dir)

    return commit_hash


def add_sport_to_experiment_name(experiment_name, dataset_path):
    sports = ['soccer', 'basketball', 'ice hockey', 'field hockey', 'futsal',
              'volleyball', 'motorsports', 'tennis', 'handball', 'floorball', 'football',
              'baseball', 'golf', 'cricket']

    sports_patterns = {sport: re.compile(re.escape(sport).replace(r'\ ', r'\S*'))
                       for sport in sports}

    for sport, pattern in sports_patterns.items():
        if pattern.search(dataset_path):
            return f"{sport.replace(' ', '')}_{experiment_name}"

    return experiment_name


def on_pretrain_routine_end(trainer):
    """
    Log training parameters to MLflow at the end of the pretraining routine.

    This function sets up MLflow logging based on environment variables and trainer arguments. It sets the tracking URI,
    experiment name, and run name, then starts the MLflow run if not already active. It finally logs the parameters
    from the trainer.

    Args:
        trainer (ultralytics.engine.trainer.BaseTrainer): The training object with arguments and parameters to log.

    Global:
        mlflow: The imported mlflow module to use for logging.

    Environment Variables:
        MLFLOW_TRACKING_URI: The URI for MLflow tracking. If not set, defaults to 'runs/mlflow'.
        MLFLOW_EXPERIMENT_NAME: The name of the MLflow experiment. If not set, defaults to trainer.args.project.
        MLFLOW_RUN: The name of the MLflow run. If not set, defaults to trainer.args.name.
    """
    global mlflow

    uri = os.environ.get('MLFLOW_TRACKING_URI') or str(RUNS_DIR / 'mlflow')
    LOGGER.debug(f'{PREFIX} tracking uri: {uri}')
    mlflow.set_tracking_uri(uri)

    # Set experiment and run names
    experiment_name = os.environ.get('MLFLOW_EXPERIMENT_NAME') or trainer.args.project or '/Shared/YOLOv8'
    experiment_name = add_sport_to_experiment_name(experiment_name, str(trainer.data['path']))
    run_name = os.environ.get('MLFLOW_RUN') or trainer.args.name
    run_name = f'{datetime.now().strftime("%Y%m%d")}_{run_name}'

    mlflow.set_experiment(experiment_name)

    mlflow.autolog()
    try:
        active_run = mlflow.active_run() or mlflow.start_run(run_name=run_name)
        LOGGER.info(f'{PREFIX}logging run_id({active_run.info.run_id}) to {uri}')
        if Path(uri).is_dir():
            LOGGER.info(f"{PREFIX}view at http://127.0.0.1:5000 with 'mlflow server --backend-store-uri {uri}'")
        LOGGER.info(f"{PREFIX}disable with 'yolo settings mlflow=False'")
        mlflow.log_params(dict(trainer.args))
    except Exception as e:
        LOGGER.warning(f'{PREFIX}WARNING ⚠️ Failed to initialize: {e}\n'
                       f'{PREFIX}WARNING ⚠️ Not tracking this run')

    try:
        mlflow.log_param("dataset_path", trainer.data['path'])
    except Exception as e:
        LOGGER.warning(f'{PREFIX}WARNING ⚠️ Failed to log dataset path: {e}\n')

    try:
        mlflow.log_param("dataset_commit", get_git_commit_hash(trainer.data['path']))
    except Exception as e:
        LOGGER.warning(f'{PREFIX}WARNING ⚠️ Failed to log dataset_commit: {e}\n')

    data_artifact_path=os.path.join('dvc_files')
    try:
        mlflow.log_artifact(trainer.data['yaml_file'], artifact_path=data_artifact_path)
    except Exception as e:
        LOGGER.warning(f'{PREFIX}WARNING ⚠️ Failed to save splits .yaml file: {e}\n')

    try:
        for root, _, files in os.walk(trainer.data['path']):
            for file in files:
                if file.endswith('.dvc'):
                    mlflow.log_artifact(os.path.join(root, file),
                                        artifact_path=data_artifact_path)
    except Exception as e:
        LOGGER.warning(f'{PREFIX}WARNING ⚠️ Failed to save .dvc files: {e}\n')

def on_fit_epoch_end(trainer):
    """Log training metrics at the end of each fit epoch to MLflow."""
    if mlflow:
        sanitized_metrics = {k.replace('(', '').replace(')', ''): float(v) for k, v in trainer.metrics.items()}
        mlflow.log_metrics(metrics=sanitized_metrics, step=trainer.epoch)


def on_train_end(trainer):
    """Log model artifacts at the end of the training."""
    if mlflow:
        mlflow.log_artifact(str(trainer.best.parent))  # log save_dir/weights directory with best.pt and last.pt

        predictions_file_path = trainer.save_dir / 'predictions.json'
        predictions_bbox_file_path = trainer.save_dir / 'predictions.bbox'
        if predictions_file_path.exists():
            convert_ultralytics_prediction_to_bbox_disk(
                predictions_file_path,
                predictions_bbox_file_path,
                label_map=None)

        for f in trainer.save_dir.glob('*'):  # log all other files in save_dir
            if f.suffix in {'.png', '.jpg', '.csv', '.pt', '.yaml', '.json', '.bbox'}:
                mlflow.log_artifact(str(f))

        mlflow.end_run()
        LOGGER.info(f'{PREFIX}results logged to {mlflow.get_tracking_uri()}\n'
                    f"{PREFIX}disable with 'yolo settings mlflow=False'")


callbacks = {
    'on_pretrain_routine_end': on_pretrain_routine_end,
    'on_fit_epoch_end': on_fit_epoch_end,
    'on_train_end': on_train_end} if mlflow else {}
