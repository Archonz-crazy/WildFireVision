"""Training pipeline with MLflow experiment tracking and HuggingFace Hub upload."""

from __future__ import annotations

import argparse
import json
import os

import mlflow
import tensorflow as tf
import yaml


def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _mlflow_epoch_callback(run):
    """Returns a Keras LambdaCallback that logs metrics to the active MLflow run."""

    def on_epoch_end(epoch, logs):
        mlflow.log_metrics(
            {
                "train_loss": logs.get("loss", 0),
                "train_accuracy": logs.get("accuracy", 0),
                "val_loss": logs.get("val_loss", 0),
                "val_accuracy": logs.get("val_accuracy", 0),
            },
            step=epoch,
        )

    return tf.keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)


def train(config_path: str = "configs/train_config.yaml") -> dict:
    """
    Full training loop with MLflow tracking.

    Fixes from original model_pred_vgg16.py:
    - Training gate bug removed (was gated on model.h5 already existing)
    - Correct VGG16 preprocessing (vgg16.preprocess_input, not resnet's)
    - History saved to JSON so evaluate.py can plot curves independently
    - MLflow tracks every hyperparameter and per-epoch metric
    """
    from wildfirevision.data import get_data_generators
    from wildfirevision.model import build_resnet50_model, build_vgg16_model

    config = _load_config(config_path)
    data_config_path = os.path.join(os.path.dirname(config_path), "data_config.yaml")
    data_config = _load_config(data_config_path)

    architecture = config["model"]["architecture"]
    data_dir = data_config["data_dir"]

    os.makedirs("models", exist_ok=True)
    os.makedirs("models/eval", exist_ok=True)

    train_data, val_data, test_data = get_data_generators(
        data_dir=data_dir,
        config=config,
        architecture=architecture,
    )

    if architecture == "vgg16":
        model = build_vgg16_model(
            input_shape=tuple(config["model"]["input_shape"]),
            num_classes=config["model"]["num_classes"],
            freeze_base=config["model"]["freeze_base"],
        )
    else:
        model = build_resnet50_model(
            input_shape=tuple(config["model"]["input_shape"]),
            num_classes=config["model"]["num_classes"],
            freeze_base=config["model"]["freeze_base"],
        )

    mlflow_cfg = config.get("mlflow", {})
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "./mlruns"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "wildfirevision"))

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "architecture": architecture,
                "epochs": config["training"]["epochs"],
                "batch_size": config["training"]["batch_size"],
                "optimizer": config["training"]["optimizer"],
                "learning_rate": config["training"]["learning_rate"],
                "freeze_base": config["model"]["freeze_base"],
            }
        )

        model_save_path = config["training"]["model_save_path"]
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=config["training"]["early_stopping_patience"],
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True),
            tf.keras.callbacks.TensorBoard(log_dir="mlruns/tensorboard"),
            _mlflow_epoch_callback(run),
        ]

        results = model.fit(
            train_data,
            validation_data=val_data,
            epochs=config["training"]["epochs"],
            callbacks=callbacks,
            verbose=1,
        )

        loss, acc = model.evaluate(test_data, verbose=1)
        mlflow.log_metrics({"test_loss": loss, "test_accuracy": acc})
        mlflow.log_artifact(model_save_path)

        history = results.history
        history_path = config["training"]["history_save_path"]
        with open(history_path, "w") as f:
            json.dump(history, f)
        mlflow.log_artifact(history_path)

        metrics = {"test_loss": loss, "test_accuracy": acc}
        with open("models/metrics.json", "w") as f:
            json.dump(metrics, f)

        print(f"\nTraining complete. Test accuracy: {acc:.4f}, Test loss: {loss:.4f}")
        print(f"Model saved to: {model_save_path}")
        print(f"MLflow run ID: {run.info.run_id}")
        print("View experiments: mlflow ui")

    return history


def push_to_hub(
    model_path: str = "models/model.h5",
    repo_id: str | None = None,
    token: str | None = None,
    config_path: str = "configs/train_config.yaml",
) -> None:
    """Uploads model.h5 to HuggingFace Hub so the Streamlit app can download it."""
    from huggingface_hub import HfApi

    config = _load_config(config_path)
    hf_cfg = config.get("huggingface", {})
    repo_id = repo_id or hf_cfg.get("repo_id", "Archonz-crazy/wildfirevision")
    filename = hf_cfg.get("model_filename", "model.h5")
    token = token or os.environ.get("HF_TOKEN")

    api = HfApi()
    try:
        api.create_repo(repo_id, repo_type="model", exist_ok=True, token=token)
    except Exception:
        pass

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )
    print(f"Model pushed to https://huggingface.co/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train wildfire detection model")
    parser.add_argument(
        "--config", default="configs/train_config.yaml", help="Path to train_config.yaml"
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push model.h5 to HuggingFace Hub after training",
    )
    parser.add_argument("--hf-repo", default=None, help="HuggingFace repo ID override")
    parser.add_argument("--hf-token", default=None, help="HuggingFace API token override")
    args = parser.parse_args()

    train(args.config)

    if args.push_to_hub:
        config = _load_config(args.config)
        model_path = config["training"]["model_save_path"]
        push_to_hub(model_path=model_path, repo_id=args.hf_repo, token=args.hf_token)


if __name__ == "__main__":
    main()
