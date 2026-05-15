"""Standalone evaluation: confusion matrix, training curves, classification report."""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="g",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_training_curves(history: dict, output_path: str) -> None:
    epochs = range(1, len(history["accuracy"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["accuracy"], "s-", label="Train")
    ax1.plot(epochs, history["val_accuracy"], "o-", label="Validation")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history["loss"], "s-", label="Train")
    ax2.plot(epochs, history["val_loss"], "o-", label="Validation")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate_model(
    model_path: str,
    data_dir: str,
    output_dir: str = "models/eval",
    class_names: list[str] | None = None,
    architecture: str = "vgg16",
    batch_size: int = 16,
) -> dict:
    """
    Loads model and test set, evaluates, and saves artifacts to output_dir.

    Artifacts:
        {output_dir}/confusion_matrix.png
        {output_dir}/training_curves.png  (if models/history.json exists)
        {output_dir}/classification_report.json
    """
    import tensorflow as tf
    from wildfirevision.data import get_data_generators

    if class_names is None:
        class_names = ["fire", "nofire"]

    os.makedirs(output_dir, exist_ok=True)
    model = tf.keras.models.load_model(model_path)

    _, _, test_data = get_data_generators(
        data_dir=data_dir,
        config={"training": {"batch_size": batch_size}},
        architecture=architecture,
    )

    preds_prob = model.predict(test_data, verbose=1)
    preds = np.argmax(preds_prob, axis=1)
    true = test_data.classes

    report = classification_report(true, preds, target_names=class_names, output_dict=True)
    report_path = os.path.join(output_dir, "classification_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    plot_confusion_matrix(
        true, preds, class_names, os.path.join(output_dir, "confusion_matrix.png")
    )

    history_path = "models/history.json"
    if os.path.isfile(history_path):
        with open(history_path) as f:
            history = json.load(f)
        plot_training_curves(history, os.path.join(output_dir, "training_curves.png"))

    loss, acc = model.evaluate(test_data, verbose=0)
    metrics = {
        "test_loss": float(loss),
        "test_accuracy": float(acc),
        "classification_report": report,
    }

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"test_loss": float(loss), "test_accuracy": float(acc)}, f)

    print(f"Test accuracy: {acc:.4f} | Test loss: {loss:.4f}")
    print(f"Artifacts saved to: {output_dir}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate wildfire detection model")
    parser.add_argument("--model", default="models/model.h5", help="Path to model.h5")
    parser.add_argument("--data-dir", default="data/processed/Classification")
    parser.add_argument("--output-dir", default="models/eval")
    parser.add_argument("--architecture", default="vgg16", choices=["vgg16", "resnet50"])
    args = parser.parse_args()
    evaluate_model(args.model, args.data_dir, args.output_dir, architecture=args.architecture)


if __name__ == "__main__":
    main()
