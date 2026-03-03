import json
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

from .checkpointing import ensure_dir


def _load_mnist() -> Tuple[tf.data.Dataset, tf.data.Dataset, int]:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    batch_size = 64
    train_ds = train_ds.shuffle(10000).batch(batch_size)
    test_ds = test_ds.batch(batch_size)
    train_size = x_train.shape[0]
    return train_ds, test_ds, int(train_size)


def _build_model() -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )


def _is_chief() -> bool:
    """Return True if this process is the chief worker for checkpointing."""
    tf_config = os.getenv("TF_CONFIG")
    if not tf_config:
        return True
    try:
        cfg = json.loads(tf_config)
        task = cfg.get("task", {})
        return task.get("type") == "worker" and int(task.get("index", 0)) == 0
    except Exception:
        return True


def _write_checkpoint_dir_for_worker(checkpoint_dir: str) -> str:
    """In MultiWorkerMirroredStrategy, non-chief workers must write to a
    temporary directory to avoid conflicts. The chief writes to the real dir."""
    if _is_chief():
        return checkpoint_dir
    # Non-chief workers use a temp subdir (TF convention).
    task_dir = os.path.join(checkpoint_dir, "temp_worker")
    ensure_dir(task_dir)
    return task_dir


def run_baseline_training(epochs: int = 5) -> None:
    """Run a baseline MultiWorkerMirroredStrategy training job on MNIST."""
    tf_config = os.getenv("TF_CONFIG")
    if tf_config:
        print("[training] Using TF_CONFIG environment.")
    else:
        print("[training] WARNING: TF_CONFIG not set, running as single-worker.")

    checkpoint_dir = os.getenv("CHECKPOINT_DIR", "/shared/checkpoints")
    ensure_dir(checkpoint_dir)

    strategy = _get_strategy()

    with strategy.scope():
        model = _build_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    train_ds, test_ds, train_size = _load_mnist()

    # Use the built-in ModelCheckpoint callback which is designed for
    # MultiWorkerMirroredStrategy. Each worker calls it, but only the
    # chief actually writes to the real checkpoint_dir.
    write_dir = _write_checkpoint_dir_for_worker(checkpoint_dir)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(write_dir, "ckpt-{epoch:02d}"),
        save_weights_only=True,
    )

    print("[training] Starting training...")
    start_time = time.time()

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs,
        callbacks=[checkpoint_cb],
    )

    wall_time = time.time() - start_time
    total_samples = train_size * epochs
    throughput = total_samples / wall_time if wall_time > 0 else 0.0

    train_acc = history.history.get("accuracy", [None])[-1]
    val_acc = history.history.get("val_accuracy", [None])[-1]

    print("[training] Training complete.")
    print(f"[training] Wall time: {wall_time:.2f}s, throughput: {throughput:.1f} samples/sec (approx).")
    print(f"[training] Final train accuracy: {train_acc}, final val accuracy: {val_acc}")


def _get_strategy() -> tf.distribute.Strategy:
    return tf.distribute.MultiWorkerMirroredStrategy()


def main() -> None:
    epochs = int(os.getenv("EPOCHS", "3"))
    run_baseline_training(epochs=epochs)


if __name__ == "__main__":
    main()
