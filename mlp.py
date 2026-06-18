from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
except Exception:  # pragma: no cover - handled at runtime
    tf = None

try:
    import albumentations as A
except Exception:  # pragma: no cover - handled at runtime
    A = None


DEFAULT_SEED = 42
REQUIRED_COLUMNS = {"image", "age", "gender"}


def _require_tensorflow() -> None:
    if tf is None:
        raise ImportError("TensorFlow is required. Install dependencies and run with `uv run ...`.")


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if tf is not None:
        tf.random.set_seed(seed)


def load_labels(path: str | Path = Path("data/labels.csv")) -> pd.DataFrame:
    labels_path = Path(path)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    df = pd.read_csv(labels_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in labels CSV: {sorted(missing)}")
    return df


def verify_images_exist(df: pd.DataFrame, images_dir: str | Path = Path("data/images")) -> tuple[pd.DataFrame, list[str]]:
    images_path = Path(images_dir)
    resolved = df.copy()
    resolved["image_path"] = resolved["image"].map(lambda name: str((images_path / str(name)).resolve()))

    exists_mask = resolved["image_path"].map(lambda p: Path(p).exists())
    missing_images = resolved.loc[~exists_mask, "image"].astype(str).tolist()
    existing_df = resolved.loc[exists_mask].reset_index(drop=True)
    return existing_df, missing_images


def default_augmentations(seed: int = DEFAULT_SEED) -> Any | None:
    if A is None:
        return None
    random.seed(seed)
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05,
                rotate_limit=10,
                border_mode=0,
                p=0.3,
            ),
        ]
    )


def preprocess_image(
    image_input: str | Path | np.ndarray,
    image_size: tuple[int, int] = (224, 224),
    training: bool = False,
    augmentations: Any | None = None,
) -> np.ndarray:
    from PIL import Image

    if isinstance(image_input, (str, Path)):
        with Image.open(image_input) as img:
            image = np.array(img.convert("RGB"))
    else:
        image = np.asarray(image_input)

    image = tf.image.resize(image, image_size).numpy() if tf is not None else np.array(Image.fromarray(image).resize(image_size))

    if training and augmentations is not None:
        image = augmentations(image=image.astype(np.uint8))["image"]

    image = image.astype(np.float32) / 255.0
    return image


def create_tf_dataset(
    df: pd.DataFrame,
    images_dir: str | Path = Path("data/images"),
    batch_size: int = 32,
    image_size: tuple[int, int] = (224, 224),
    augment: bool = False,
    shuffle: bool = True,
    seed: int = DEFAULT_SEED,
    augmentations: Any | None = None,
):
    _require_tensorflow()

    data = df.copy()
    if "image_path" not in data.columns:
        data["image_path"] = data["image"].map(lambda name: str((Path(images_dir) / str(name)).resolve()))

    paths = data["image_path"].astype(str).to_numpy()
    genders = data["gender"].astype(np.float32).to_numpy()
    ages = data["age"].astype(np.float32).to_numpy()

    ds = tf.data.Dataset.from_tensor_slices((paths, genders, ages))

    def _load(path: tf.Tensor, gender: tf.Tensor, age: tf.Tensor):
        image_bytes = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image_bytes, channels=3)
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32) / 255.0

        if augment:
            aug = augmentations or default_augmentations(seed)

            if aug is not None:
                def _apply_aug(np_image: np.ndarray) -> np.ndarray:
                    out = aug(image=(np_image * 255.0).astype(np.uint8))["image"]
                    return out.astype(np.float32) / 255.0

                image = tf.numpy_function(_apply_aug, [image], tf.float32)
                image.set_shape((*image_size, 3))

        labels = {
            "gender": tf.expand_dims(tf.cast(gender, tf.float32), axis=-1),
            "age": tf.expand_dims(tf.cast(age, tf.float32), axis=-1),
        }
        return image, labels

    if shuffle:
        ds = ds.shuffle(buffer_size=len(data), seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_multitask_model(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    learning_rate: float = 1e-3,
):
    _require_tensorflow()

    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    gender_output = tf.keras.layers.Dense(1, activation="sigmoid", name="gender")(x)
    age_output = tf.keras.layers.Dense(1, activation="linear", name="age")(x)

    model = tf.keras.Model(inputs=inputs, outputs={"gender": gender_output, "age": age_output})
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={"gender": "binary_crossentropy", "age": "mse"},
        metrics={"gender": ["accuracy"], "age": ["mae", "mse"]},
    )
    return model


def train_model(
    model,
    train_ds,
    val_ds=None,
    epochs: int = 2,
    callbacks: list[Any] | None = None,
    seed: int = DEFAULT_SEED,
):
    set_global_seed(seed)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks or [],
        verbose=1,
    )
    return history


def evaluate_model(model, test_ds) -> dict[str, float]:
    gender_true: list[np.ndarray] = []
    age_true: list[np.ndarray] = []

    for _, labels in test_ds:
        gender_true.append(labels["gender"].numpy().reshape(-1))
        age_true.append(labels["age"].numpy().reshape(-1))

    y_gender = np.concatenate(gender_true)
    y_age = np.concatenate(age_true)

    preds = model.predict(test_ds, verbose=0)
    if isinstance(preds, dict):
        pred_gender = preds["gender"].reshape(-1)
        pred_age = preds["age"].reshape(-1)
    else:
        pred_gender = preds[0].reshape(-1)
        pred_age = preds[1].reshape(-1)

    y_gender_hat = (pred_gender >= 0.5).astype(np.int32)
    y_gender_int = y_gender.astype(np.int32)

    accuracy = float((y_gender_hat == y_gender_int).mean())
    tp = float(((y_gender_hat == 1) & (y_gender_int == 1)).sum())
    fp = float(((y_gender_hat == 1) & (y_gender_int == 0)).sum())
    fn = float(((y_gender_hat == 0) & (y_gender_int == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    mae = float(np.mean(np.abs(pred_age - y_age)))
    mse = float(np.mean((pred_age - y_age) ** 2))

    return {
        "gender_accuracy": accuracy,
        "gender_f1": f1,
        "age_mae": mae,
        "age_mse": mse,
    }


def save_model(model, path: str | Path = Path("artifacts/multitask_model.keras")) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    return output


def run_small_demo(
    sample_size: int = 100,
    image_size: tuple[int, int] = (128, 128),
    batch_size: int = 16,
    epochs: int = 2,
    seed: int = DEFAULT_SEED,
) -> dict[str, float]:
    _require_tensorflow()
    from sklearn.model_selection import train_test_split

    set_global_seed(seed)

    labels = load_labels()
    labels = labels.sample(n=min(sample_size, len(labels)), random_state=seed).reset_index(drop=True)
    labels, missing = verify_images_exist(labels)
    if missing:
        print(f"Skipped {len(missing)} missing images in sample")
    if labels.empty:
        raise ValueError("No valid images found in sampled labels.")

    train_df, temp_df = train_test_split(labels, test_size=0.3, random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed)

    train_ds = create_tf_dataset(train_df, batch_size=batch_size, image_size=image_size, augment=True, shuffle=True, seed=seed)
    val_ds = create_tf_dataset(val_df, batch_size=batch_size, image_size=image_size, augment=False, shuffle=False, seed=seed)
    test_ds = create_tf_dataset(test_df, batch_size=batch_size, image_size=image_size, augment=False, shuffle=False, seed=seed)

    model = build_multitask_model(input_shape=(*image_size, 3))
    train_model(model, train_ds, val_ds=val_ds, epochs=epochs, seed=seed)

    metrics = evaluate_model(model, test_ds)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    return metrics


if __name__ == "__main__":
    try:
        run_small_demo()
    except Exception as exc:  # pragma: no cover - convenience for CLI use
        print("Demo run failed:", exc)
        print("Ensure dataset exists under data/labels.csv and data/images/ then run: uv run python mlp.py")
