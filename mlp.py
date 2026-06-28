import json
import random
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.utils.version_utils import training

try:
    import tensorflow as tf
except Exception as exc:
    raise ImportError("TensorFlow is required. Install dependencies and run with `uv sync`.") from exc

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet50_preprocess_input
from tensorflow.keras.models import load_model


DEFAULT_SEED = 42
REQUIRED_COLUMNS = {"image", "age", "gender", "race"}
NUM_ETHNICITY_CLASSES = 5  # 0=white, 1=black, 2=asian, 3=indian, 4=other


# Set global random seeds for reproducibility.
def SetGlobalSeed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# Load the labels CSV and validate the required columns.
def LoadLabels(path: str | Path = Path("data/labels.csv")) -> pd.DataFrame:
    labels_path = Path(path)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    df = pd.read_csv(labels_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in labels CSV: {sorted(missing)}")
    return df


# Resolve image paths and separate existing rows from missing files.
def VerifyImagesExist(df: pd.DataFrame, images_dir: str | Path = Path("data/images")) -> tuple[pd.DataFrame, list[str]]:
    images_path = Path(images_dir)
    resolved = df.copy()
    resolved["image_path"] = resolved["image"].map(lambda name: str((images_path / str(name)).resolve()))

    exists_mask = resolved["image_path"].map(lambda p: Path(p).exists())
    missing_images = resolved.loc[~exists_mask, "image"].astype(str).tolist()
    existing_df = resolved.loc[exists_mask].reset_index(drop=True)
    return existing_df, missing_images

def DefaultAugmentations() -> tf.keras.Sequential:
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomBrightness(factor=0.1),
        tf.keras.layers.RandomContrast(factor=0.1),
        tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        tf.keras.layers.RandomRotation(factor=10/360), # convert to radians
        tf.keras.layers.RandomZoom(height_factor=(-0.05, 0.05))
    ])

# Create a batched tf.data pipeline for multitask training and evaluation.
def CreateTfDataset(
    df: pd.DataFrame,
    images_dir: str | Path = Path("data/images"),
    batch_size: int = 32,
    image_size: tuple[int, int] = (224, 224),
    augment: bool = False,
    shuffle: bool = True,
    seed: int = DEFAULT_SEED,
    augmentations: tf.keras.Sequential | None = None,
):
    data = df.copy()
    if "image_path" not in data.columns:
        data["image_path"] = data["image"].map(lambda name: str((Path(images_dir) / str(name)).resolve()))

    paths = data["image_path"].astype(str).to_numpy()
    genders = data["gender"].astype(np.float32).to_numpy()
    ages = data["age"].astype(np.float32).to_numpy()
    races = data["race"].astype(np.int32).to_numpy()

    ds = tf.data.Dataset.from_tensor_slices((paths, genders, ages, races))

    if augment:
        aug = augmentations or DefaultAugmentations()

    def _LoadAndResize(path: tf.Tensor, gender: tf.Tensor, age: tf.Tensor, race: tf.Tensor):
        image_bytes = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image_bytes, channels=3)
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32)
        return image, gender, age, race

    def _AugmentAndFormat(image: tf.Tensor, gender: tf.Tensor, age: tf.Tensor, race: tf.Tensor):
        if augment:
            image = aug(image, training=True)
            image.set_shape((*image_size, 3))
        image = tf.keras.applications.resnet50.preprocess_input(image)

        labels = {
            "gender": tf.expand_dims(tf.cast(gender, tf.float32), axis=-1),
            "age": tf.expand_dims(tf.cast(age, tf.float32), axis=-1),
            "ethnicity": tf.one_hot(tf.cast(race, tf.int32), NUM_ETHNICITY_CLASSES),
        }
        return image, labels

    ds = ds.map(_LoadAndResize, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()

    if shuffle:
        ds = ds.shuffle(buffer_size=len(data), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(_AugmentAndFormat, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# Build and compile the ResNet50-based multitask model.
def BuildMultitaskModel(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    learning_rate: float = 1e-3,
):
    inputs = tf.keras.Input(shape=input_shape, name="image")
    base_model = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs, pooling="avg")
    base_model.trainable = False

    shared_features = base_model.output

    # --- Gender Branch ---
    gender_branch = tf.keras.layers.Dense(512, name="gender_dense_0")(shared_features)
    gender_branch = tf.keras.layers.BatchNormalization(name="gender_bn_0")(gender_branch)
    gender_branch = tf.keras.layers.Activation("gelu", name="gender_act_0")(gender_branch)
    gender_branch = tf.keras.layers.Dropout(0.4, name="gender_drop_0")(gender_branch)

    gender_branch = tf.keras.layers.Dense(256, name="gender_dense_1")(gender_branch)
    gender_branch = tf.keras.layers.BatchNormalization(name="gender_bn_1")(gender_branch)
    gender_branch = tf.keras.layers.Activation("gelu", name="gender_act_1")(gender_branch)
    gender_branch = tf.keras.layers.Dropout(0.4, name="gender_drop_1")(gender_branch)

    gender_output = tf.keras.layers.Dense(1, activation="sigmoid", name="gender")(gender_branch)

    # --- Age Branch ---
    age_branch = tf.keras.layers.Dense(512, name="age_dense_0")(shared_features)
    age_branch = tf.keras.layers.BatchNormalization(name="age_bn_0")(age_branch)
    age_branch = tf.keras.layers.Activation("gelu", name="age_act_0")(age_branch)
    age_branch = tf.keras.layers.Dropout(0.3, name="age_drop_0")(age_branch)

    age_branch = tf.keras.layers.Dense(256, name="age_dense_1")(age_branch)
    age_branch = tf.keras.layers.BatchNormalization(name="age_bn_1")(age_branch)
    age_branch = tf.keras.layers.Activation("gelu", name="age_act_1")(age_branch)
    age_branch = tf.keras.layers.Dropout(0.3, name="age_drop_1")(age_branch)

    age_branch = tf.keras.layers.Dense(128, name="age_dense_2")(age_branch)
    age_branch = tf.keras.layers.BatchNormalization(name="age_bn_2")(age_branch)
    age_branch = tf.keras.layers.Activation("gelu", name="age_act_2")(age_branch)
    age_branch = tf.keras.layers.Dropout(0.2, name="age_drop_2")(age_branch)

    age_output = tf.keras.layers.Dense(1, activation="linear", name="age")(age_branch)

    # --- Ethnicity Branch ---
    ethnicity_branch = tf.keras.layers.Dense(512, name="ethnicity_dense_0")(shared_features)
    ethnicity_branch = tf.keras.layers.BatchNormalization(name="ethnicity_bn_0")(ethnicity_branch)
    ethnicity_branch = tf.keras.layers.Activation("gelu", name="ethnicity_act_0")(ethnicity_branch)
    ethnicity_branch = tf.keras.layers.Dropout(0.5, name="ethnicity_drop_0")(ethnicity_branch)

    ethnicity_branch = tf.keras.layers.Dense(256, name="ethnicity_dense_1")(ethnicity_branch)
    ethnicity_branch = tf.keras.layers.BatchNormalization(name="ethnicity_bn_1")(ethnicity_branch)
    ethnicity_branch = tf.keras.layers.Activation("gelu", name="ethnicity_act_1")(ethnicity_branch)
    ethnicity_branch = tf.keras.layers.Dropout(0.4, name="ethnicity_drop_1")(ethnicity_branch)

    ethnicity_output = tf.keras.layers.Dense(NUM_ETHNICITY_CLASSES, activation="softmax", name="ethnicity")(
        ethnicity_branch)

    model = tf.keras.Model(inputs=inputs, outputs={"gender": gender_output, "age": age_output, "ethnicity": ethnicity_output})
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={"gender": "binary_crossentropy", "age": tf.keras.losses.Huber(delta=7.0), "ethnicity": "categorical_crossentropy"},
        loss_weights={"gender": 4.0, "age": 0.01, "ethnicity": 1.4},
        metrics={"gender": ["accuracy"], "age": ["mae", "mse"], "ethnicity": ["accuracy"]},
    )
    return model


# Train the model with a fixed seed and optional validation/callbacks.
def TrainModel(
    model,
    train_ds,
    val_ds=None,
    epochs: int = 2,
    callbacks: list[Any] | None = None,
    seed: int = DEFAULT_SEED,
):
    SetGlobalSeed(seed)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks or [],
        verbose=1,
    )
    return history


# Evaluate gender, age, and ethnicity predictions on the test set.
def EvaluateModel(model, test_ds) -> dict[str, Any]:
    gender_true: list[np.ndarray] = []
    age_true: list[np.ndarray] = []
    ethnicity_true: list[np.ndarray] = []

    for _, labels in test_ds:
        gender_true.append(labels["gender"].numpy().reshape(-1))
        age_true.append(labels["age"].numpy().reshape(-1))
        ethnicity_true.append(np.argmax(labels["ethnicity"].numpy(), axis=1))

    y_gender = np.concatenate(gender_true)
    y_age = np.concatenate(age_true)
    y_ethnicity = np.concatenate(ethnicity_true)

    preds = model.predict(test_ds, verbose=0)
    if isinstance(preds, dict):
        pred_gender = preds["gender"].reshape(-1)
        pred_age = preds["age"].reshape(-1)
        pred_ethnicity = preds["ethnicity"]
    else:
        pred_gender = preds[0].reshape(-1)
        pred_age = preds[1].reshape(-1)
        pred_ethnicity = preds[2]

    y_gender_hat = (pred_gender >= 0.5).astype(np.int32) # prediction
    y_gender_int = y_gender.astype(np.int32) # label

    accuracy = float((y_gender_hat == y_gender_int).mean())

    mae = float(np.mean(np.abs(pred_age - y_age)))
    mse = float(np.mean((pred_age - y_age) ** 2))

    pred_ethnicity_hat = np.argmax(pred_ethnicity, axis=1)
    ethnicity_accuracy = float((pred_ethnicity_hat == y_ethnicity).mean())
    cm = confusion_matrix(y_ethnicity, pred_ethnicity_hat)

    return {
        "gender_accuracy": accuracy,
        "age_mae": mae,
        "age_mse": mse,
        "ethnicity_accuracy": ethnicity_accuracy,
        "confusion_matrix": cm,
    }


# Save the trained model plus history and metrics metadata.
def SaveModelAndMetrics(model, history, metrics, base_path: str | Path = "models/multitask_model") -> Path:
    output = Path(base_path).with_suffix(".keras")
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)

    history_data = history.history if hasattr(history, "history") else history

    metrics_data = dict(metrics)
    if isinstance(metrics_data.get("confusion_matrix"), np.ndarray):
        metrics_data["confusion_matrix"] = metrics_data["confusion_matrix"].tolist()

    metadata = {
        "history": history_data,
        "metrics": metrics_data,
    }
    with output.with_name(output.stem + ".json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    return output


# Load the saved model and its JSON metadata sidecar.
def LoadModelAndMetrics(base_path: str | Path) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    model_path = Path(base_path).with_suffix(".keras")
    metadata_path = model_path.with_name(model_path.stem + ".json")
    "return model, history, metrics"

    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Saved metrics/history file not found: {metadata_path}")

    model = load_model(model_path)
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    history = metadata.get("history", {})
    metrics = metadata.get("metrics", {})
    metrics["confusion_matrix"] = np.asarray(metrics["confusion_matrix"])
    return model, history, metrics


# Train and save a model with metrics and history.
def TestRun(
    sample_size: int = None,
    image_size: tuple[int, int] = (128, 128),
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    epochs: int = 2,
    save_path: str | Path = "models/multitask_model",
    seed: int = DEFAULT_SEED,
) -> None:
    from sklearn.model_selection import train_test_split

    SetGlobalSeed(seed)

    labels = LoadLabels()
    sample_size = len(labels) if sample_size is None else sample_size
    labels = labels.sample(n=min(sample_size, len(labels)), random_state=seed).reset_index(drop=True)
    labels, missing = VerifyImagesExist(labels)
    if missing:
        print(f"Skipped {len(missing)} missing images in sample")
    if labels.empty:
        raise ValueError("No valid images found in sampled labels.")

    train_df, temp_df = train_test_split(labels, test_size=0.3, random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed)


    train_ds = CreateTfDataset(train_df, batch_size=batch_size, image_size=image_size, augment=True, shuffle=True, seed=seed)
    val_ds = CreateTfDataset(val_df, batch_size=batch_size, image_size=image_size, augment=False, shuffle=False, seed=seed)
    test_ds = CreateTfDataset(test_df, batch_size=batch_size, image_size=image_size, augment=False, shuffle=False, seed=seed)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # Monitor the overall validation loss
        patience=5,  # Number of epochs with no improvement after which training will be stopped
        min_delta=0.01,  # Minimum change in the monitored quantity to qualify as an improvement
        restore_best_weights=True,
        # Restores model weights from the epoch with the best value of the monitored quantity
        verbose=1  # Prints a message when early stopping is triggered
    )

    model = BuildMultitaskModel(input_shape=(image_size[0], image_size[1], 3), learning_rate=learning_rate)
    history = TrainModel(model, train_ds, val_ds=val_ds, epochs=epochs, callbacks=[early_stopping], seed=seed)
    metrics = EvaluateModel(model, test_ds)
    metrics["sample_size"] = sample_size
    metrics["image_size_x"] = image_size[0]
    metrics["image_size_y"] = image_size[1]
    metrics["batch_size"] = batch_size
    metrics["epochs"] = epochs

    SaveModelAndMetrics(model, history, metrics, base_path=save_path)

    for name, value in metrics.items():
        if isinstance(value, (int, float)): # don't try to print cm
            print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    try:
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        TestRun(image_size=(200, 200), batch_size=64, learning_rate=1e-3, epochs=25, save_path="models/gpu_v3")
    except Exception as exc:
        print("Test run failed:", exc)