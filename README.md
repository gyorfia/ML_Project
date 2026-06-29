# Gender and Age prediction based on images

## Short description
This project is a simple implementation of a gender and age prediction model based on images. 
It uses a convolutional neural network (CNN) to classify the gender and predict the age of a person based on their features.

## How to set up the environment

The project uses `uv` for dependency management. Install dependencies and activate:

```bash
uv sync
source .venv/bin/activate
```

Or run commands directly with `uv`:

```bash
uv run python script.py
```

Register `.venv` as a Jupyter kernel (optional):

```bash
.venv/bin/python -m ipykernel install --user --name=uv --display-name "Python (uv)"
```

## Project setup

The project utilizes the UTKFace dataset containing aligned and cropped face images. The dataset consists of over 23,000 images, each containing annotations for age, gender, and ethnicity.

### Setup Instructions
1. Download the UTKFace dataset from [Kaggle](https://www.kaggle.com/datasets/alifshahariar/utkface-dataset-face-aligned-and-labeled).
2. Place the images in the directory: `data/images/`
3. Place the annotations file `labels.csv` in: `data/labels.csv`

### Dataset CSV Schema
- `image`: The image filename (e.g., `26_0_1_20170116175949583.jpg.chip.jpg`).
- `age`: Integer representing age (0-116).
- `gender`: Binary indicator (0 for Male, 1 for Female).
- `gender_name`: Text label (`Male` or `Female`).
- `race`: Class indicator (0=White, 1=Black, 2=Asian, 3=Indian, 4=Others).
- `race_name`: Text label for ethnicity.

## Files

- [data/](file:///home/gyorfia/source/Python/ML_Project/data): Houses the dataset resources.
  - `images/`: Directory for input face image JPG files.
  - `labels.csv`: Metadata mapping filenames to age, gender, and race labels.
- [models/](file:///home/gyorfia/source/Python/ML_Project/models): Saved model weights and metadata files.
  - `gpu_vx.keras`: Standard Keras model weights file.
  - `gpu_vx.json`: .json for storing history and model evaluation metrics.
- [mlp.py](file:///home/gyorfia/source/Python/ML_Project/mlp.py): Python module containing the dataset pipeline (`CreateTfDataset`), network architecture (`BuildMultitaskModel`), training orchestration (`TwoPhaseTrainingRun`), and evaluation functions (`EvaluateModel`).
- [eda.ipynb](file:///home/gyorfia/source/Python/ML_Project/eda.ipynb): Jupyter notebook containing exploratory data analysis, class distributions, and sample visualizations.
- [evaluation.ipynb](file:///home/gyorfia/source/Python/ML_Project/evaluation.ipynb): Jupyter notebook for analyzing model outputs and evaluating results.
- [pyproject.toml](file:///home/gyorfia/source/Python/ML_Project/pyproject.toml): Lock specification and python dependency declarations.