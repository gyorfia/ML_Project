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

## Files
- data/: Contains the dataset used for training and testing the model downloadable from [Kaggle](https://www.kaggle.com/datasets/alifshahariar/utkface-dataset-face-aligned-and-labeled).