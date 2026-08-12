# Chinese Handwriting Classification (Computer Vision)

Classifies handwritten Chinese characters from the Chinese MNIST dataset using a convolutional neural network or classical ML classifier.

## Overview

Chinese MNIST contains 15,000 images of handwritten Chinese numerals (零 to 拾五) from 100 writers. This project applies a classification pipeline — feature extraction and model training — to recognise these characters from pixel data.

## Tech Stack

- **Language:** Python 3
- **Libraries:** TensorFlow/Keras or scikit-learn, pandas, NumPy, matplotlib
- **Dataset:** `chinese_mnist.csv` (flattened 64×64 pixel images + labels)
- **Environment:** Jupyter Notebook

## Key Concepts

- Loading image data from CSV (flattened pixel rows)
- Reshaping flat arrays into 2D images
- CNN architecture for character recognition
- Multi-class classification (15 classes)
- Confusion matrix and per-class accuracy analysis

## Project Structure

```
cv-chinese-handwriting-classification/
├── chinese_handwriting_classification.ipynb            # Classification notebook
└── chinese_mnist.csv     # Labels only — included in repo. Character images are NOT included, see Dataset section
```

## How to Run

```bash
pip install tensorflow pandas numpy matplotlib scikit-learn jupyter
jupyter notebook chinese_handwriting_classification.ipynb
```

## Dataset

- **Source:** Chinese MNIST (Kaggle: https://www.kaggle.com/datasets/gpreda/chinese-mnist)
- **Images:** 15,000 × 64×64 grayscale
- **Classes:** 15 Chinese numerals (零、一、二、...、十五)
- **Writers:** 100 individuals (150 samples each)

`chinese_mnist.csv` is included in this repository as-is. Credit for the data belongs to the source above; see it for the original license and citation terms.

**This repo is not fully self-contained.** `chinese_mnist.csv` holds only the `suite_id`/`sample_id`/`code`/`value`/`character` label metadata (one row per sample) — it does not contain pixel data. The actual 15,000 handwritten-character images (64×64 grayscale JPEGs, named `input_<suite_id>_<sample_id>_<code>.jpg`) are a separate image directory that ships alongside `chinese_mnist.csv` in the original Kaggle dataset and is **not** included here. To run the notebook end-to-end you must download the full dataset from the source above and place the image folder locally (matching the path the notebook expects) — the CSV alone is not enough to reproduce the pixel-loading steps.

## Environment

Developed and tested with:

- Python 3.9+
- Jupyter Notebook / JupyterLab

Install dependencies:

```bash
pip install -r requirements.txt      # if provided
# or manually: pip install numpy pandas matplotlib scikit-learn torch torchvision
```

Open notebooks in order — each notebook builds on outputs from the previous one.
