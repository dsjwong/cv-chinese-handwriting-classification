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
└── chinese_mnist.csv     # Dataset (pixel values + labels)
```

## How to Run

```bash
pip install tensorflow pandas numpy matplotlib scikit-learn jupyter
jupyter notebook chinese_handwriting_classification.ipynb
```

## Dataset

- **Source:** Chinese MNIST (Kaggle)
- **Images:** 15,000 × 64×64 grayscale
- **Classes:** 15 Chinese numerals (零、一、二、...、十五)
- **Writers:** 100 individuals (150 samples each)
