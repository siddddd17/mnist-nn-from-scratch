# 2-Layer Neural Network from Scratch for MNIST Digit Recognition

This project implements a two-layer neural network from scratch (using only NumPy) for classifying handwritten digits from the MNIST dataset. The workflow includes data loading, preprocessing, model implementation, training, evaluation, and visualization of results.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Details](#training-details)
- [Results](#results)
- [How to Run](#how-to-run)
- [Limitations and Improvements](#limitations-and-improvements)

---

## Overview

This repository demonstrates a simple feedforward neural network (multi-layer perceptron) built from scratch to recognize digits (0-9) from grayscale 28x28 pixel images. The implementation covers the entire pipeline, including manual forward and backward propagation, weight updates, and performance evaluation without using high-level machine learning libraries.

---

## Dataset

- **Source:** [Kaggle Digit Recognizer](https://www.kaggle.com/c/digit-recognizer)
- **Format:** CSV, with each row representing a flattened 28x28 image (784 pixels) and its label.
- **Preprocessing:** Images are normalized to [0, 1], and the data is split into training and validation sets.

---

## Model Architecture

| Layer            | Details                      |
|------------------|-----------------------------|
| Input            | 784 neurons (28x28 pixels)  |
| Hidden Layer 1   | 64 neurons, ReLU activation |
| Hidden Layer 2   | 64 neurons, ReLU activation |
| Output           | 10 neurons, Softmax         |

- **Loss Function:** Categorical Cross-Entropy
- **Optimizer:** Mini-batch Gradient Descent

---

## Training Details

- **Epochs:** 1000
- **Validation Split:** First 1000 samples held out for validation
- **Learning Rate, Batch Size:** Set in the notebook
- **Shuffling:** Data shuffled before training

---

## Results

### Performance Metrics

- **Final Validation Accuracy:** **97.7%**
- **Initial Validation Accuracy:** 79.9%
- **Training Loss:** Decreased smoothly to near zero

---


## How to Run

1. **Clone the repository** and open the notebook:

```bash
git clone git@github.com:siddddd17/mnist-nn-from-scratch.git
cd mnist-nn-from-scratch
jupyter notebook 2-layer-nueral-network-from-scratch.ipynb
```


2. **Download the dataset** from [Kaggle Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) and place `train.csv` in the notebook directory.

3. **Run all cells** in the notebook.

---

## Limitations and Improvements

- **Current Model:**
- Simple architecture, suitable for educational purposes.
- Achieves strong accuracy for a basic neural network.

- **Potential Improvements:**
- Add convolutional layers (CNN) for better spatial feature extraction.
- Implement regularization (dropout, batch normalization).
- Experiment with optimizers (Adam, RMSprop).
- Data augmentation for robustness.

---


