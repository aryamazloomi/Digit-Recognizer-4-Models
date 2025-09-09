# Kaggle Digit Recognizer Challenge (>99.5% Accuracy)
## 1. Introduction

Competition: Kaggle Digit Recognizer

Goal: Classify handwritten digits (0–9) from the MNIST dataset using different models.

### Why this project?

Classic ML benchmark.

Good way to compare simple models vs. deep learning architectures.

Shows skills in feature engineering, model selection, evaluation, and explanation.

## 2. Dataset

Source: MNIST (handwritten digit images, 28×28 grayscale).

Train: 42,000 labeled samples.

Test: 28,000 unlabeled samples.

### Preprocessing:

1- Plot data using Matplotlib.pyplot

2- Normalized pixel values (/255.0) for simple MLP models. ( Model 1 and 2)

3- Reshaped to (28, 28, 1) for CNN models.

4- Train/Validation split: 85/15, stratified on labels.

## 3. Models

I experimented with five different approaches:

### M1: Simple MLP (25-15-10)

Architecture: Dense(25, ReLU) -> Dense(15, ReLU) -> Dense(10, SoftMax).

Why: Baseline fully connected model with minimal layers.

### M2: MLP with hyperparameter tuning

Grid search on learning rate (LR) & L2 regularization.

Added early stopping to avoid overfitting.

Why: Shows how parameter choices affect model performance.

### M3: Baseline CNN

Conv layers with pooling and dropout.

Why: CNNs are state-of-the-art for image classification.

### M4: Stronger CNN with BN + Augmentation

BatchNorm, Data Augmentation, AdamW optimizer.

Why: Improves generalization & stability.

### M5: Leaderboard-grade CNN

Deeper conv blocks, Global Average Pooling, dropout.

Why: Approaches >99% accuracy, competitive with Kaggle kernels.

## 4. Training setup

Optimizer: Mostly Adam (AdamW for stronger CNN).

Loss: Sparse categorical crossentropy (integer labels).

Callbacks:

EarlyStopping (patience 10–12).

ReduceLROnPlateau (patience 5–6, factor 0.5).

Batch size: 256.

Epochs: 40–100 (with early stopping).

Regularization: Dropout in CNNs; L2 in tuned MLP.

## 5. Results

### Model	Val Accuracy	Notes

M1 (Simple MLP)	95.44%	Weak baseline

M2 (Tuned MLP)	96.75%	Improved with L2+LR tuning

M3 (Baseline CNN)	99.35%	Huge jump over MLP

M4 (CNN + BN + Aug)	99.60%	Stable, better generalization

M5 (Leaderboard CNN)	99.65%	Highest accuracy (>99%)

### Submission Result

M1 (Simple MLP) 95.328%

M4 (CNN + BN + AUG) 99.532%

Blend (M3 + M4 + M5) 99.582%
