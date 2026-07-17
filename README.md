# Image Classification of Road Signs

A binary image classifier that detects whether a road scene contains a **stop sign**, built with **transfer learning** on a pretrained ResNet-18. The project covers the full pipeline: data ingestion, preprocessing, training with checkpointing, evaluation, and inference on new images.

## Overview

Training a convolutional network from scratch requires a large labeled dataset and significant compute. This project instead uses ResNet-18 pretrained on ImageNet as a fixed feature extractor: all convolutional layers are frozen, and only the final fully connected layer is replaced and trained on the stop-sign task. This reaches strong validation accuracy with a small dataset and only a few minutes of training.

## Approach

- **Model:** ResNet-18 pretrained on ImageNet, with all feature layers frozen and the final layer replaced to output 2 classes (`stop`, `not_stop`).
- **Preprocessing:** images resized to 224×224 and normalized using standard ImageNet statistics, matching the pretrained model's original training.
- **Data split:** stratified 90/10 train/validation split, organized into class folders and loaded with `ImageFolder`.
- **Training:** SGD with momentum, cross-entropy loss, and a cyclical learning rate scheduler over 10 epochs. The weights from the best-performing epoch are checkpointed and restored at the end, rather than simply using the final epoch.

## Pipeline

1. **Data** — the dataset downloads and extracts automatically from a hosted archive.
2. **Preprocessing** — images are resized, normalized, and split into training and validation sets.
3. **Model setup** — load pretrained ResNet-18, freeze feature layers, swap the classifier head.
4. **Training** — train the classifier while tracking loss and validation accuracy per epoch.
5. **Evaluation** — plot the training loss and validation accuracy curves.
6. **Inference** — load the saved model and classify a new image.

## Stack

Python, PyTorch, torchvision, PIL, NumPy, Matplotlib.
