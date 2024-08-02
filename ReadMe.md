# MNIST Digit Classification with CNN

## Overview

This project involves image classification using Convolutional Neural Networks (CNNs) on the MNIST dataset. The goal is to accurately classify handwritten digits from 0 to 9. The model achieved a validation accuracy of 98%.

## Table of Contents

1. [Dataset](#dataset)
2. [Model Architecture](#model-architecture)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Requirements](#requirements)
6. [Usage](#usage)
7. [Code](#code)
8. [Acknowledgments](#acknowledgments)

## Dataset

The MNIST dataset is a classic benchmark in the field of machine learning. It consists of 70,000 grayscale images of handwritten digits, each of size 28x28 pixels. The dataset is divided into:

- **Training Set**: 60,000 images
- **Validation Set**: 10,000 images

Each image is labeled with the corresponding digit (0-9).

### Source

The MNIST dataset is available from the [Yann LeCun's website](http://yann.lecun.com/exdb/mnist/).

## Model Architecture

The CNN model used for classification includes:

1. **Convolutional Layer**: Applies 32 filters of size 3x3 with 'same' padding to detect features.
2. **Batch Normalization**: Normalizes activations to improve training stability.
3. **Activation Function**: ReLU (Rectified Linear Unit) introduces non-linearity.
4. **MaxPooling Layer**: Downsamples feature maps with 2x2 pooling and strides of 2.
5. **Convolutional Layer**: Applies 64 filters of size 3x3 with 'same' padding.
6. **Batch Normalization**: Normalizes activations.
7. **Activation Function**: ReLU (Rectified Linear Unit).
8. **MaxPooling Layer**: Downsamples feature maps.
9. **Flatten Layer**: Flattens the output from the previous layers.
10. **Dense Layer**: Fully connected layer with 128 units and ReLU activation.
11. **Output Layer**: Dense layer with 10 units and softmax activation for classification.

## Training

The model was trained using the following parameters:

- **Optimizer**: Adam
- **Loss Function**: Categorical Cross-Entropy
- **Batch Size**: 128
- **Epochs**: 20

The training process involves normalizing the dataset, one-hot encoding labels, and validating the model on a separate validation set.

## Evaluation

The model achieved a validation accuracy of **98%**. Performance was measured using accuracy as the primary metric.

## Requirements

The project requires the following Python packages:

- `numpy`
- `tensorflow`

You can install the required packages using pip:

```bash
pip install numpy tensorflow
