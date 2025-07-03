# IBM - Deep Learning with PyTorch (on Coursera)
## Final Assignment : Fashion MNIST Classification with Convolutional Neural Networks (CNN) in PyTorch
This project demonstrates image classification on the **Fashion-MNIST dataset** using Convolutional Neural Networks built with **PyTorch**. The goal is to train a neural network that accurately classifies grayscale images of clothing items into 10 distinct categories.

---
## Overview
The Fashion-MNIST dataset contains 28×28 grayscale images of fashion items (such as shirts, trousers, bags, etc.).  
In this project:
- Images are resized to 16×16 for computational efficiency.
- The dataset is preprocessed and loaded using custom Dataset and DataLoader classes from PyTorch.
- Two types of CNN models are implemented:  
  - A standard CNN  
  - A CNN with Batch Normalization layers to improve training stability and performance.
- The models are trained to predict the correct category out of 10 possible classes.
---
## Project Objectives

- Understand how to preprocess image data for deep learning models.
- Build and compare different CNN architectures.
- Train the models and track the training process (cost and accuracy).
- Visualize training progress to gain insights into model performance.
- Achieve a test accuracy close to or above 88% on the validation dataset.
---

## Key Components

- **Dataset and DataLoader:**  
  - Download and transform the Fashion-MNIST dataset.
  - Resize images and convert them to tensors suitable for model training.

- **Model Architectures:**  
  - **Basic CNN:** A simple two-layer convolutional network followed by fully connected layers.
  - **CNN with Batch Normalization:** An enhanced version of the basic CNN with batch normalization layers to improve convergence.

- **Training Loop:**  
  - Uses Stochastic Gradient Descent (SGD) as the optimizer with a learning rate of 0.1.
  - Cross-Entropy Loss as the loss function.
  - Runs for 5 epochs and tracks both training cost and validation accuracy.

- **Visualization:**  
  - Plots showing cost and accuracy trends over epochs help understand how the model learns.

---
## Results

After training for 5 epochs, the model achieved a test accuracy close to ~88% on the validation data.  
Key results:
- Smooth decrease in training cost across epochs.
- Consistent improvement (or stabilization) in validation accuracy.

---
## Dependencies

To run this project, the following Python libraries are required :
- PyTorch
- torchvision
- matplotlib
- PIL (Python Imaging Library)

---
## Dataset

- **Fashion-MNIST**: Contains images representing 10 classes of clothing items.
- Dataset source: https://github.com/zalandoresearch/fashion-mnist

---
## Visualization

The project includes visual plots:
- **Training cost vs. epoch**
- **Validation accuracy vs. epoch**

These plots help understand the effectiveness of the training process and highlight the impact of model architecture choices.

---

### Course Link : https://www.coursera.org/learn/advanced-deep-learning-with-pytorch
---
### Course Certificate : https://www.coursera.org/account/accomplishments/verify/0RKWKJHJLE54
---
