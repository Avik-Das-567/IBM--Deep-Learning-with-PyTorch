# IBM - Deep Learning with PyTorch (Coursera)
---
## Final Assignment : Fashion MNIST Classification with CNNs using PyTorch
This project demonstrates how to build, train, and evaluate a Convolutional Neural Network (CNN) to classify clothing items from the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) using PyTorch.

---

## Project Overview

Fashion-MNIST is a dataset of 28×28 grayscale images across 10 fashion categories, designed as a more challenging alternative to the classic MNIST digit dataset.

In this project:
- The dataset is resized and transformed into tensors.
- Two CNN models are implemented: one with batch normalization and one without.
- Models are trained to classify images into categories such as shirts, coats, sneakers, etc.
- Training performance is tracked using cost and accuracy metrics.
- Visualizations of training progress are created to evaluate performance over multiple epochs.

---

## Dependencies

To run this project, you'll need the following Python libraries:
- torch
- torchvision
- matplotlib
- pillow

---

## Project Structure

- **Dataset Preparation**: Download and preprocess Fashion-MNIST data.
- **Model Definition**: Build CNN architectures with and without batch normalization.
- **Training**: Train models using stochastic gradient descent and cross-entropy loss.
- **Evaluation & Visualization**: Plot cost and accuracy curves to monitor training performance.

---

## Results

After training for 5 epochs:
- The model achieves around **88% validation accuracy**.
- Training cost decreases steadily, demonstrating effective learning.

These results show the strength of CNNs in handling real-world image classification tasks.

---

## Dataset

- **Source**: [Fashion-MNIST by Zalando Research](https://github.com/zalandoresearch/fashion-mnist)
- Contains:
  - 60,000 training images
  - 10,000 test images
- Each image represents a clothing item belonging to one of 10 categories.

---

## Summary

This project is a hands-on implementation of deep learning for image classification using PyTorch.  
It illustrates how convolutional neural networks can be applied effectively on practical datasets beyond the classic MNIST digits.

---

### Course Link : https://www.coursera.org/learn/advanced-deep-learning-with-pytorch
---
### Course Certificate : https://www.coursera.org/account/accomplishments/verify/0RKWKJHJLE54
---
