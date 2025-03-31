# Lung Cancer Detection Using Deep Learning

This project implements a deep learning-based approach for lung cancer detection using a convolutional neural network (CNN). The model is trained and evaluated on a dataset of lung cancer images, with preprocessing, data augmentation, and K-Fold Cross-Validation to ensure robust performance.

---

## Features
- **Preprocessing**: Images are resized, normalized, and optionally augmented for training.
- **Data Augmentation**: Random transformations such as flipping, rotation, zooming, and contrast adjustments are applied to improve generalization.
- **K-Fold Cross-Validation**: The model is evaluated using 5-fold cross-validation to ensure reliable performance metrics.
- **Class Imbalance Handling**: Class weights are used to address imbalanced datasets.
- **Final Model Training**: The best model is trained on the full dataset for deployment.

---

## Dataset
The dataset contains lung cancer images categorized into the following classes:
1. Adenocarcinoma
2. Large Cell Carcinoma
3. Normal
4. Squamous Cell Carcinoma

The dataset is organized into subdirectories for each class under the `train` directory.

---

## Project Structure
