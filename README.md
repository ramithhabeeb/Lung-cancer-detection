# Lung Cancer Detection Using Deep Learning

## Overview
This project focuses on building a deep learning model to classify lung cancer images into four categories:
1. **Adenocarcinoma**
2. **Large Cell Carcinoma**
3. **Normal**
4. **Squamous Cell Carcinoma**

The model is trained using convolutional neural networks (CNNs) and employs data augmentation, class balancing, and K-Fold Cross-Validation to ensure robust performance.

---

## Dataset
- **Training Data**: The dataset contains 612 labeled images distributed across the four classes.
- **Classes**:
  - Adenocarcinoma
  - Large Cell Carcinoma
  - Normal
  - Squamous Cell Carcinoma

---

## Model Architecture
The CNN model is built using TensorFlow/Keras and consists of:
1. Three convolutional layers with ReLU activation and max-pooling.
2. A fully connected layer with 128 neurons and ReLU activation.
3. A dropout layer (30%) to prevent overfitting.
4. A final softmax layer for multi-class classification.

### Input Shape
- Image size: \(256 \times 256 \times 1\) (grayscale images).

---

## Preprocessing
1. **Data Augmentation**:
   - Random horizontal flipping.
   - Random rotations, zooms, and contrast adjustments.
2. **Normalization**:
   - Pixel values are scaled to the range [0, 1].
3. **Class Balancing**:
   - Class weights are calculated to handle class imbalance.

---

## Training Strategy
1. **K-Fold Cross-Validation**:
   - 5 splits are used to evaluate the model's performance across different subsets of data.
   - Average cross-validation accuracy: ~90.5%.
2. **Callbacks**:
   - Early stopping to halt training when validation loss stops improving.
   - Learning rate reduction on plateau.

---

## Results
| Metric              | Value        |
|---------------------|--------------|
| Training Accuracy   | ~97%         |
| Validation Accuracy | ~90.5% (average across folds) |

---

## Installation and Usage
### Prerequisites
- Python 3.12 or later.
- TensorFlow >= 2.x.
- Required Python libraries: `numpy`, `matplotlib`, `scikit-learn`.

### Steps to Run the Project
1. Clone the repository:
git clone https://github.com/ramithhabeeb/lung-cancer-detection.git
cd lung-cancer-detection
2. Install dependencies:
pip install -r requirements.txt
3. Train the model:
python train_model.py
4. Evaluate the model:
python evaluate_model.p


---

## Visualization
The training loss over epochs is visualized below:

![image](https://github.com/user-attachments/assets/ce0e17a4-845b-4152-adba-ce3cde58da3a)


---

## Model Saving and Deployment
- The trained model is saved as `macc.keras` for future inference.

---

## Future Work
1. Expand the dataset for better generalization.
2. Experiment with advanced architectures like ResNet or EfficientNet.
3. Deploy the model as a web application for real-time predictions.

---

## License
This project is licensed under the MIT License.

---
