<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lung Cancer Detection Using Deep Learning</title>
</head>
<body>
    <h1>Lung Cancer Detection Using Deep Learning</h1>
    <p>
        This project implements a deep learning-based approach for lung cancer detection using a convolutional neural network (CNN). 
        The model is trained and evaluated on a dataset of lung cancer images, with preprocessing, data augmentation, and K-Fold Cross-Validation to ensure robust performance.
    </p>

    <h2>Features</h2>
    <ul>
        <li><strong>Preprocessing:</strong> Images are resized, normalized, and optionally augmented for training.</li>
        <li><strong>Data Augmentation:</strong> Random transformations such as flipping, rotation, zooming, and contrast adjustments are applied to improve generalization.</li>
        <li><strong>K-Fold Cross-Validation:</strong> The model is evaluated using 5-fold cross-validation to ensure reliable performance metrics.</li>
        <li><strong>Class Imbalance Handling:</strong> Class weights are used to address imbalanced datasets.</li>
        <li><strong>Final Model Training:</strong> The best model is trained on the full dataset for deployment.</li>
    </ul>

    <h2>Dataset</h2>
    <p>
        The dataset contains lung cancer images categorized into the following classes:
    </p>
    <ol>
        <li>Adenocarcinoma</li>
        <li>Large Cell Carcinoma</li>
        <li>Normal</li>
        <li>Squamous Cell Carcinoma</li>
    </ol>
    <p>The dataset is organized into subdirectories for each class under the <code>train</code> directory.</p>

    <h2>Project Structure</h2>
    <pre>
.
├── project_augmented3.ipynb   # Main Jupyter Notebook for training and evaluation
├── train/                     # Training dataset (organized by class)
├── macc.keras                 # Saved final model
└── README.md                  # Project documentation
    </pre>

    <h2>Setup Instructions</h2>
    <ol>
        <li>Clone the repository:
            <pre>
git clone &lt;repository-url&gt;
cd &lt;repository-folder&gt;
            </pre>
        </li>
        <li>Install the required dependencies:
            <pre>
pip install -r requirements.txt
            </pre>
        </li>
        <li>Place the dataset in the <code>train/</code> directory, organized by class.</li>
        <li>Run the Jupyter Notebook:
            <pre>
jupyter notebook project_augmented3.ipynb
            </pre>
        </li>
    </ol>

    <h2>Model Architecture</h2>
    <p>The CNN model consists of:</p>
    <ul>
        <li>3 convolutional layers with ReLU activation and max-pooling.</li>
        <li>A fully connected dense layer with 128 neurons and dropout for regularization.</li>
        <li>A softmax output layer for multi-class classification.</li>
    </ul>

    <h2>Training Process</h2>
    <ol>
        <li><strong>Preprocessing:</strong> Images are resized to 256x256 pixels and normalized to the range [0, 1].</li>
        <li><strong>Data Augmentation:</strong> Random transformations (flip, rotation, zoom, contrast) are applied during training.</li>
        <li><strong>K-Fold Cross-Validation:</strong> The dataset is split into 5 folds, and the model is trained and validated on different folds.</li>
        <li><strong>Final Training:</strong> The best model is trained on the entire dataset for 60 epochs.</li>
    </ol>

    <h2>Evaluation</h2>
    <ul>
        <li>The model is evaluated using K-Fold Cross-Validation.</li>
        <li>Metrics:
            <ul>
                <li><strong>Training Loss:</strong> Visualized over epochs.</li>
                <li><strong>Validation Accuracy:</strong> Calculated for each fold and averaged.</li>
            </ul>
        </li>
    </ul>

    <h2>Results</h2>
    <ul>
        <li><strong>Cross-Validation Accuracy:</strong> ~90.5%</li>
        <li><strong>Final Model Accuracy:</strong> Achieved high accuracy on the training dataset.</li>
    </ul>

    <h2>Usage</h2>
    <ol>
        <li>Load the saved model:
            <pre>
from tensorflow.keras.models import load_model
model = load_model('macc.keras')
            </pre>
        </li>
        <li>Predict on new images:
            <pre>
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load and preprocess the image
img = load_img('path_to_image.jpg', target_size=(256, 256), color_mode='grayscale')
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make predictions
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)
print(f"Predicted Class: {predicted_class}")
            </pre>
        </li>
    </ol>

    <h2>Dependencies</h2>
    <ul>
        <li>Python 3.8+</li>
        <li>TensorFlow 2.0+</li>
        <li>NumPy</li>
        <li>Matplotlib</li>
        <li>scikit-learn</li>
    </ul>
    <p>Install dependencies using:
        <pre>
pip install tensorflow numpy matplotlib scikit-learn
        </pre>
    </p>

    <h2>Acknowledgments</h2>
    <ul>
        <li>This project is part of the EE708 course project.</li>
        <li>Special thanks to the dataset contributors and the deep learning community.</li>
    </ul>

    <h2>License</h2>
    <p>This project is licensed under the MIT License. See the <code>LICENSE</code> file for details.</p>
</body>
</html>
