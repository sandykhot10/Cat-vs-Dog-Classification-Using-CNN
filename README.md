# Cat-vs-Dog-Classification-Using-Convolutional-Neural-Networks
This project focuses on classifying images of cats and dogs using a Convolutional Neural Network (CNN). The model is built with TensorFlow and Keras, and leverages deep learning techniques to accurately distinguish between images of cats and dogs. The project demonstrates the use of CNNs for image classification tasks and includes data preprocess.


<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
    <p>The goal of this project is to develop a model that can accurately classify images of cats and dogs. The model is built using TensorFlow and Keras, and the training and evaluation are performed using Google Colab.</p>
    <h2>Setup Instructions</h2>
    <p>To run this project, you'll need to set up a Google Colab environment and ensure that you have access to the necessary libraries. Follow these steps:</p>
    <h3>1. Clone the Repository</h3>
    <pre><code>!git clone https://github.com/USERNAME/REPOSITORY.git</code></pre>
    <h3>2. Upload Data</h3>
    <p>Upload your dataset (cats and dogs images) to the Colab environment. You can use Google Drive or Colab's file upload feature.</p>
    <h3>3. Install Required Libraries</h3>
    <pre><code>!pip install tensorflow matplotlib</code></pre>
    <h2>Usage</h2>
    <p>To run the notebook, open the `Colab Notebook.ipynb` file in Google Colab and execute the cells sequentially. The notebook contains:</p>
    <ul>
        <li>Data loading and preprocessing</li>
        <li>Model building and training</li>
        <li>Evaluation and predictions</li>
    </ul>
    <h2>Example Code</h2>
    <p>Here's a brief example of how to use the model:</p>
    <pre><code>
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load and preprocess the image
def process(image, label):
    image = tf.cast(image/255., tf.float32)
    return image, label

# Example of loading an image
image_path = '/path/to/your/image.jpg'  # Update with your image path
test_img = Image.open(image_path)
test_img = test_img.resize((256, 256))
test_img = np.array(test_img)
test_img = test_img.astype(np.float32) / 255.0
test_input = test_img.reshape((1, 256, 256, 3))

# Example model
model = keras.Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Load pre-trained weights
model.load_weights('/path/to/your/weights.h5')

# Make a prediction
prediction = model.predict(test_input)
print('Prediction:', 'Cat' if prediction[0] < 0.5 else 'Dog')
    </code></pre>
    <h2>Contributing</h2>
    <p>If you'd like to contribute to this project, please fork the repository and submit a pull request with your changes.</p>
    <h2>License</h2>
    <p>This project is licensed under the MIT License. See the <a href="LICENSE">LICENSE</a> file for details.</p>
</body>
</html>
