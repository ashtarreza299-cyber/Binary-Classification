
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Load the trained model
model = tf.keras.models.load_model("digit_model.keras")

def predict_digit(image_path):
    # Open and convert image to grayscale
    image = Image.open(image_path).convert("L")

    # Resize to match the model's expected input
    image = image.resize((28, 28))

    # Convert image to a NumPy array
    image_array = np.array(image, dtype=np.float32)

    # MNIST-style images usually have a black background
    # and a white digit. Invert if necessary.
    if image_array.mean() > 127:
        image_array = 255 - image_array

    # Normalize pixel values to [0, 1]
    image_array = image_array / 255.0

    # Add the batch dimension: (1, 28, 28) for image input
    # or (1, 784) for a flattened-feature model
    if len(model.input_shape) == 2:
        image_array = image_array.reshape(1, 784)
    else:
        image_array = image_array.reshape(1, 28, 28, 1)

    # Predict
    probabilities = model.predict(image_array, verbose=0)[0]
    predicted_digit = np.argmax(probabilities)
    confidence = probabilities[predicted_digit] * 100

    print(f"Predicted digit: {predicted_digit}")
    print(f"Model confidence: {confidence:.2f}%")

if __name__ == "__main__":
    image_path = "5.png"
    predict_digit(image_path)
