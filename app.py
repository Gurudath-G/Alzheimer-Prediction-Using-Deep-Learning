import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import io
from PIL import Image

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("model.keras")  # Update with your model path

# Define class labels
class_labels = ["COVID-19", "Normal", "Pneumonia"]

# Function to preprocess image
def preprocess_image(img, img_size=(224, 224)):  # Resize to (224, 224)
    img = img.resize(img_size)  # Resize image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
    return img_array

# Function to make prediction
def predict_image(img):
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100
    return class_labels[predicted_class], confidence

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        try:
            img = Image.open(io.BytesIO(file.read()))  # Read image from memory
            prediction, confidence = predict_image(img)
            return jsonify({"prediction": prediction, "confidence": f"{confidence:.2f}%"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
