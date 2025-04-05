from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load trained model
MODEL_PATH = "alzheimer_model.h5"  # Update with your model path
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels (ensure these match the training dataset order)
class_labels = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")  # Serve frontend

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=(244, 244))  # Change to (244, 244)
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    result = {
        "class": class_labels[predicted_class],
        "confidence": f"{confidence:.2f}%",
        "filepath": filepath
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
