import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template, jsonify, url_for
from werkzeug.utils import secure_filename
import io
from PIL import Image

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = tf.keras.models.load_model("model.keras")  # Update with your model path

# Define class labels
class_labels = ["COVID-19", "Normal", "Pneumonia"]

# Function to preprocess image
def preprocess_image(img, img_size=(224, 224)):
    img = img.resize(img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
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
            return render_template("index.html", error="No file uploaded")
        
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No file selected")

        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            img = Image.open(file_path)
            prediction, confidence = predict_image(img)

            return render_template("index.html",
                                   prediction=prediction,
                                   confidence=f"{confidence:.2f}%",
                                   image_url=url_for('static', filename=f'uploads/{filename}'))
        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
