import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

app = Flask(__name__)

# Load trained model
model = load_model("Blood_Cell.h5")

# Class names (must match TRAIN folder names)
classes = ['Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']


def predict_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    return classes[np.argmax(prediction)]


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    if file:
        img_bytes = file.read()
        result = predict_image(img_bytes)

        return render_template("result.html", prediction=result)

    return "No file uploaded"


if __name__ == "__main__":
    app.run(debug=True)
