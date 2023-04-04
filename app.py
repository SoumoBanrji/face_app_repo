import os
import pickle
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Load scaler and classifier objects
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('clf.pkl', 'rb') as f:
    clf = pickle.load(f)

# Define a function to make predictions on uploaded images
def predict_images(files):
    predictions = []
    for file in files:
        # Read image and preprocess it
        image_data = file.read()
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (64, 64))
        image = image.flatten()
        image = scaler.transform([image])

        # Make prediction
        prediction = clf.predict(image)
        if prediction == 1:
            prediction_label = 'Acne Detected'
        else:
            prediction_label = 'No Acne Detected'

        # Save prediction and image data as a tuple
        predictions.append(prediction_label)

    return predictions


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get uploaded images and make predictions
        files = request.files.getlist('files')
        predictions = predict_images(files)

        # Render results template with predictions
        return render_template('results.html', predictions=predictions)

    # Render index template
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)

