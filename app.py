import os
import pickle
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import base64

app = Flask(__name__)

# Load scaler and classifier objects
with open('Saved_pickle/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('Saved_pickle/clf.pkl', 'rb') as f:
    clf = pickle.load(f)

app.config['IMAGES_UPLOADS'] = 'static/images_uploaded'

# Define a function to make predictions on uploaded images
def predict_images(files):
    predictions = []
    images = []
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
        image_64_encode = base64.b64encode(image_data).decode('utf-8')
        images.append('data:image/png;base64,' + image_64_encode)

    return predictions, images


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get uploaded images and make predictions
        files = request.files.getlist('files')
        print(files)
        for i in files:
            if i.filename == '':
                return render_template('index.html')
        predictions, images = predict_images(files)

        # Render results template with predictions and images
        return render_template('results.html', predictions=predictions, images=images)

    # Render index template
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
