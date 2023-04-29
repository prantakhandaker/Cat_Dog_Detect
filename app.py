from flask import Flask, request, jsonify
from PIL import Image
import tensorflow as tf
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model('yolov5/runs/train/exp/weights/best.pt')

# Define the Flask app
app = Flask(__name__)

# Define the API endpoint
@app.route('/detect_cat_dog', methods=['POST'])
def detect_cat_dog():
    # Get the image data from the POST request
    image_file = request.files['image']
    
    # Load the image and preprocess it
    image = Image.open(image_file)
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    
    # Make the prediction
    prediction = model.predict(image)[0]
    
    # Format the output
    output = {
        'class': 'cat' if prediction[0] > prediction[1] else 'dog',
        'confidence': max(prediction),
        'bounding_box': {
            'x1': int(prediction[2]),
            'y1': int(prediction[3]),
            'x2': int(prediction[4]),
            'y2': int(prediction[5])
        }
    }
    
    # Return the output as a JSON object
    return jsonify(output)