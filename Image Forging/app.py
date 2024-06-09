from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

labels = ['real', 'fake']

# Load the pre-trained model
model = tf.keras.models.load_model('./fake_weights.h5')

# Define a function to classify the uploaded image
def classify_image(file):
    try:
        # Open and preprocess the image
        image = Image.open(file)
        image = image.resize((128, 128))
        image = image.convert("RGB")
        img = np.asarray(image)
        img = np.expand_dims(img, 0)

        # Make predictions using the model
        predictions = model.predict(img)
        label = labels[np.argmax(predictions[0])]
        probability = float(round(predictions[0][np.argmax(predictions[0])]*100, 2))

        result = {
            'label': label,
            'probability': probability
        }

        return result
    except Exception as e:
        # Handle any errors that might occur during image processing
        return {'error': str(e)}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    # Check if 'file' key is present in request.files
    if 'file' not in request.files:
        error_message = "No file part in the request"
        # Return an error response to the client
        return jsonify({'error': error_message}), 400

    # Get the file from the request
    file = request.files['file']

    # Proceed with classification if file is present
    result = classify_image(file)

    # Check if an error occurred during image processing
    if 'error' in result:
        return jsonify(result), 400

    # Return the classification result
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
