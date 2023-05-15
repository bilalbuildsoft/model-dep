from flask import Flask, jsonify, request
import tensorflow as tf
# import cv2
import numpy as np
from PIL import Image
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
# Load the trained machine learning model
model = tf.keras.models.load_model('N_CAT-Model')

# Define a function to preprocess the image
def preprocess_image(image):
    # print(image)
    imgA = Image.open(image)

    # Resize the image
    img_resized = imgA.resize((256, 256))
    
    img = np.array(img_resized)
    return img

@app.route('/')
def home():
    return "hello"


# Define the API endpoint for receiving image uploads
@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['image']
   
    image = preprocess_image(image_file)

    prediction = ( model.predict(np.array([image])) > 0.070).astype("int32")
    print("Pred: ",prediction)
  
    response = {'result': int(prediction[0])}


    # Return the response to the client
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
