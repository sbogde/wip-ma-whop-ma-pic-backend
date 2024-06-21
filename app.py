# app.py
import numpy as np
import os
import sqlite3
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
# from waitress import serve

# from tensorflow.keras.applications import VGG16, VGG19, EfficientNetB7, InceptionV3, Xception # type: ignore
# from tensorflow.keras.applications import VGG16, VGG19
# from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.applications.efficientnet import EfficientNetB1

from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from tensorflow.keras.applications.imagenet_utils import decode_predictions, preprocess_input # type: ignore

# from lib.vgg16_prediction import predict_image as predict_image_vgg16
# from lib.vgg19_prediction import predict_image as predict_image_vgg19
from lib.mobilenet_prediction import predict_image as predict_image_mobilenet
from lib.efficient_net_b0 import predict_image as predict_image_efficientnet


app = Flask(__name__)
CORS(app)

models = {
    "mobilenet": MobileNet(weights='imagenet'),
    "efficientnetb0": EfficientNetB0(weights='imagenet'),
    "efficientnetb1": EfficientNetB1(weights='imagenet'),
    # "vgg16": VGG16(weights='imagenet'),
    # "vgg19": VGG19(weights='imagenet'),
}

def store_prediction(filename_original, filename_server, model_name, prediction, confidence):
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO predictions (filename_original, filename_server, model_name, prediction, confidence) VALUES (?, ?, ?, ?, ?)
    ''', (filename_original, filename_server, model_name, prediction, confidence))
    conn.commit()
    conn.close()

def preprocess_image(img_path, target_size=(224, 224)):
    img = load_img(img_path, target_size=target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def predict_image(model, model_name, img_path):
    target_size = (224, 224)
    if model_name == "efficientnetb1":
        target_size = (240, 240)

    img = preprocess_image(img_path, target_size)
    preds = model.predict(img)
    decoded_preds = decode_predictions(preds, top=5)[0]
    resized_img = img[0]  # Get the resized image without batch dimension
    return decoded_preds, resized_img

def denormalize_image(img_array):
    img_array = img_array.copy()
    img_array += [123.68, 116.779, 103.939]  # VGG mean values for RGB channels
    img_array = img_array[..., ::-1]  # Convert BGR to RGB
    img_array = np.clip(img_array, 0, 255)
    return img_array

def save_image(img_array, save_path):
    img_array = denormalize_image(img_array)
    img_array = np.uint8(img_array)
    img = Image.fromarray(img_array)
    img.save(save_path)

def initialize_db():
    if not os.path.exists('predictions.db'):
        conn = sqlite3.connect('predictions.db')
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename_original TEXT,
            filename_server TEXT,
            model_name TEXT,
            prediction TEXT,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        conn.commit()
        conn.close()
        print("Created 'predictions.db' database.")

@app.route('/')
def index():
    uploads_exists = os.path.exists('uploads')
    uploads_models_exists = os.path.exists('uploads/models')
    db_exists = os.path.exists('predictions.db')
    return jsonify({
        "message": "Welcome to the Image Classification API!",
        "uploads_exists": uploads_exists,
        "uploads_models_exists": uploads_models_exists,
        "db_exists": db_exists
    })

@app.route('/mkdirs', methods=['POST'])
def create_directories():
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('uploads/models', exist_ok=True)
    return jsonify({"message": "Directories created."})

@app.route('/mkdb', methods=['POST'])
def create_database():
    initialize_db()
    return jsonify({"message": "Database created."})

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        print('No image uploaded')
        return jsonify({'error': 'No image uploaded'}), 400
    
    model_name = request.form.get('model', 'vgg16')  # Default to VGG16 if not specified
    file = request.files['image']
    filename_original = file.filename
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    filename_server = f"{timestamp}_{filename_original}"
    filepath = os.path.join('uploads', filename_server)
    file.save(filepath)
    filename_resized = f"resized_{filename_server}"


    try:
        model = models.get(model_name)
        print(model_name, '<--------------- model_name')

        if model is None:
            return jsonify({'error': 'Model not found'}), 400

        preds, resized_img = predict_image(model, model_name, filepath)

        # Save the resized image
        resized_img_path = os.path.join('uploads/models', filename_resized)
        save_image(resized_img, resized_img_path)

        results = [{'label': p[1], 'confidence': float(p[2] * 100)} for p in preds]
        for result in results:
            store_prediction(filename_original, filename_resized, model_name, result['label'], result['confidence'])
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        print(filename_original, '<---------------')
        print(filename_resized, '<---------------')
        # os.remove(filename)
    
    return jsonify({
        'model': model_name,
        'results': results,
        'resized_image': filename_resized
    })


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

# Ensure the uploads directories and database are created
if os.environ.get('FLASK_ENV') == 'production':
    # Ensure the uploads directories and database are created
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('uploads/models', exist_ok=True)
    print("Created 'uploads' directory.")
    print("Created 'uploads/models' directory.")
    initialize_db()


if __name__ == '__main__':
    # Ensuring the directories are created
    # if not os.path.exists('uploads'):
    #     os.makedirs('uploads')
    #     print("Created 'uploads' directory.")
    # if not os.path.exists('uploads/models'):
    #     os.makedirs('uploads/models')
    #     print("Created 'uploads/models' directory.")

    app.run(debug=True)
    # app.run(host="0.0.0.0", port=5000, debug=True)

    # if os.environ.get('FLASK_ENV') == 'production':
        # serve(app, host='0.0.0.0', port=5000)
    # else:
        # app.run(host="0.0.0.0", port=5000, debug=True)


