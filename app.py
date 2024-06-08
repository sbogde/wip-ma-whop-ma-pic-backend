# app.py
import numpy as np
import os
import sqlite3
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
# from tensorflow.keras.applications import VGG16, VGG19, EfficientNetB7, InceptionV3, Xception # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from tensorflow.keras.applications.imagenet_utils import decode_predictions, preprocess_input # type: ignore

from lib.vgg19_prediction import predict_image as predict_image_vgg19
from lib.vgg16_prediction import predict_image as predict_image_vgg16
from PIL import Image

app = Flask(__name__)
CORS(app)

def store_prediction(filename_original, filename_server, model_name, prediction, confidence):
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO predictions (filename_original, filename_server, model_name, prediction, confidence) VALUES (?, ?, ?, ?, ?)
    ''', (filename_original, filename_server, model_name, prediction, confidence))
    conn.commit()
    conn.close()

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
        if model_name == 'vgg16':
            preds, resized_img = predict_image_vgg16(filepath)
        else:
            preds, resized_img = predict_image_vgg19(filepath)

        # Save the resized image
        resized_img_pil = Image.fromarray(np.uint8(resized_img))
        resized_img_path = os.path.join('uploads/models', filename_resized)
        resized_img_pil.save(resized_img_path)

        
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


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('uploads/models', exist_ok=True)

    app.run(debug=True)
