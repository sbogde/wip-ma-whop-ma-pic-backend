# app.py
import os
import sqlite3

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.applications import VGG16, VGG19, EfficientNetB7, InceptionV3, Xception
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions, preprocess_input

from lib.vgg19_prediction import predict_image

app = Flask(__name__)
CORS(app)

def store_prediction(filename, model_name, prediction, confidence):
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO predictions (filename, model_name, prediction, confidence) VALUES (?, ?, ?, ?)
    ''', (filename, model_name, prediction, confidence))
    conn.commit()
    conn.close()

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        print('No image uploaded')
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    filename = os.path.join('uploads', file.filename)
    file.save(filename)

    try:
        preds = predict_image(filename)
        results = [{'label': p[1], 'confidence': float(p[2] * 100)} for p in preds]
        for result in results:
            store_prediction(file.filename, 'vgg19', result['label'], result['confidence'])
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        print(filename, '<---------------')
        # os.remove(filename)
    
    return jsonify({'results': results})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
