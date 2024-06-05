# app.py
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
    
    model_name = request.form.get('model', 'vgg19')  # Default to VGG16 if not specified
    file = request.files['image']
    original_filename = file.filename
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    server_filename = f"{timestamp}_{original_filename}"
    filepath = os.path.join('uploads', server_filename)
    file.save(filepath)

    try:
        if model_name == 'vgg16':
            preds = predict_image_vgg16(filepath)
        else:
            preds = predict_image_vgg19(filepath)
        
        results = [{'label': p[1], 'confidence': float(p[2] * 100)} for p in preds]
        for result in results:
            store_prediction(original_filename, model_name, result['label'], result['confidence'])
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        print(original_filename, '<---------------')
        # os.remove(filename)
    
    return jsonify({
        'model': model_name,
        'results': results
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
