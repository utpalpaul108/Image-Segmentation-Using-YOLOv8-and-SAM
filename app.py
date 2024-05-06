from flask import Flask, render_template, jsonify, request
from flask_cors import CORS, cross_origin
from src.pipeline.training_pipeline import TrainingPipeline
from src.utils.common import create_directories, decodeImage, encodeImageIntoBase64
from src.pipeline.prediction_pipeline import PredictionPipeline
import os


from ultralytics import YOLO
import numpy as np
import cv2
import sys
import torch
from PIL import Image
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
import shutil


app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/train')
def trainRoute():
    training_pipeline = TrainingPipeline()
    training_pipeline.train()
    return 'Model trained successfully'


@app.route('/predict', methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        def load_image():
            os.makedirs('artifacts/prediction', exist_ok=True)
            imgpath = os.path.join('artifacts/prediction', 'inputImage.jpg')
            image = request.json['image']
            decodeImage(image, imgpath)
            return imgpath
        
        def prediction(imgpath):
            prediction_pipeline = PredictionPipeline()
            result = prediction_pipeline.predict(imgpath)
            return result
        
        imgpath = load_image()
        result = prediction(imgpath)
        return jsonify(result)
    
    except Exception as e:
        raise e

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)