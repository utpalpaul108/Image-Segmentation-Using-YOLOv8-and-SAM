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
import os


class TrainingPipeline:
    def train(self):
        try:
            model = YOLO("artifacts/model/best.pt")
            model.train(data="data.yaml", epochs=100, imgsz=640)
            # Check if the folder exists
            if os.path.exists('runs'):
                # Move best Weight
                shutil.move('runs/detect/train/weights/best.pt', 'artifacts/model')
                # Delete the folder and all its contents
                shutil.rmtree('runs')

        except Exception as e:
            raise e