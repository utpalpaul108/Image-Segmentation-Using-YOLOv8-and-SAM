from ultralytics import YOLO
import torch
import shutil
import os


class TrainingPipeline:
    def train(self):
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = YOLO("yolov8x.pt")
            model.train(data="data.yaml", epochs=100, imgsz=640, device=device)
            
            # Check if the folder exists
            if os.path.exists('runs/detect/train/weights/best.pt'):
                shutil.copy('runs/detect/train/weights/best.pt', 'artifacts/model')

            if os.path.exists('runs'):
                shutil.rmtree('runs')

        except Exception as e:
            raise e