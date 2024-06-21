from ultralytics import YOLO
import torch
import shutil
import os
import mlflow


class TrainingPipeline:
    def train(self):
        
        try:
            # Start an MLflow run
            with mlflow.start_run():
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = YOLO("yolov8x.pt")
                results = model.train(data="data.yaml", epochs=100, imgsz=640, device=device)

                # MLflow Log parameters
                mlflow.log_param("epochs", 100)
                mlflow.log_param("imgsz", 640)
                mlflow.log_param("device", device)

                # Log metrics
                # mlflow.log_metric('train_loss', results[-1].metrics['train_loss'])
                # mlflow.log_metric('val_loss', results[-1].metrics['val_loss'])
                # mlflow.log_metric('mAP', results[-1].metrics['mAP50'])
                
                # Check if the folder exists
                model_path = os.path.join('runs','detect','train','weights', 'best.pt')
                if os.path.exists(model_path):
                    # Log the model
                    # mlflow.log_artifact(model_path)
                    os.makedirs('artifacts/model', exist_ok=True)
                    shutil.copy(model_path, 'artifacts/model')

                if os.path.exists('runs'):
                    shutil.rmtree('runs')

        except Exception as e:
            print(f"An error occurred during training: {e}")
            mlflow.end_run(status='FAILED')
            raise
            # raise e