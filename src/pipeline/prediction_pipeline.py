from ultralytics import YOLO
import numpy as np
import pandas as pd
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
from src.utils.common import create_directories, decodeImage, encodeImageIntoBase64


class PredictionPipeline:

    def show_mask(self, mask, ax, color):
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
    
    def show_points(self, coords, labels, ax, marker_size=375):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
        
    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 


    def predict(self, imgpath):
        try:
            yolo_model = YOLO('artifacts/model/best.pt')
            image = cv2.imread(imgpath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = yolo_model.predict(source=image)

            boxes_class = np.array([])
            boxes_class_name = np.array([])
            for result in results:
                for c in result.boxes.cls:
                    boxes_class = np.append(boxes_class,int(c))
                    boxes_class_name = np.append(boxes_class_name,yolo_model.names[int(c)])
                

            number_of_classes = len(yolo_model.names)
            boxes_class_color = [np.concatenate([np.random.random(3), np.array([0.7])], axis=0) for x in range(number_of_classes)]
            boxes = result.boxes.xyxy

            df = pd.DataFrame({'class':boxes_class_name, 'bbox':boxes.tolist()})
            df.to_csv('artifacts/prediction/prediction.csv', index=False)

            DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            MODEL_TYPE = "vit_h"
            CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"

            sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
            predictor = SamPredictor(sam)
            predictor.set_image(image)
             
            transformed_boxes = predictor.transform.apply_boxes_torch(boxes, image.shape[:2])
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

            plt.figure(figsize=(10, 10))
            plt.imshow(image)

            # for mask in masks:
            for i, mask in enumerate(masks):
                self.show_mask(mask.cpu().numpy(), plt.gca(), color=boxes_class_color[int(boxes_class[i])])
            for box in boxes:
                self.show_box(box.cpu().numpy(), plt.gca())
            plt.axis('off')
            plt.savefig('artifacts/prediction/outputImage.jpg')
            
            opencodedbase64 = encodeImageIntoBase64('artifacts/prediction/outputImage.jpg')
            result = {"image": opencodedbase64.decode('utf-8')}
            return result

        except Exception as e:
            raise e