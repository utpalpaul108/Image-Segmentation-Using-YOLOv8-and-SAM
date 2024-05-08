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
from src.utils.common import create_directories, decodeImage, encodeImageIntoBase64


class PredictionPipeline:

    def yolov8_detection(self, model, image_path):

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model(image, stream=True)  # generator of Results objects

        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
        
        bbox = boxes.xyxy.tolist()
        bbox = [[int(i) for i in box] for box in bbox]
        return bbox, image


    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
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
            image = cv2.imread(imgpath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            model=YOLO('artifacts/model/best.pt')

            yolov8_boxex, image = self.yolov8_detection(model, imgpath)
            sam_checkpoint = "sam_vit_h_4b8939.pth"
            model_type = "vit_h"
            
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)

            predictor = SamPredictor(sam)
            predictor.set_image(image)
            input_boxes = torch.tensor(yolov8_boxex, device=predictor.device)

            transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

            for i, mask in enumerate(masks):

                binary_mask = masks[i].squeeze().cpu().numpy().astype(np.uint8)

                # Find the contours of the mask
                contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                largest_contour = max(contours, key=cv2.contourArea)

                # Get the new bounding box
                bbox = [int(x) for x in cv2.boundingRect(largest_contour)]

                # Get the segmentation mask for object 
                segmentation = largest_contour.flatten().tolist()

                # Write bounding boxes to file in YOLO format
                with open('BBOX_yolo.txt', 'w') as f:
                    for contour in contours:
                        # Get the bounding box coordinates of the contour
                        x, y, w, h = cv2.boundingRect(contour)
                        # Convert the coordinates to YOLO format and write to file
                        f.write('0 {:.6f} {:.6f} {:.6f} {:.6f}\n'.format((x+w/2)/image.shape[1], (y+h/2)/image.shape[0], w/image.shape[1], h/image.shape[0]))
                        f.write('\n')
                mask=segmentation
                
                # load the image
                #width, height = image_path.size
                img = Image.open(imgpath)
                width, height = img.size

                # convert mask to numpy array of shape (N,2)
                mask = np.array(mask).reshape(-1,2)

                # normalize the pixel coordinates
                mask_norm = mask / np.array([width, height])

                # compute the bounding box
                xmin, ymin = mask_norm.min(axis=0)
                xmax, ymax = mask_norm.max(axis=0)
                bbox_norm = np.array([xmin, ymin, xmax, ymax])

                # concatenate bbox and mask to obtain YOLO format
                yolo = np.concatenate([bbox_norm, mask_norm.reshape(-1)])

                # compute the bounding box
                # write the yolo values to a text file
                with open('yolomask_format.txt', 'w') as f:
                    for val in yolo:
                        f.write("{:.6f} ".format(val))
                    f.write('\n')

                # Print the bounding box and segmentation mask
                print("Bounding box:", bbox)
                #print("Segmentation mask:", segmentation)
                print("yolo",yolo)
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            for mask in masks:
                self.show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
            for box in input_boxes:
                self.show_box(box.cpu().numpy(), plt.gca())
            plt.axis('off')
            plt.savefig('artifacts/prediction/outputImage.jpg')

            opencodedbase64 = encodeImageIntoBase64('artifacts/prediction/outputImage.jpg')
            result = {"image": opencodedbase64.decode('utf-8')}
            return result

        except Exception as e:
            raise e