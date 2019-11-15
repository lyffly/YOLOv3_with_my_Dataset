from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import yaml
import cv2
import numpy as np

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


yaml_config_name ="config.yaml"

def toTensor(img):
    assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(256).unsqueeze(0)  # 255也可以改为256


if __name__ == "__main__":    

    opt = yaml.safe_load(open("config.yaml","r",encoding='utf-8').read())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt["model_def"], img_size=opt["img_size"]).to(device)

    if opt["weights_path"].endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt["weights_path"])
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt["weights_path"]))

    model.eval()  # Set in evaluation mode

    classes = load_classes(opt["class_path"])  # Extracts class labels from file

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("camera is not ready !!!!!")
        exit(0)
    
    while True:
        ret,frame = camera.read()
        if ret is None:
            break
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        image_resized = cv2.resize(image,(opt["img_size"],opt["img_size"]))  
    

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        #imgs = []  # Stores image paths
        #img_detections = []  # Stores detections for each image index

        print("\nPerforming object detection:")
        prev_time = time.time()
        

        # Get detections
        with torch.no_grad():
            input_imgs = Variable(toTensor(image_resized).cuda())

            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt["conf_thres"], opt["nms_thres"])

            # Log progress
            current_time = time.time()
            inference_time = datetime.timedelta(seconds=current_time - prev_time)
            prev_time = current_time
            print("\t+ Inference Time: %s" % (inference_time))

            # Save image and detections  
            imgs = image_resized
            img_detections = detections
            
            print(img_detections)

            cv2.imshow("WWW",imgs)

            cv2.waitKey(1)

"""
        # Bounding-box colors
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]

        print("\nSaving images:")
        # Iterate through images and save plot of detections
        

            

        # Create plot
        img = np.array(image)

        # Draw bounding boxes and labels of detections
        if detections is not None:
        # Rescale boxes to original image
            detections = rescale_boxes(detections, opt["img_size"], img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1,
                    s=classes[int(cls_pred)],
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )

            # Save generated image with detections
  """          
