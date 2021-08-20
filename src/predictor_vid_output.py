import cv2
import os
os.makedirs("output/video",exist_ok=True)
import os
import glob
classes = ['signal', 'car', 'board', 'sign', 'person']
import numpy as np
import pandas as pd
import torch
import os
import random

import matplotlib.pyplot as plt
from matplotlib import patches
import seaborn as sns
import cv2
import itertools
import copy


import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode
import detectron2.data.transforms as T
from detectron2.data import detection_utils as utils

# MODEL = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'
MODEL = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
def load_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MODEL))
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    # cfg.MODEL.WEIGHTS = os.path.join("output_RUN1", "model_final.pth")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
    cfg.DATASETS.TEST = ('dashcam_test', )
    predictor = DefaultPredictor(cfg)
    return predictor


def annotate_imgs(predictor,img):


    font = cv2.FONT_HERSHEY_SIMPLEX     
    fontScale = 0.3 
    color = (255, 0, 0)
    thickness = 1

    colors = {
        0:(255,0,0),
        1:(0,255,255),
        2:(0,0,255),
        3:(255,255,255),
        4:(255,255,0)
    }
    outputs = predictor(img)
    out = outputs["instances"].to("cpu")
    scores = out.get_fields()['scores'].numpy()
    boxes = out.get_fields()['pred_boxes'].tensor.numpy().astype(int)
    labels= out.get_fields()['pred_classes'].numpy()
    boxes = boxes.astype(int)
    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
    im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    # im /= 255.0

    for b,s,l in zip(boxes,scores,labels):
        cv2.rectangle(im, (b[0],b[1]), (b[0]+b[2],b[1]+b[3]), colors[l], thickness)
        cv2.putText(im, '{}'.format(classes[l]), (b[0],b[1]-3), font, fontScale, colors[l], thickness)
    return im

cap = cv2.VideoCapture('../input/square_vid.mp4')
i=0
predictor = load_model()
counter= 0

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output/output.avi',fourcc, 20.0, (700,700))


while(cap.isOpened()):
  counter+=1
  
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:

    # Display the resulting frame
    # if counter<15000:
    #   continue
    cv2.imshow('frame',frame)
    model_output = annotate_imgs(predictor,frame)
    model_output = cv2.cvtColor(model_output, cv2.COLOR_BGR2RGB)
    model_output = (model_output*1).astype(np.uint8)
    # print(model_output.max())
    cv2.imshow('model_output',model_output)

    out.write(model_output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  else: 
    break
  

# When everything done, release the video capture object
cap.release()
out.release()
cv2.destroyAllWindows()