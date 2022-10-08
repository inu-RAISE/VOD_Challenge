import sys
sys.path.insert(0, './yolov5')
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.augmentations import letterbox
from yolov5.utils.torch_utils import select_device, time_sync

import numpy as np
import cv2
import base64
import copy
from glob import glob
import os
import os.path as osp
import logging
import argparse
import torch
import time
from tqdm import tqdm
import copy
from easydict import EasyDict as edict
import datetime

opt = edict({
    "yolo_model" : "/data/IEEE_BigData/sub/weights/car_best.pt",
    "output" : "inference/output",
    "conf_thres" : 0.1,
    "iou_thres" : 0.3,
    "device" : "cuda:0",
    "save_txt" : True,
    "save_vid" : False,
    "show_vid" : False,
    "classes" : [0],
    "agnostic_nms" : True,
    "augment" : False,
    "evaluate" : True,
    "half" : True,
    "visualize" : True,
    "max_det" : 1000,
    "dnn" : True,
    "project" : "runs/track",
    "exist_ok" : True
})

def detect_save(img_list,txt):
    for i in tqdm(range(len(img_list))):
        img = cv2.imread(img_list[i])
        shape = img.shape
        img = letterbox(img, imgsz, stride, auto=pt)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        
        img = torch.from_numpy(img)
        img = img.to(device)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img, augment=True, visualize=False)
        det = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)[0]
        
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], shape)
        
        # Print results
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class

        xyxys = det[:, 0:4]
        confs = det[:, 4]
        clss = det[:, 5]
        
        f_name = img_list[i].split("/")[-1].replace("jpg", "txt")
        
        sentence = f"{f_name}"
        
        for j in range(len(xyxys)):
            sentence = sentence +  f" {dic[str(int(clss[j]))]} {confs[j]} {xyxys[j][0]} {xyxys[j][1]} {xyxys[j][2]} {xyxys[j][3]}"
            
        sentence = sentence + "\n"
        
        txt.write(sentence)
        
    txt.close()

device = select_device(opt.device)
model = DetectMultiBackend(opt.yolo_model, device=device, dnn=opt.dnn, fp16=False)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(640, s=stride)  # check image size
model = model.eval()

imgv1_list = sorted(glob("/data/IEEE_BigData/test/test_v1/*.jpg"))
imgv2_list = sorted(glob("/data/IEEE_BigData/test/test_v2/*.jpg"))

dic = {"0" : "car",
       "1" : "truck",
       "2" : "motorcycle",
       "3" : "bicycle",}

txt_v1 = open("./submission/0913_only_car_v1.txt" ,"w")
txt_v2 = open("./submission/0913_only_car_v2.txt" ,"w")
detect_save(imgv1_list,txt_v1)
detect_save(imgv2_list,txt_v2)

