import os
import sys
import time
import datetime

import yaml
import simplejson as sjson


# sjson.load("jsontest.json")

f = open("yamltest.yaml","w",encoding='utf-8')

data = {
"image_folder":"data/samples",
"model_def":"config/yolov3.cfg",
"weights_path":"weights/yolov3.weights",
"class_path":"data/coco.names",
"conf_thres":0.8,
"nms_thres":0.4,
"batch_size":1,
"n_cpu":0,
"img_size":416,
}

yaml.dump(data,f)

yy = yaml.safe_load(open("yamltest.yaml","r",encoding='utf-8').read())
print(yy)
print(yy['img_size'])


print(data["n_cpu"])
