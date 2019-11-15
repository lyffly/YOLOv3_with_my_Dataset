import glob
import cv2

import os

names = []

for i in range(107):
    names.append("data/custom/images/im_"+str(i)+".jpg\n")

with open("train.txt","w+") as f:
    f.writelines(names)



