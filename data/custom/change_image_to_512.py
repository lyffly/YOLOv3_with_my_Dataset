import glob
import cv2

import os



fnames = glob.glob("orgin_image/*.jpg")

for i,name in enumerate(fnames):
    img = cv2.imread(name,1)
    basename = name.split("/")[-1]
    img2 = cv2.resize(img,(512,512))
    cv2.imwrite("images/im_"+str(i)+".jpg",img2)
    print("image ok",i)

print("done")