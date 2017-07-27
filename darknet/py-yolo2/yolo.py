import sys
import glob
import os
from PIL import Image
import numpy as np
from cStringIO import StringIO
import pyDarknet
import base64
import cPickle
import time
import cv2
import matplotlib.pyplot as plt


gpu = 0

cnt = 0

#init detector
pyDarknet.ObjectDetector.set_device(gpu)
#detector = pyDarknet.ObjectDetector('../cfg/tiny-yolo-coco.cfg', '../cfg/tiny-yolo-coco.weights')
detector = pyDarknet.ObjectDetector('../cfg/yolo-voc.cfg', '../cfg/yolo-voc.weights')


im_data = 'data/dog.jpg'

im_org = cv2.imread(im_data)
im = cv2.cvtColor(im_org, cv2.COLOR_BGR2RGBA)

rst, rt = detector.detect_object(im)

plt.figure()
print len(rst)
for i in range(len(rst)):
    cv2.rectangle(im_org, (rst[i].left, rst[i].top), (rst[i].right, rst[i].bottom), (255, 0, 0))
im_org = cv2.cvtColor(im_org, cv2.COLOR_BGR2RGBA)
plt.imshow(im_org)
plt.show() 
cnt += 1

print cnt, rt




