import argparse
import os
from random import shuffle
import shutil
import subprocess
import sys
import xml.etree.cElementTree as ET
import re
import cv2
#create txt annotations
# The root directory which holds all information of the dataset.
data_dir = "/data2/datasets-obj"

#for VOC
voc_dir = "VOC/VOCdevkit"
# The directory name which holds the image sets.
imgset_dir = "ImageSets/Main"
# The direcotry which contains the images.
img_dir = "vocImages"
img_ext = "jpg"
# The directory which contains the annotations.
anno_dir = "Annotations"
anno_ext = "xml"
out_dir = "person_data"
anno_out_ext = "txt"

datasets = ["VOC2007", "VOC2012"]
subsets = ["trainval"]
for subset in subsets:
    anno_out_dir = os.path.join(data_dir, out_dir, "txtAnnotations")
    jpg_out_dir = os.path.join(data_dir, out_dir, "JPEGImages")
    if not os.path.exists(anno_out_dir):
        os.makedirs(anno_out_dir)
    for dataset in datasets:
        imgset_file = "{}/{}/{}/{}/{}.txt".format(data_dir, voc_dir, dataset, imgset_dir, subset)
        if not os.path.exists((imgset_file)):
            print(imgset_file, "not exist.")
            continue
        with open(imgset_file, "r") as f:
            for line in f.readlines():
                name = line.strip("\n").split(" ")[0]
                anno_file = "{}/{}/{}/{}/{}.{}".format(data_dir, voc_dir, dataset, anno_dir, name, anno_ext)
                img_file = "{}/{}/{}/{}.{}".format(data_dir, voc_dir, img_dir, name, img_ext)
                if not os.path.exists(anno_file):
                    print(anno_file, "not exist.")
                    continue
                if not os.path.exists(img_file):
                    print(img_file, "not exist.")
                    continue
                im = cv2.imread(img_file)
                if im.shape[2]!=3 or len(im.shape)!=3:
                    print(img_file, "is not rgb image")
                    continue
                xml_tree = ET.parse(anno_file)
                xml_root = xml_tree.getroot()
                anno_out_file = "{}/{}.txt".format(anno_out_dir, name)
                img_out_file = "{}/{}.jpg".format(jpg_out_dir, name)
                for obj in xml_root.findall('object'):
                    cls = obj.find('name').text
                    if cls == "person":
                        cv2.imwrite(img_out_file, im)
                        bndbox = obj.find('bndbox')
                        with open(anno_out_file, "a") as fid:
                            fid.write("%s %s %d %d %d %d\n"%("{}.jpg".format(name), cls, int(bndbox.find('xmin').text), int(bndbox.find('ymin').text), int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)))                               
                            print(cls, img_file)

