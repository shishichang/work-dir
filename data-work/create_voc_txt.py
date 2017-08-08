import argparse
import os
from random import shuffle
import shutil
import subprocess
import sys
import xml.etree.cElementTree as ET
import re
#create txt annotations
# The root directory which holds all information of the dataset.
data_dir = "/data2/datasets-obj"
voc_dir = "VOC/VOCdevkit"
# The directory name which holds the image sets.
imgset_dir = "ImageSets/Main"
# The direcotry which contains the images.
img_dir = "JPEGImages"
img_ext = "jpg"
# The directory which contains the annotations.
anno_dir = "Annotations"
anno_ext = "xml"
cache_dir = "vocAnnotations"
anno_out_ext = "txt"

labelmap_file = "labelmap_voc.prototxt" 
pat =  'display_name'
cls_to_labels = dict()
cls_idx = 0
classes = []
with open(labelmap_file, 'r') as f1:
    for line in f1.readlines():
        if re.search(pat, line):
            line_strs = line.split('"')
            class_name = line_strs[-2]
            classes.append(class_name)
            cls_to_labels[class_name] = cls_idx
            cls_idx += 1
datasets = ["VOC2007", "VOC2012"]
# Create training set.
# We follow Ross Girschick's split in R-CNN.
subsets = ["trainval", "test"]
for subset in subsets:
    anno_out_dir = os.path.join(data_dir, voc_dir, cache_dir, subset)
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
                if not os.path.exists(anno_file):
                    print(anno_file, "not exist.")
                    continue
                anno_out_file = "{}/{}.txt".format(anno_out_dir, name)
                with open(anno_out_file, "w") as fid:
                    # Ignore image if it does not have annotation. These are the negative images in ILSVRC.
                    xml_tree = ET.parse(anno_file)
                    xml_root = xml_tree.getroot()
                    for obj in xml_root.findall('object'):
                        cls = obj.find('name').text
                        bndbox = obj.find('bndbox')
                        fid.write("%s %d %d %d %d\n"%(cls_to_labels[cls], int(bndbox.find('xmin').text), int(bndbox.find('ymin').text), int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)))
                        print("%s %d %d %d %d\n"%(cls, int(bndbox.find('xmin').text), int(bndbox.find('ymin').text), int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)))
                            


