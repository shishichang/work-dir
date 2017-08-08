import argparse
import os
from random import shuffle
import shutil
import subprocess
import sys
import json
import re
#create txt annotations
# The root directory which holds all information of the dataset.
data_dir = "/data2/datasets-obj"
# The directory which contains the annotations.
anno_dir = "Annotations"
anno_ext = "json"
cache_dir = "coco_voc_Annotations"
anno_out_ext = "txt"

labelmap_file = "labelmap_coco.prototxt" 
name_to_labels = dict()
cls_idx = 0
classes = []
pat =  ' name'
#build coco labelmap
with open(labelmap_file, 'r') as f1:
    for line in f1.readlines():
        if re.search(pat, line):
            line_strs = line.split('"')
            class_name = line_strs[-2]
            classes.append(class_name)
            name_to_labels[class_name] = cls_idx
            cls_idx += 1
print(name_to_labels)
coco_voc_map = "coco_voc_map.txt"
coco_voc = dict()
with open(coco_voc_map, 'r') as f1:
    for line in f1.readlines():
        if re.search(pat, line):
            line_strs = line.split('"')
            class_name = line_strs[-2]
            classes.append(class_name)
            name_to_labels[class_name] = cls_idx
            cls_idx += 1

PATTERN = ('.json')
def find_files(directory, pattern=PATTERN):
    files = []
    for path, d ,filelist in os.walk(directory):
        for filename in filelist:
            if filename.lower().endswith(pattern):
                files.append(filename)
    return files

dataset = "COCO"
# Create training set.
# We follow Ross Girschick's split in R-CNN.
subset = "train2014"
anno_out_dir = os.path.join(data_dir, dataset, cache_dir, subset)
anno_in_dir = os.path.join(data_dir, dataset, anno_dir, subset)
if not os.path.exists(anno_out_dir):
    os.makedirs(anno_out_dir)

for i, filename in enumerate(find_files(anno_in_dir)):
    name = filename.split('.')[0]
    img_file = '{}/{}/{}/{}.jpg'.format(data_dir, dataset, subset, name)
    anno_file = '{}/{}.json'.format(anno_in_dir, name)
    anno_out_file = '{}/{}.txt'.format(anno_out_dir, name)
    with open(anno_file, 'r') as f_in:
        with open(anno_out_file, 'w') as f_out:
            info = json.load(f_in)
            for i, v in enumerate(info['annotation']):
                xmin = int(v['bbox'][0])
                ymin = int(v['bbox'][1])
                xmax = int(v['bbox'][0] + v['bbox'][2])
                ymax = int(v['bbox'][1] + v['bbox'][3])
                category_id = v['category_id']
                
                label = name_to_labels[str(category_id)]  
                f_out.write("%s %d %d %d %d\n"% (label, xmin, ymin, xmax, ymax))
