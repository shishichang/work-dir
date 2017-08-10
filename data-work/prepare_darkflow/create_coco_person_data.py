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
data_dir = "/data2/datasets-obj/person_data"
# The directory which contains the annotations.
anno_dir = "txtAnnotations"
img_dir =  "JPEGImages"
anno_ext = "txt"
img_ext = "jpg"

def find_files(directory, pattern=PATTERN):
    files = []
    for path, d ,filelist in os.walk(directory):
        for filename in filelist:
            if filename.lower().endswith(pattern):
                files.append(filename)
    return files

for i, filename in enumerate(find_files(anno_dir)):
    name = filename.split('.')[0]
    img_file = '{}/{}/{}.{}'.format(data_dir, img_dir, name, img_ext)
    anno_file = '{}/{}/{}.{}'.format(data_dir, anno_dir, name, anno_ext)
    if not os.path.exist(img_file):
        print(img_file, "not exist.")
        continue
    im = cv2.imread(img_file)
    with open(anno_file, 'r') as f_in:
        info = json.load(f_in)
        for i, v in enumerate(info['annotation']):
            xmin = int(v['bbox'][0])
            ymin = int(v['bbox'][1])
            xmax = int(v['bbox'][0] + v['bbox'][2])
            ymax = int(v['bbox'][1] + v['bbox'][3])
            category_id = v['category_id']
            
            label = name_to_labels[str(category_id)]  
            if label_to_classes[str(label)] == "person":
                cv2.imwrite(img_out_file, im)
                with open(anno_out_file, 'a') as f_out:
                    f_out.write("%s %s %d %d %d %d\n"% ("{}.jpg".format(name), label_to_classes[str(label)], xmin, ymin, xmax, ymax))
            else:
                print(name, ".jpg doesn't contain person")
