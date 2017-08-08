import os
from random import shuffle
import sys
import cv2
import re

PATTERN = ('.txt')
def find_files(directory, pattern=PATTERN):
    files = []
    for path, d ,filelist in os.walk(directory):
        for filename in filelist:
            if filename.lower().endswith(pattern):
                files.append(filename)
    return files




data_root = "/data2/datasets-obj"
voc_dir = "VOC/VOCdevkit"

train_list_file = "{}/{}/train.txt".format(data_root, voc_dir)
test_list_file = "{}/{}/test.txt".format(data_root, voc_dir)
test_name_size = "{}/{}/test_name_size.txt".format(data_root, voc_dir)

f_train = open(train_list_file, 'w')
f_test = open(test_list_file, 'w')
f_test_name_size = open(test_name_size, 'w')

anno_voc_train_dir = os.path.join(data_root, voc_dir,  "vocAnnotations/trainval") 
anno_voc_test_dir = os.path.join(data_root, voc_dir, "vocAnnotations/test")
voc_images_dir = os.path.join(data_root,voc_dir, "vocImages")
img_files = [] 
anno_files = []
#generate train list
for i, filename in enumerate(find_files(anno_voc_train_dir)):
    name = filename.split('.')[0]
    img_file = '{}/vocImages/{}.jpg'.format(voc_dir, name)
    anno_file = '{}/vocAnnotations/trainval/{}.txt'.format(voc_dir, name)
    assert os.path.exists(os.path.join(data_root, anno_file)) 
    assert os.path.exists(os.path.join(data_root, img_file))
    img_files.append(img_file)
    anno_files.append(anno_file)

idx = [i for i in xrange(len(img_files))]
shuffle(idx)
for i in idx:
   f_train.write("%s %s\n"%(img_files[i], anno_files[i]))

for i, filename in enumerate(find_files(anno_voc_test_dir)):
    name = filename.split('.')[0]
    img_file = '{}/vocImages/{}.jpg'.format(voc_dir, name)
    anno_file = '{}/vocAnnotations/test/{}.txt'.format(voc_dir, name)
    assert os.path.exists(os.path.join(data_root, anno_file)) 
    assert os.path.exists(os.path.join(data_root, img_file))
    print(img_file)
    f_test.write("%s %s\n"%(img_file, anno_file))
    h, w, _, = (cv2.imread(os.path.join(data_root,img_file))).shape
    f_test_name_size.write("%d %d %d\n"%((i+1, h, w)))



f_train.close()
f_test.close()
f_test_name_size.close()
