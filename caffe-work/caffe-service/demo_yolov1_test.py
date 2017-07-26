#!/usr/bin/env python

# --------------------------------------------------------
#allResults contain detections for all images
#results contain detections for an image
#result contains 'label' 'class' 'x1' 'y1' 'x2' 'y2' 
# --------------------------------------------------------
import _init_paths
import numpy as np
import caffe, os, sys, cv2
import time, re
#import pickle
from scipy import misc
from io import BytesIO
from collections import Counter
import matplotlib.pyplot as plt

#classes:
#VOC:   20
#COCO:  80 
#ILSVRC:200
train_set = 'VOC'
#models:
#ssd
#mobilessd
#yolo
model_name = 'yolo1'
data_dir = './data'
prototxt_file = './models/{}_{}_deploy.prototxt'.format(train_set, model_name)
model_file = './models/{}_{}.caffemodel'.format(train_set, model_name)
labelmap_file = './models/{}_labelmap.prototxt'.format(train_set)
classes = []
f = open(labelmap_file, 'r')
pat = 'display_name'
for line in f.readlines():
    if re.search(pat, line):
        line_strs = line.split('"')
        class_name = line_strs[-2]
        classes.append(class_name)
f.close()

NO_OBJECT_IN_PIC = 0
mean_b = 104
mean_g = 117
mean_r = 123

class YOLO1:
    def __init__(self,
                gpu_id = 4):
        self.prototxt_file = prototxt_file 
        self.model_file = model_file
        self.top_num_in_video = 5
        self.yolo1Net = caffe.Net(prototxt_file, model_file, caffe.TEST)
        self.gpu_id = gpu_id
        caffe.set_mode_gpu()
        caffe.set_device(self.gpu_id)
        self.batch_size = 16

        self.threshold = 0.2
        self.iou_threshold = 0.5
        self.num_class = 20
        self.num_box = 2
        self.grid_size = 7
    
    def getResults(self, out_put, img):
        h_img = img.shape[0]
        w_img = img.shape[1]
        print w_img, h_img
        probs = np.zeros((self.grid_size, self.grid_size, self.num_box, self.num_class))
        class_probs = np.reshape(out_put[0:980], (7,7,20))
        scales = np.reshape(out_put[980:1078], (7,7,2))
        bboxes = np.reshape(out_put[1078:], (7,7,2,4))
        offset = np.transpose(np.reshape(np.array([np.arange(7)]*14),(2,7,7)), (1,2,0))
        bboxes[:,:,:,0] += offset
        bboxes[:,:,:,1] += np.transpose(offset, (1,0,2))
        bboxes[:,:,:,0:2] = bboxes[:,:,:,0:2] / 7.0
        bboxes[:,:,:,2] = np.multiply(bboxes[:,:,:,2], bboxes[:,:,:,2]) 
        bboxes[:,:,:,3] = np.multiply(bboxes[:,:,:,3], bboxes[:,:,:,3]) 

        bboxes[:,:,:,0] -= bboxes[:,:,:,2]/2
        bboxes[:,:,:,1] -= bboxes[:,:,:,3]/2
        bboxes[:,:,:,2] += bboxes[:,:,:,0]  
        bboxes[:,:,:,3] += bboxes[:,:,:,1]  
        #bboxes[:,:,:,2] += bboxes[:,:,:,0:1]
        #

        for i in range(2):
            for j in range(20):
                probs[:,:,i,j] = np.multiply(class_probs[:,:,j], scales[:,:,i])
        filter_mat_probs = np.array(probs>=self.threshold, dtype='bool')
        filter_mat_bboxes = np.nonzero(filter_mat_probs)
        bboxes_filtered = bboxes[filter_mat_bboxes[0], filter_mat_bboxes[1], filter_mat_bboxes[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(probs, axis=3)[filter_mat_bboxes[0], filter_mat_bboxes[1], filter_mat_bboxes[2]]

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        bboxes_filtered = bboxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        for i in range(len(bboxes_filtered)):
            if probs_filtered[i] == 0: continue
            for j in range(i+1, len(bboxes_filtered)):
                if iou(bboxes_filtered[i], bboxes_filtered[j]) > iou_threshold:
                    probs_filtered[j] = 0.0
        filter_iou = np.array(probs_filtered>0.0, dtype='bool')
        bboxes_filtered = bboxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]
        print('probs_filtered2.shape')
        print(probs_filtered)
        print(classes_num_filtered)

        results = []
        numbox = len(bboxes_filtered)
        for i in range(len(bboxes_filtered)):
            result = dict()
            result['score'] = probs_filtered[i] 
            result['x1'] = bboxes_filtered[i][0] 
            result['y1'] = bboxes_filtered[i][1] 
            result['x2'] = bboxes_filtered[i][2] 
            result['y2'] = bboxes_filtered[i][3]
            result['class'] = classes[classes_num_filtered[i] + 1]
            results.append(result)
        print results
        return results, numbox
        
    def iou(box1,box2):
    	tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
    	lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
    	if tb < 0 or lr < 0 : intersection = 0
    	else : intersection =  tb*lr
    	return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)

        

    def detect_object(self, input_img):
        #input img is RGB 255
        #img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
        img = input_img
        transformer = caffe.io.Transformer({'data': self.yolo1Net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        inputs = img
        out = self.yolo1Net.forward_all(data=np.asarray([transformer.preprocess('data', inputs)]))
        
        results,numbox = self.getResults(out['connect2'][0], img)
        return results,numbox
    
    def demo_image(self):
        image_file = './data/000456.jpg'
        print(image_file)
        #img = cv2.imread(image_file)
        img = caffe.io.load_image(image_file)
        results, numbox = self.detect_object(img)
        draw = img.copy()
        h, w, _ = draw.shape
        for i in range(numbox):
            cv2.putText(draw,str(results[i]['class']),(int(w*results[i]['x1']),int(h*results[i]['y1'])),cv2.FONT_HERSHEY_SIMPLEX,1,(1,1,0))
            cv2.rectangle(draw,(int(w*results[i]['x1']),int(h*results[i]['y1'])),(int(w*results[i]['x2']),int(h*results[i]['y2'])),(0,1,0),1)
        fig = plt.figure()
        plt.imshow(draw)
        plt.show()

if __name__ == '__main__':
    yolo1 = YOLO1()
    yolo1.demo_image()

