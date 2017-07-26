#!/usr/bin/env python

# --------------------------------------------------------
#results contain top 5 predicted scene classes  
# --------------------------------------------------------
import _init_paths
import numpy as np
import caffe, os, sys, cv2
import time, re
import matplotlib.pyplot as plt
#VGGNet_64: 64 scene categories
#googleNet_205: 205 scene categories
#
#model_name = 'VGGNet_64'
model_name = 'googleNet_205'
data_dir = './data'
prototxt_file = '{}_deploy.prototxt'.format(model_name)
model_file = '{}_scene.caffemodel'.format(model_name)
labelmap_file = '{}_categories_places.txt'.format(model_name)
classes = []
f = open(labelmap_file, 'r')
for line in f.readlines():
    line_strs = line.split(' ')
    class_name = line_strs[0]
    classes.append(class_name)
f.close()

NO_OBJECT_IN_PIC = 0
mean_b = 104
mean_g = 117
mean_r = 123

class VGG:
    def __init__(self,
                gpu_id = 3):
        self.prototxt_file = prototxt_file 
        self.model_file = model_file
        self.resized_width = 224
        self.resized_height = 224
        self.vggNet = caffe.Net(prototxt_file, model_file, caffe.TEST)
        self.gpu_id = gpu_id
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
    def getResult(self, blob):
        scores = dict()
        for i in range(blob.shape[1]):
            scores[i] = blob[0, i]
      #  print scores
        sorted_scores = sorted(scores.iteritems(), key = lambda asd:asd[1], reverse=True)
        topNum = 5
        results = []
      #  print "top 5 class labels:"
        for i in range(topNum):
            result = dict()
            result['label'] = sorted_scores[i][0]
            result['score'] = sorted_scores[i][1]
            result['classname'] = classes[sorted_scores[i][0]]
            
            results.append(result)
        return results, len(results)
    def vis_result(self, im, results):
        show_step = 10 
        for i in range(len(results)):
            cv2.putText(im, '{:s} : {:.3f}'.format(results[i]['classname'], results[i]['score']),
                            (im.shape[0]/30, (i+1)*im.shape[1]/30),
                            cv2.FONT_ITALIC, 0.5, (255, 0 ,255255), thickness = 1,  lineType = 2)
        cv2.imshow('classfy result', im)
        cv2.waitKey()

    def classfy_scene(self, img_input):
        img = img_input.copy()
        img = img.astype(float)
        img = cv2.resize(img, (self.resized_height, self.resized_width))
        mean1 = np.ones((1, 1, 3))
        mean1 = mean1.astype(float)
        mean1[0, 0, 0] = mean_b
        mean1[0, 0, 1] = mean_g
        mean1[0, 0, 2] = mean_r
        mean2 = np.tile(mean1, (self.resized_width, self.resized_height, 1))
        img2 = img - mean2
       
        im_data = np.transpose(img2, [2, 0, 1])
        self.vggNet.blobs['data'].reshape(1, 3, self.resized_width, self.resized_height)
        self.vggNet.blobs['data'].data[...] = im_data
        out = self.vggNet.forward()
        results, topNum = self.getResult(out['cls_prob'])
        self.vis_result(img_input, results)
        print "top %d classes: " %topNum
        print results

    def demo(self):
        im_names = ['0.jpg', '1.jpg', '2.jpg',
                    '3.jpg', '4.jpg', '5.jpg']
        for i in range(len(im_names)):
            image_file = 'data/{}'.format(im_names[i])
            im = cv2.imread(image_file)
            self.classfy_scene(im)

if __name__ == '__main__':
    vgg = VGG()
    vgg.demo()
