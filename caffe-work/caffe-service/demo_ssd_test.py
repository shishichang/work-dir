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
model_name = 'mobile_ssd'
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

class SSD:
    def __init__(self,
                gpu_id = 4):
        self.prototxt_file = prototxt_file 
        self.model_file = model_file
        self.resized_width = 300
        self.resized_height = 300
        self.top_num_in_video = 5
        self.ssdNet = caffe.Net(prototxt_file, model_file, caffe.TEST)
        self.gpu_id = gpu_id
        caffe.set_mode_gpu()
        caffe.set_device(self.gpu_id)
        self.batch_size = 16
    def getResults(self, blob):
        #print blob.shape
        results = []
        numbox = blob.shape[2]
        if numbox is 0:
            print "no object detected in this image"
            return results, numbox
        for pic_i in range(numbox):
            result = dict()
            result['label'] = blob[0, 0, pic_i, 1].astype(int)
            if result['label'] < 0:
                continue
            result['score'] = blob[0, 0, pic_i, 2]
            result['x1'] = max(blob[0, 0, pic_i, 3], 0)
            result['y1'] = max(blob[0, 0, pic_i, 4], 0)
            result['x2'] = min(blob[0, 0, pic_i, 5], 1)
            result['y2'] = min(blob[0, 0, pic_i, 6], 1)
            result['class'] = classes[result['label']]
            results.append(result)
        return results, len(results)

    def prepare_data(self, img_list):
        img_num = len(img_list)
        img_matrix = np.empty([img_num, 3, self.resized_height, self.resized_width])
        for k, v in enumerate(img_list):
            img = v
            #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            img = cv2.resize(img, (self.resized_height, self.resized_width))
            img = img.astype(float)
            im_data = np.empty([self.resized_height, self.resized_width, 3])  
            if len(img.shape) == 2:
                im_data[:,:,0] = img
                im_data[:,:,1] = img
                im_data[:,:,2] = img
            elif len(img.shape) == 3: 
                im_data = img[:,:,::-1]
            else:
                print('image channel must be 1 or 3')
                raise ValueError
            mean_array = np.array([mean_b, mean_g, mean_r])
            image = im_data - mean_array
            im_data = np.transpose(image, [2, 0, 1])
            img_matrix[k] = im_data
        return img_matrix

    def detect_batch(self, img_list, img_names):

        img_matrix = self.prepare_data(img_list)
        img_idx = 0
        num = len(img_list)
        iters = num/self.batch_size
        mod = np.mod(num, self.batch_size)
        ctg_dict = dict()
        img_objects = []
        if mod != 0:
            iters += 1
        for i in range(iters):
            idx0 = i * self.batch_size
            if i == (iters - 1) and mod != 0:
                idx1 = idx0 + mod
            else:
                idx1 = (i + 1) * self.batch_size
            num_in_batch = idx1 - idx0
            
            batch_input = img_matrix[idx0:idx1]
            self.ssdNet.blobs['data'].reshape(num_in_batch, 3, self.resized_height, self.resized_width)
            self.ssdNet.blobs['data'].data[...] = batch_input
            out = self.ssdNet.forward_all(data=batch_input)
            detections = out['detection_out']
            
            det_idx = 0
            for j in range(num_in_batch):
                frame_obj = dict() 
                frame_obj['img_id'] = img_names[img_idx]
                frame_obj['obj_num'] = 0 
                frame_obj['object_info'] = [] 
                while det_idx < detections.shape[2]:
                    det2img = detections[0, 0, det_idx, 0].astype(int)
                    if det2img == j:
                        obj_info = dict()
                        if detections[0, 0, det_idx, 1] < 0:
                            det_idx += 1
                            continue
                        frame_obj['obj_num'] += 1
                        obj_info['label'] = detections[0, 0, det_idx, 1].astype(int)
                        obj_info['score'] = detections[0, 0, det_idx, 2]
                        obj_info['x1'] = max(detections[0, 0, det_idx, 3], 0.0)
                        obj_info['y1'] = max(detections[0, 0, det_idx, 4], 0.0)
                        obj_info['x2'] = min(detections[0, 0, det_idx, 5], 1.0)
                        obj_info['y2'] = min(detections[0, 0, det_idx, 6], 1.0)
                        obj_info['category'] = classes[obj_info['label']]
                        det_idx += 1
                        frame_obj['object_info'].append(obj_info) 
                        
                        ctg_name = obj_info['category']
                        if ctg_name not in ctg_dict:
                            ctg_dict[ctg_name] = 1
                        else:
                            ctg_dict[ctg_name] += 1

                    else:
                        break
                img_idx += 1
                img_objects.append(frame_obj)
        ctg_details = []
        for (k, v) in Counter(ctg_dict).most_common(self.top_num_in_video):
            ctg_detail = dict()
            ctg_detail['category'] = k
            ctg_detail['times'] = v
#            print(ctg_detail)
            ctg_details.append(ctg_detail)
        results = dict() 
        results['ctg_details'] = ctg_details 
        results['img_objects'] = img_objects
            
        return results             

    def detect_object(self, img_input):
        img = cv2.resize(img_input, (self.resized_height, self.resized_width))
        img = img.astype(float)
        
        im_data = np.empty([self.resized_height, self.resized_width, 3])  
        if len(img.shape) == 2:
            im_data[:,:,0] = img
            im_data[:,:,1] = img
            im_data[:,:,2] = img
        elif len(img.shape) == 3:
            im_data = img[:,:,::-1]
        else:
            print('image channel must be 1 or 3')
            raise ValueError
        mean_array = np.array([mean_b, mean_g, mean_r])
        image = im_data - mean_array
        if 'mobile_ssd' == model_name:
            image = image * 0.017
       
        im_data = np.transpose(image, [2, 0, 1])
        self.ssdNet.blobs['data'].reshape(1, 3, self.resized_width, self.resized_height)
        self.ssdNet.blobs['data'].data[...] = im_data
        out = self.ssdNet.forward()
        
        results,numbox = self.getResults(out['detection_out'])
        return results,numbox
    
    def demo_image(self):
        image_file = './data/000456.jpg'
        print(image_file)
        img = cv2.imread(image_file)
        results, numbox = self.detect_object(img)
        draw = img.copy()
        h, w, _ = draw.shape
        for i in range(numbox):
            cv2.putText(draw,str(results[i]['class']),(int(w*results[i]['x1']),int(h*results[i]['y1'])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
            cv2.rectangle(draw,(int(w*results[i]['x1']),int(h*results[i]['y1'])),(int(w*results[i]['x2']),int(h*results[i]['y2'])),(255,0,0),1)
        fig = plt.figure()
        plt.imshow(draw)
        plt.show()

if __name__ == '__main__':
    ssd = SSD()
    ssd.demo_image()
