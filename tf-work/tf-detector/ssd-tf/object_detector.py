import os
import math
import random
from time import time

import numpy as np
import tensorflow as tf
import cv2
slim = tf.contrib.slim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
sys.path.append('./')
from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from tools import visualization,util
from collections import Counter

class SSD_detector:
    def __init__(self, gpu_id=0, gpu_memory_fraction=0.7, wrap_size=300, data_format='NHWC',
            labelmap_file='data/VOC_labelmap.prototxt', ckpt_file='./checkpoints/ssd_300_vgg.ckpt'):
        os.environ["VISIBLE_CUDA_DEVICES"]='4'

        self.top_num_in_video = 5
        self.gpu_memory_fraction = gpu_memory_fraction
        gpu_options_ssd = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(gpu_options=gpu_options_ssd, log_device_placement=False)
        self.sess = tf.Session(config=sess_config)
        self.class_names=util.map2classnames(labelmap_file)
        self.net_shape = (wrap_size, wrap_size)
        self.data_format = data_format
        self.img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
        self.image_pre, self.labels_pre, self.bboxes_pre, self.bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(self.img_input, None, None, self.net_shape, self.data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
        self.image_4d = tf.expand_dims(self.image_pre, 0)
        
        self.ssd_net = ssd_vgg_300.SSDNet()
        reuse = True if 'ssd_net' in locals() else None
        with slim.arg_scope(self.ssd_net.arg_scope(data_format=data_format)):
            self.predictions, self.localisations, _, _ = self.ssd_net.net(self.image_4d, is_training=False, reuse=reuse)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, ckpt_file)
        self.ssd_anchors = self.ssd_net.anchors(self.net_shape)

    def detect_batch(self, img_list, img_names):
        num = len(img_list) 
        ctg_dict = dict()
        img_objects = []
        for i in range(num):
            frame_obj = dict()
            frame_obj['img_id'] = img_names[i]
            img = img_list[i]
            results, _ = self.object_detect(img) 
            frame_obj['object_info'] = results
            print(results)
            frame_obj['obj_num'] = len(results)
            for j in range(len(results)):
                result = results[j]
                ctg_name = result['category']
                if ctg_name not in ctg_dict:
                    ctg_dict[ctg_name] = 1
                else:
                    ctg_dict[ctg_name] += 1
            img_objects.append(frame_obj)
        ctg_details = []
        for (k,v) in Counter(ctg_dict).most_common(self.top_num_in_video):
            ctg_detail = dict()
            ctg_detail['category'] = k
            ctg_detail['times'] = v
            ctg_details.append(ctg_detail)
        result_list = dict()
        result_list['ctg_details'] = ctg_details
        result_list['img_objects'] = img_objects

        return result_list

    # Main image processing routine.
    def object_detect(self, img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300), demo = False):
        # Run SSD network.
        rimg, rpredictions, rlocalisations, rbbox_img = self.sess.run([self.image_4d, self.predictions, self.localisations, self.bbox_img],
                                                                  feed_dict={self.img_input: img})
        
        # Get classes and bboxes from the net outputs.
        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
                rpredictions, rlocalisations, self.ssd_anchors,
                select_threshold=select_threshold, img_shape=self.net_shape, num_classes=21, decode=True)
        
        rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
        # Resize bboxes to original image shape. Note: useless for Resize.WARP!
        rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
        if demo == True:
            return rclasses, rscores, rbboxes
        results = []
        numbox = rclasses.shape[0]
        if numbox is 0:
            print "no object detected in this image"
            return results, numbox
        for i in range(numbox):
            result = dict()
            result['label'] = rclasses[i].astype(int)
            result['category'] = self.class_names[result['label']]
            result['score'] = rscores[i]
            result['x1'] = rbboxes[i, 0]
            result['y1'] = rbboxes[i, 1]
            result['x2'] = rbboxes[i, 2]
            result['y2'] = rbboxes[i, 3]
            results.append(result)


        return results, numbox

    def demo_run(self):

        # Test on some demo image and visualize output.
        path = './demo/'
        #image_names = sorted(os.listdir(path))
        image_names = util.find_files(path) 
        
        for filename in image_names:
            img = mpimg.imread(filename)
            start = time()
            rclasses, rscores, rbboxes =  self.object_detect(img, demo=True)
            total_time = time() - start
            print("time cost%s"%total_time)
            visualization.plt_bboxes(img, rclasses, self.class_names, rscores, rbboxes)
           # visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)

if __name__ == '__main__':
    ssd = SSD_detector()
    ssd.demo_run()


