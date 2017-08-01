import os
import math
import random
import time

import numpy as np
import tensorflow as tf
import cv2
slim = tf.contrib.slim
import matplotlib.pyplot as plt
import sys
import json
from net.framework_deploy import create_framework
from multiprocessing.pool import ThreadPool
from tools import visualization,util
import matplotlib.pyplot as plt

pool = ThreadPool()


class YOLO_detector(object):
    
    def __init__(self):
        model_name = 'yolo-coco'
        model_dir = './models'
        gpu_id = 4
        os.environ["VISIBLE_CUDA_DEVICES"]='4'
        self.gpu_utility = 0.9
        
        self.pb_file = '{}/{}.pb'.format(model_dir, model_name)
        self.meta_file = '{}/{}.meta'.format(model_dir, model_name)
        self.batch = 4
        
        self.graph = tf.Graph()
        with tf.device('/gpu:4'):
            with self.graph.as_default() as g:
                self.build_from_pb()
        return
    def say(self, *msgs):
        msgs = list(msgs)
        for msg in msgs:
            if msg is None: continue
            print(msg)
    def build_from_pb(self):
        with tf.gfile.FastGFile(self.pb_file, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")
        
        with open(self.meta_file, "r") as fp:
            self.meta = json.load(fp)
            #print("meta:_")
            #print(self.meta)

        self.framework = create_framework(self.meta)

        #Placeholders
        self.inp = tf.get_default_graph().get_tensor_by_name('input:0')
        self.feed = dict() #other placeholders
        self.out = tf.get_default_graph().get_tensor_by_name('output:0')

        self.setup_meta_ops()
        
    def setup_meta_ops(self):
        cfg = dict({
            'allow_soft_placement': False,
            'log_device_placement': False
            })
        utility = min(self.gpu_utility, 1.0)
        if utility > 0.0:
            print('GPU model with {} usage'.format(utility))
            cfg['gpu_options'] = tf.GPUOptions(per_process_gpu_memory_fraction = utility)        
            cfg['allow_soft_placement'] = True
        else:
            print('Run totally on CPU')
            cfg['device_count'] = {'GPU': 0}

        self.sess = tf.Session(config = tf.ConfigProto(**cfg))
        self.sess.run(tf.global_variables_initializer())
    
    def detect_object(self, im):
        this_inp = self.framework.preprocess(im)
        expanded = np.expand_dims(this_inp, 0)
        inp_feed = list()
        feed_dict = {self.inp: expanded}
        inp_feed.append(expanded)
        print("expanded.shape:")
        print(expanded.shape)
        feed_dict = {self.inp : expanded}    
        self.say("Forwarding the image input.")
        start = time.time()
        out = self.sess.run(self.out, feed_dict)
        print(type(out))
        time_value = time.time()
        last = time_value - start
        self.say('Cost time of run = {}s.'.format(last))
        imgcv, results = self.framework.postprocess(out[0], im)
        last = time.time() - time_value
        self.say('Cost time of postprocess = {}s.'.format(last))
        return imgcv, results
        
    def detect_batch(self, img_list, img_names):
        batch = min(self.batch, len(img_list))
        all_inps = [i for i in range(len(img_list))]
        n_batch = int(math.ceil(len(all_inps)/batch))
        img_idx = 0
        img_objects = []
        for j in range(n_batch):
            from_idx = j * batch
            to_idx = min(from_idx + batch, len(img_list))
            inp_feed = list(); new_all = list()
            num_in_batch = to_idx - from_idx
            this_batch = all_inps[from_idx:to_idx]
            for inp in this_batch:
                this_inp = self.framework.preprocess(img_list[inp])
                expanded = np.expand_dims(this_inp, 0)
                inp_feed.append(expanded)
            # Feed to the net
            feed_dict = {self.inp : np.concatenate(inp_feed, 0)}    
            self.say('Forwarding {} inputs ...'.format(len(inp_feed)))
            start = time.time()
            out = self.sess.run(self.out, feed_dict)
            stop = time.time(); 
            last = stop - start
            self.say('Total time = {}s / {} inps = {} ips'.format(
                last, len(inp_feed), len(inp_feed) / last))
            # Post processing
            self.say('Post processing {} inputs ...'.format(len(inp_feed)))
            start = time.time()
            for j in range(num_in_batch):
                frame_obj = dict() 
                frame_obj['img_id'] = img_names[img_idx]
                im = img_list[img_idx]
                imgcv, results = self.framework.postprocess(out[j], im)
                frame_obj['obj_num'] = len(results) 
                frame_obj['object_info'] = results
                img_idx += 1
                img_objects.append(frame_obj)
            #pool.map(lambda p: (lambda i, prediction:
            #    self.framework.postprocess(
            #       prediction, os.path.join(inp_path, this_batch[i])))(*p),
            #    enumerate(out))
            stop = time.time(); last = stop - start

            # Timing
            self.say('Total time = {}s '.format(last))
        batch_results = dict() 
        batch_results['img_objects'] = img_objects 
        return img_objects

    def deploy_predict(self, inp_path):
        all_inps = os.listdir(inp_path)
        all_inps = [i for i in all_inps if self.framework.is_inp(i)]
        if not all_inps:
            msg = 'Failed to find any images in {}.'
            exit('Error: {}').format(msg.format(inp_path))

        batch = min(self.batch, len(all_inps))
        n_batch = int(math.ceil(len(all_inps)/batch))

        # predict in batches
        for j in range(n_batch):
            from_idx = j * batch
            to_idx = min(from_idx + batch, len(all_inps))

            # collect images input in the batch
            inp_feed = list(); new_all = list()
            this_batch = all_inps[from_idx:to_idx]
            for inp in this_batch:
                new_all += [inp]
                this_inp = os.path.join(inp_path, inp)
                this_inp = self.framework.preprocess(this_inp)
                expanded = np.expand_dims(this_inp, 0)
                inp_feed.append(expanded)
            this_batch = new_all

            # Feed to the net
            feed_dict = {self.inp : np.concatenate(inp_feed, 0)}    
            self.say('Forwarding {} inputs ...'.format(len(inp_feed)))
            start = time.time()
            out = self.sess.run(self.out, feed_dict)
            stop = time.time(); 
            last = stop - start
            self.say('Total time = {}s / {} inps = {} ips'.format(
                last, len(inp_feed), len(inp_feed) / last))

            # Post processing
            self.say('Post processing {} inputs ...'.format(len(inp_feed)))
            start = time.time()
            pool.map(lambda p: (lambda i, prediction:
                self.framework.postprocess(
                   prediction, os.path.join(inp_path, this_batch[i])))(*p),
                enumerate(out))
            stop = time.time(); last = stop - start

            # Timing
            self.say('Total time = {}s / {} inps = {} ips'.format(
                last, len(inp_feed), len(inp_feed) / last))

def demo_im():
    yolo = YOLO_detector()
    img_dir = "./sample_img/"
    image_names = util.find_files(img_dir) 
    for filename in image_names:
        im = cv2.imread(filename)
        plt.figure()
        imgcv, results = yolo.detect_object(im) 
        plt.imshow(imgcv)
        plt.show()

def demo_batch():
    yolo = YOLO_detector()
    img_dir = "./sample_img/"
    image_names = util.find_files(img_dir) 
    img_list = []
    img_names = []
    for filename in image_names:
        im = cv2.imread(filename)
        img_list.append(im)
        img_names.append(filename)
    batch_results = yolo.detect_batch(img_list, img_names)
    print(batch_results)


def demo_deploy():
    yolo = YOLO_detector()
    img_dir = "sample_img"
    yolo.deploy_predict(img_dir)

if __name__ == '__main__':
    demo_im()
    demo_batch()
    demo_deploy()

