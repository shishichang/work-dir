import sys
sys.path.append("./")
from utils.im_transform import imcv2_recolor, imcv2_affine_trans
from utils import box
from utils import box_1
import math
import random
import time
import os

import numpy as np
import tensorflow as tf
import cv2
slim = tf.contrib.slim
import matplotlib.pyplot as plt
from multiprocessing.pool import ThreadPool
from tools import visualization,util
import matplotlib.pyplot as plt
from collections import Counter
import json

pool = ThreadPool()
os.environ["CUDA_VISIBLE_DEVICES"]='4'




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
                gpu_options = tf.GPUOptions(allow_growth=True)
                sess_config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
                self.sess = tf.Session(config = sess_config)
                self.sess.run(tf.global_variables_initializer())
        return
    
    def build_from_pb(self):
        with tf.gfile.FastGFile(self.pb_file, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")
        
        with open(self.meta_file, "r") as fp:
            self.meta = json.load(fp)
            #print("meta:_")
            #print(self.meta)

        #self.framework = create_framework(self.meta)

        #Placeholders
        self.inp = tf.get_default_graph().get_tensor_by_name('input:0')
        self.out = tf.get_default_graph().get_tensor_by_name('output:0')

        #self.setup_meta_ops()
        
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

    def resize_input(self, im):
        h, w, c = self.meta['inp_size']
        imsz = cv2.resize(im, (w, h))
        imsz = imsz / 255.
        imsz = imsz[:,:,::-1]
        return imsz
    
    def process_box(self, b, h, w, threshold):
        max_indx = np.argmax(b.probs)
        max_prob = b.probs[max_indx]
        label = self.meta['labels'][max_indx]
        if max_prob > threshold:
        	left  = int ((b.x - b.w/2.) * w)
        	right = int ((b.x + b.w/2.) * w)
        	top   = int ((b.y - b.h/2.) * h)
        	bot   = int ((b.y + b.h/2.) * h)
        	if left  < 0    :  left = 0
        	if right > w - 1: right = w - 1
        	if top   < 0    :   top = 0
        	if bot   > h - 1:   bot = h - 1
        	mess = '{}'.format(label)
        	return (left, right, top, bot, mess, max_indx, max_prob)
        return None
       
    def preprocess(self, im, allobj = None):
        """
        Takes an image, return it as a numpy tensor that is readily
        to be fed into tfnet. If there is an accompanied annotation (allobj),
        meaning this preprocessing is serving the train process, then this
        image will be transformed with random noise to augment training data,
        using scale, translation, flipping and recolor. The accompanied
        parsed annotation (allobj) will also be modified accordingly.
        """
        if type(im) is not np.ndarray:
        	im = cv2.imread(im)
        
        if allobj is not None: # in training mode
        	result = imcv2_affine_trans(im)
        	im, dims, trans_param = result
        	scale, offs, flip = trans_param
        	for obj in allobj:
        		_fix(obj, dims, scale, offs)
        		if not flip: continue
        		obj_1_ =  obj[1]
        		obj[1] = dims[0] - obj[3]
        		obj[3] = dims[0] - obj_1_
        	im = imcv2_recolor(im)
        
        im = self.resize_input(im)
        if allobj is None: return im
        return im#, np.array(im) # for unit testing
    
    def postprocess(self, net_out):
        meta = self.meta
        result = box.box_constructor(meta,net_out)
        return result

        #original postprocess
    def postprocess_1(self, net_out, im):
        meta = self.meta
        
        boxes=box_1.box_constructor(meta,net_out)
        # meta
        meta = self.meta
        threshold = meta['thresh']
        colors = meta['colors']
        labels = meta['labels']
        imgcv = im.copy()
        h, w, _ = imgcv.shape
        result = []
        for b in boxes:
            boxResults = self.process_box(b, h, w, threshold)
            if boxResults is None:
            	continue
            left, right, top, bot, mess, max_indx, confidence = boxResults
            result.append({"label": max_indx, "category": mess, "score": float('%.2f' % confidence), "x1": float(left)/w, "y1": float(top)/h, "x2": float(right)/w, "y2": float(bot)/h})
    
        return result
        
    def detect_batch(self, img_list, img_names):
        all_inps = [i for i in range(len(img_list))]
        
        batch = min(self.batch, len(all_inps))
        n_batch = int(math.ceil(len(all_inps)/batch))

        # predict in batches
        ctg_dict = dict()
        batch_objects = []
        img_idx = -1
        for j in range(n_batch):
            from_idx = j * batch
            to_idx = min(from_idx + batch, len(all_inps))

            # collect images input in the batch
            inp_feed = list(); 
            this_batch = all_inps[from_idx:to_idx]
            for inp in this_batch:
                this_inp = img_list[inp]
                this_inp = self.preprocess(this_inp)
                expanded = np.expand_dims(this_inp, 0)
                inp_feed.append(expanded)
            
            # Feed to the net
            feed_dict = {self.inp : np.concatenate(inp_feed, 0)}    
            print('Forwarding {} inputs ...'.format(len(inp_feed)))
            start = time.time()
            out = self.sess.run(self.out, feed_dict)
            stop = time.time(); last = stop - start
            print('Forward time = {}s '.format(last))
            start = time.time()
            for i, v in enumerate(out):
                img_idx += 1
                frame_obj = dict()
                frame_obj['img_id'] = img_names[img_idx]
                frame_obj['obj_num'] = 0
                frame_obj['object_info'] = []
                detections = self.postprocess(v)
                for det in detections:
                    frame_obj['obj_num'] += 1
                    frame_obj['object_info'].append(det)
                    ctg_name = det['category']
                    if ctg_name not in ctg_dict:
                        ctg_dict[ctg_name] = 1
                    else:
                        ctg_dict[ctg_name] += 1

                batch_objects.append(frame_obj)
            stop = time.time(); last = stop - start
            print('Postprocess time = {}s '.format(last))
        ctg_details = []
        for (k, v) in Counter(ctg_dict).most_common(len(ctg_dict)):
            ctg_detail = dict()
            ctg_detail['category'] = k
            ctg_detail['times'] = v
            ctg_details.append(ctg_detail)

        results = dict()
        results['ctg_details'] = ctg_details
        results['img_objects'] = batch_objects

        return results

    def detect_object(self, im):
        this_inp = self.preprocess(im)
        expanded = np.expand_dims(this_inp, 0)
        inp_feed = list()
        feed_dict = {self.inp: expanded}
        inp_feed.append(expanded)
        feed_dict = {self.inp : expanded}    
       
        print("Forwarding the image input.")
        start = time.time()
        out = self.sess.run(self.out, feed_dict)
        
        time_value = time.time()
        last = time_value - start
        print('Cost time of run = {}s.'.format(last))
        result = self.postprocess(out[0])
        last = time.time() - time_value
        
        print('Cost time of postprocess = {}s.'.format(last))
        return result
        
def demo_batch():
    yolo = YOLO_detector()
    colors = yolo.meta['colors']
    img_dir = "./sample_img/"
    image_names = util.find_files(img_dir) 
    image_list = []
    for filename in image_names:
        im = cv2.imread(filename)
        image_list.append(im)
    results = yolo.detect_batch(image_list, image_names) 
    print(results)

def demo_im():
    yolo = YOLO_detector()
    colors = yolo.meta['colors']
    img_dir = "./sample_img/"
    image_names = util.find_files(img_dir) 
    for filename in image_names:
        im = cv2.imread(filename)
        h,w,_ = im.shape
        results = yolo.detect_object(im) 
        thick = int((h + w) // 300)
        draw = im.copy()
        h, w, _ = draw.shape
        for i in range(len(results)):
            cv2.putText(draw,str(results[i]['category']),(int(w*results[i]['x1']),int(h*results[i]['y1'])-12), 0, 1e-3*h, colors[results[i]['label']], thick//3)
            cv2.rectangle(draw,(int(w*results[i]['x1']),int(h*results[i]['y1'])),(int(w*results[i]['x2']),int(h*results[i]['y2'])), colors[results[i]['label']], thick)
        cv2.imshow("result", draw)
        cv2.waitKey()

if __name__ == '__main__':
    print("run demo_im...")
    demo_im()
#    print("run demo_batch...")
#    demo_batch()

