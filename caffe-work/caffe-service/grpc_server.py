#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
from concurrent import futures
import grpc
import consul
import object_model
from PIL import Image
from io import BytesIO
from scipy import misc
import numpy as np
import cv2
import caffe
import sys
import cPickle as pickle
sys.path.append('protobuf')
import object_pb2

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

class ObjectSSD(object_pb2.ObjectServicer):
    def __init__(self):
        global gpu_id
        self.gpu_id = gpu_id
        self.object_model = object_model.SSD(gpu_id=self.gpu_id)
        print('gpu_id=%d'%self.gpu_id)
        
    def Judge(self, request, context):
        caffe.set_device(self.gpu_id)
        caffe.set_mode_gpu()
        pimg = request.object_img
        obj_rep = object_pb2.ObjectReply()
        read_state = 1
        try:
            image = misc.imread(BytesIO(pimg))
        except:
            read_state = 0
        if  0 == read_state:
            #ERR_BADF
            complete = 4 
            obj_rep.common.desc = 'bad image file'
            results = []
        else:
            results ,numbox = self.object_model.detect_object(image)
            complete = 1
       
        obj_rep.msgheader.version = request.msgheader.version
        obj_rep.msgheader.type    = request.msgheader.type
        obj_rep.common.appid    = request.common.appid
        obj_rep.common.tracer_id = request.common.tracer_id
        obj_rep.common.state = complete
        obj_rep.common.state = " " 
        obj_rep.url = request.url
        obj_rep.img_id = request.img_id
        obj_rep.obj_num = len(results)

        for i in range(len(results)):
            fi = obj_rep.object_info.add()
            fi.label = results[i]['label']
            fi.category = results[i]['class']
            fi.score = results[i]['score']
            fi.x1 = results[i]['x1']
            fi.y1 = results[i]['y1']
            fi.x2 = results[i]['x2']
            fi.y2 = results[i]['y2']
        return obj_rep

    def JudgeBatch(self, request, context):
        caffe.set_device(self.gpu_id)
        caffe.set_mode_gpu()
        image_list = []
        image_names = []
        read_state = 1
        for k, v in enumerate(request.frame_image):
            try:
                image = misc.imread(BytesIO(v.object_img))
                image_names.append(v.img_id)
                image_list.append(image)
            except:
                read_state = 0
                break

        obj_batch_rep = object_pb2.ObjectBatchReply()         
        obj_batch_rep.msgheader.version = request.msgheader.version
        obj_batch_rep.msgheader.type    = request.msgheader.type
        obj_batch_rep.common.appid    = request.common.appid
        obj_batch_rep.common.tracer_id = request.common.tracer_id
        obj_batch_rep.img_num = request.img_num
        obj_batch_rep.url = request.url
        obj_batch_rep.batch_id = request.batch_id
        if 0 == read_state:
            obj_batch_rep.common.state = 4
            obj_batch_rep.common.desc = 'contain bad file'
            return obj_batch_req
        else:
            obj_batch_rep.common.state = 1
            obj_batch_rep.common.desc = ' '

            results = self.object_model.detect_batch(image_list, image_names)
            for ctg_detail_iter in results['ctg_details']:
                ctg_detail = obj_batch_rep.category_detail.add()
                ctg_detail.category = ctg_detail_iter['category'] 
                ctg_detail.times = ctg_detail_iter['times']

            for img_objects_iter in results['img_objects']:
                frame_obj = obj_batch_rep.frame_object.add() 
                frame_obj.img_id = img_objects_iter['img_id']
                frame_obj.obj_num = img_objects_iter['obj_num']
                for object_info_iter in img_objects_iter['object_info']:
                    obj_info = frame_obj.object_info.add()
                    obj_info.label = object_info_iter['label'] 
                    obj_info.score = object_info_iter['score']
                    obj_info.x1 = object_info_iter['x1'] 
                    obj_info.y1 = object_info_iter['y1']
                    obj_info.x2 = object_info_iter['x2']
                    obj_info.y2 = object_info_iter['y2']
                    obj_info.category = object_info_iter['category']


            return obj_batch_rep

def serve(port):
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    object_pb2.add_ObjectServicer_to_server(ObjectSSD(), server)
    server.add_insecure_port('[::]:'+ port)
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    port = '40300'
    global gpu_id
    gpu_id = 0
    if len(sys.argv) > 1:
        port = sys.argv[1]
    if len(sys.argv) > 2:
        gpu_id = int(sys.argv[2])
    serve(port)

