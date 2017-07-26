#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import grpc
import cv2
import sys, os
sys.path.append('protobuf')
import object_pb2
from scipy import misc
from io import BytesIO
import numpy as np
import time
from PIL import Image
import multiprocessing as mp
#import visualize_result2 as vis
import uuid

#rgb images are resized to fixed size 500X500
resized_dim = 500
#The max number of images allow to pass each time 
max_input = 90 

channel = grpc.insecure_channel('localhost:49020')
stub = object_pb2.ObjectStub(channel)

DIRECTORY = "data"
PATTERN = ('.jpg', '.jpeg')
def find_files(directory, pattern=PATTERN):
  files = []
  for path, d, filelist in os.walk(directory):
      for filename in filelist:
          if filename.lower().endswith(pattern):
              files.append(os.path.join(path, filename))
  return files

def run_video(times):
    video_name = 'test.mp4'
    video_file = os.path.join(DIRECTORY, video_name)
    print(video_file)
    vcap = cv2.VideoCapture(video_file)
    if False == vcap.isOpened():
        print "video cannot open!\n"
        return -1
    frame_count = 0
    batch_req = object_pb2.ObjectBatchRequest()
    #填充基本标识
    batch_req.msgheader.version =  1
    batch_req.msgheader.type    =  1
    batch_req.common.appid      =  'short_video'
    batch_req.common.tracer_id  = str(uuid.uuid4())
    batch_req.common.timeout_ms = 2000                 
    batch_req.url               = ''
    batch_req.batch_id = video_name    
    
    while True:
        ret, img = vcap.read()
        if False == ret:
            break
        
        if frame_count < max_input:
            frame_count += 1
            item = batch_req.frame_image.add()
            image = Image.fromarray(img)
            image = image.resize([resized_dim, resized_dim])
            
            fs = BytesIO()
            image.save(fs, format='JPEG')
            imgbyte = fs.getvalue()
            item.img_id = str(frame_count)
            item.object_img = imgbyte 
        else:
            print('Maximun number is %s'%max_input)
            break
            
    batch_req.img_num = frame_count    
    start = time.time()
    response = stub.JudgeBatch(batch_req)
    test_time = time.time() - start
   # vis.print_batch_results(response)
    print(response)

def run_batch(times):
    if not os.path.exists(DIRECTORY):
        raise Exception('The directory is not exist!')
    else:
        for i in range(times):
            files = find_files(DIRECTORY, PATTERN)
            echo_size = min(max_input, len(files))
            input_files = files[0:echo_size]
            batch_req = object_pb2.ObjectBatchRequest()
            #填充基本标识
            batch_req.msgheader.version =  1
            batch_req.msgheader.type    =  1
            batch_req.common.appid      =  'batch images'
            batch_req.common.tracer_id  = str(uuid.uuid4())
            batch_req.common.timeout_ms = 2000                 
            batch_req.url               = ''
            batch_req.batch_id = DIRECTORY    
            
            for filename in input_files:
                item = batch_req.frame_image.add()
                
                image = Image.open(filename)
                image = image.resize([resized_dim, resized_dim])

                fs = BytesIO()
                image.save(fs, format='JPEG')
                imgbyte = fs.getvalue()
                
                item.img_id = filename
                item.object_img = imgbyte 
           
            batch_req.img_num = echo_size    
            start = time.time()
            response = stub.JudgeBatch(batch_req)
            test_time = time.time() - start
   #         vis.print_batch_results(response)
            print(response)

def run(times):
    filename = 'data/000456.jpg'
    image = Image.open(filename)
    image = image.resize([resized_dim, resized_dim])

    fs = BytesIO()
    image.save(fs, format='JPEG')
    imgbyte = fs.getvalue()

    obj_req = object_pb2.ObjectRequest(object_img=imgbyte)
#填充基本标识
    obj_req.msgheader.version =  1
    obj_req.msgheader.type    =  1
    obj_req.common.appid      =  'single image'
    obj_req.common.tracer_id  = str(uuid.uuid4())
    obj_req.common.timeout_ms = 2000                 
    obj_req.url               = ''
    obj_req.img_id             = filename
    for i in range(times):
        start = time.time()
        response = stub.Judge(obj_req)
        print('detect %s, time:%s'% (filename, time.time() - start))
        print(response)

def main(nProcs, nTimes):
    procs = list()
    for i in range(nProcs):
        #run_batch
        p = mp.Process(target = run_batch, kwargs={'times':nTimes})
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

if __name__ == '__main__':
  main(1,1)

'''
说明：如下代码在nginx中不会轮询
def run():
    channel = grpc.insecure_channel('localhost:5000')
    stub = face_pb2.FaceStub(channel)
    imgbyte = open('image/test.jpg', 'rb').read()
    #img = cv2.imread('test.jpg')
    while True:
        response = stub.Judge(face_pb2.FaceReq(face_img=imgbyte))
        print('detect img=%s'%response)
'''
