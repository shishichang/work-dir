import sys
import _init_paths
sys.path.append('.')
#sys.path.append('/root/ssc/caffe/python')
import tools_matrix as tools
import caffe
import os
import cv2
import numpy as np
import time
deploy = 'models/12net.prototxt'
caffemodel = 'models/12net.caffemodel'
net_12 = caffe.Net(deploy,caffemodel,caffe.TEST)

deploy = 'models/24net.prototxt'
caffemodel = 'models/24net.caffemodel'
net_24 = caffe.Net(deploy,caffemodel,caffe.TEST)

deploy = 'models/48net.prototxt'
caffemodel = 'models/48net.caffemodel'
net_48 = caffe.Net(deploy,caffemodel,caffe.TEST)

deploy = 'models/landmark_deploy.prototxt'
caffemodel = 'models/VanFace.caffemodel'
net = caffe.Net(deploy,caffemodel,caffe.TEST)
net.name = 'FaceThink_face_landmark_test'


def detectFace(img_path,threshold):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (int(img.shape[1]), int(img.shape[0])))
    #cv2.imshow('resized_img', img)
    #cv2.waitKey()
    #print('img.shape')
    #print(img.shape)
    caffe_img = (img.copy()-127.5)/128
    origin_h,origin_w,ch = caffe_img.shape
    scales = tools.calculateScales(img)
    out = []
    #print(len(scales))
    #print(scales)
    for scale in scales:
        hs = int(origin_h*scale)
        ws = int(origin_w*scale)
        scale_img = cv2.resize(caffe_img,(ws,hs))
        #print('scale_img.shape')
        #print(scale_img.shape)
        scale_img = np.swapaxes(scale_img, 0, 2)
        net_12.blobs['data'].reshape(1,3,ws,hs)
        net_12.blobs['data'].data[...]=scale_img
	caffe.set_device(3)
	caffe.set_mode_gpu()
	out_ = net_12.forward()
        out.append(out_)
    image_num = len(scales)
    rectangles = []
    for i in range(image_num):    
        cls_prob = out[i]['prob1'][0][1]
        roi      = out[i]['conv4-2'][0]
        out_h,out_w = cls_prob.shape
        out_side = max(out_h,out_w)
        rectangle = tools.detect_face_12net(cls_prob,roi,out_side,1/scales[i],origin_w,origin_h,threshold[0])
        rectangles.extend(rectangle)
    rectangles = tools.NMS(rectangles,0.7,'iou')
    rectangles_24 = rectangles    

    if len(rectangles)==0:
        return rectangles, rectangles_24
    net_24.blobs['data'].reshape(len(rectangles),3,24,24)
    crop_number = 0
    for rectangle in rectangles:
        crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
        #cv2.imshow('crop_image',crop_img)
        #print('crop_img.shape')
        #print(crop_img.shape)
        #cv2.waitKey()
        scale_img = cv2.resize(crop_img,(24,24))
        #cv2.imshow('image',scale_img)
        #print('scale_img1.shape')
        #print(scale_img.shape)
        #cv2.waitKey()
        scale_img = np.swapaxes(scale_img, 0, 2)
        #print('scale_img2.shape')
        #print(scale_img.shape)
        net_24.blobs['data'].data[crop_number] =scale_img 
        crop_number += 1
    out = net_24.forward()
    cls_prob = out['prob1']
    roi_prob = out['conv5-2']
    rectangles = tools.filter_face_24net(cls_prob,roi_prob,rectangles,origin_w,origin_h,threshold[1])
    rectangles_24 = rectangles    

    if len(rectangles)==0:
        return rectangles_24,rectangles
    net_48.blobs['data'].reshape(len(rectangles),3,48,48)
    crop_number = 0
    for rectangle in rectangles:
        crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
        scale_img = cv2.resize(crop_img,(48,48))
        #cv2.imshow("scale_img", scale_img)
        #cv2.waitKey()
        scale_img = np.swapaxes(scale_img, 0, 2)
        net_48.blobs['data'].data[crop_number] =scale_img 
        crop_number += 1
    out = net_48.forward()
    cls_prob = out['prob1']
    roi_prob = out['conv6-2']
    pts_prob = out['conv6-3']
    rectangles = tools.filter_face_48net(cls_prob,roi_prob,pts_prob,rectangles,origin_w,origin_h,threshold[2])

    return rectangles_24, rectangles


#DIRECTORY = "test_hard"
DIRECTORY = "/data1/img_data/CelebA/Img/img_celeba.7z/img_celeba"
#DIRECTORY = '/data1/shishch/mtcnn/face-landmark/images/'
PATTERN = ('.jpg', '.jpeg')
def find_files(directory, pattern=PATTERN):
  files = []
  for path, d, filelist in os.walk(directory):
      for filename in filelist:
          if filename.lower().endswith(pattern):
              files.append(os.path.join(path, filename))
  return files

def run_images():
    files = find_files(DIRECTORY, PATTERN)
    img_idx = 0
    for imgpath in files:
        threshold = [0.6,0.6,0.7]
        img_idx += 1
        print('img_idx:%s'%img_idx)
        start = time.time()
        rectangles_24, rectangles= detectFace(imgpath,threshold)
        last = time.time() - start
        print("detection time: {}s.".format(last))
        img = cv2.imread(imgpath)
        img = cv2.resize(img, (int(img.shape[1]), int(img.shape[0])))
        draw = img.copy()
        face_total = 0
        for index, det in enumerate(rectangles):
            face_total += 1
            x1 = int(det[0])
            y1 = int(det[1]) 
            x2 = int(det[2])
            y2 = int(det[3])
            if x1 < 0: x1 = 0
            if y1 < 0: y1 = 0
            if x2 > img.shape[1]: x2 = img.shape[1]
            if y2 > img.shape[0]: y2 = img.shape[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            roi = img[y1:y2 + 1, x1:x2 + 1, ]
            gray_img = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            w = 60
            h = 60
            res = cv2.resize(gray_img, (w, h), 0.0, 0.0, interpolation=cv2.INTER_CUBIC)
            resize_mat = np.float32(res)

            m = np.zeros((w, h))
            sd = np.zeros((w, h))
            mean, std_dev = cv2.meanStdDev(resize_mat, m, sd)
            new_m = mean[0][0]
            new_sd = std_dev[0][0]
            new_img = (resize_mat - new_m) / (0.000001 + new_sd)

            if new_img.shape[0] != net.blobs['data'].data[0].shape or new_img.shape[1] != net.blobs['data'].data[1].shape:
                print "Incorrect resize to correct dimensions."

            net.blobs['data'].data[...] = new_img
            landmark_time_start = time.time()
            out = net.forward()
            landmark_time_end = time.time()
            landmark_time = landmark_time_end - landmark_time_start
            print "landmark time is {}".format(landmark_time)
            points = net.blobs['Dense3'].data[0].flatten()

            point_pair_l = len(points)
            for i in range(point_pair_l / 2):
                x = points[2*i] * (x2 - x1) + x1
                y = points[2*i+1] * (y2 - y1) + y1
                cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), 2)

        cv2.imshow("landmark", img) 
        cv2.waitKey(500)

def run_video():

    threshold = [0.6,0.6,0.7]
    video_file = '../../mtcnn-hdf5/501.rmvb'
    vcap = cv2.VideoCapture(video_file)
    if False == vcap.isOpened():
        print "video cannot open"
    else:
        frame_count = 0
        while True:
            ret, img = vcap.read()
            if False == ret:
                print "read over"
                break
            frame_count += 1
            rectangles_24, rectangles= detectFace(img,threshold)
            threshold = [0.6,0.6,0.7]
            draw = img.copy()
            
            for rectangle in rectangles:
               # cv2.putText(draw,str(rectangle[4]),(int(rectangle[0]),int(rectangle[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
                cv2.rectangle(draw,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,0,0),1)
               # for i in range(5,15,2):
               # 	cv2.circle(draw,(int(rectangle[i+0]),int(rectangle[i+1])),2,(0,255,0))
            cv2.imshow('faces', draw)
            cv2.waitKey(1)
run_images()
