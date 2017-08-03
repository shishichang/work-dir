import sys
import _init_paths
sys.path.append('.')
#sys.path.append('/root/ssc/caffe/python')
import tools_matrix as tools
import caffe
import os
import cv2
import numpy as np
deploy = 'models/12net.prototxt'
caffemodel = 'models/12net.caffemodel'
net_12 = caffe.Net(deploy,caffemodel,caffe.TEST)

deploy = 'models/24net.prototxt'
caffemodel = 'models/24net.caffemodel'
net_24 = caffe.Net(deploy,caffemodel,caffe.TEST)

deploy = 'models/48net.prototxt'
caffemodel = 'models/48net.caffemodel'
#caffemodel = '_iter_10000.caffemodel'
#deploy = '48net_5pts.prototxt'
net_48 = caffe.Net(deploy,caffemodel,caffe.TEST)
tmp_s = 1 


def detectFace(img_path,threshold):
    img = cv2.imread(img_path)
    #print('img.shape')
    #print(img.shape)
    #img = img_path
    img = cv2.resize(img, (int(img.shape[1]*tmp_s), int(img.shape[0]*tmp_s)))
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
#DIRECTORY = "/data1/img_data/CelebA/Img/img_celeba.7z/img_celeba"
DIRECTORY = "/data1/obj_dec/tmp/48/positive-5pts"
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
        #threshold = [0.3,0.3,0.7]
        img_idx += 1
        print('img_idx:%s'%img_idx)
        rectangles_24, rectangles= detectFace(imgpath,threshold)
        img = cv2.imread(imgpath)
        img = cv2.resize(img, (int(img.shape[1]*tmp_s), int(img.shape[0]*tmp_s)))
        draw = img.copy()
        
        for rectangle in rectangles:
            cv2.putText(draw,str(rectangle[4]),(int(rectangle[0]),int(rectangle[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
            cv2.rectangle(draw,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,0,0),1)
            for i in range(5,15,2):
            	cv2.circle(draw,(int(rectangle[i+0]),int(rectangle[i+1])),2,(0,255,0))
        
        draw24 = img.copy()
        for rectangle in rectangles_24:
            cv2.putText(draw24,str(rectangle[4]),(int(rectangle[0]),int(rectangle[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
            cv2.rectangle(draw24,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,0,0),1)
        print 'img: ', imgpath 
        cv2.imshow('test', draw)
        #cv2.imshow('test_24', draw24)
        cv2.waitKey()
#cv2.imwrite('test.jpg',draw)

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
