import matplotlib.pyplot as plt
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
    bboxes = np.empty([len(rectangles), 4])
    for i in range(len(rectangles)):
        bboxes[i,0] = int(rectangles[i][0])
        bboxes[i,1] = int(rectangles[i][2])
        bboxes[i,2] = int(rectangles[i][1])
        bboxes[i,3] = int(rectangles[i][3])
    print("bboxes")
    print(bboxes)

    return bboxes 

#usage :python landmarkPredict.py predictImage  testList.txt

system_height = 650
system_width = 1280
channels = 1
test_num = 1
pointNum = 68

S0_width = 60
S0_height = 60
vgg_height = 224
vgg_width = 224
M_left = -0.15
M_right = +1.15
M_top = -0.10
M_bottom = +1.25
pose_name = ['Pitch', 'Yaw', 'Roll']     # respect to  ['head down','out of plane left','in plane right']

def recover_coordinate(largetBBox, facepoint, width, height):
    point = np.zeros(np.shape(facepoint))
    cut_width = largetBBox[1] - largetBBox[0]
    cut_height = largetBBox[3] - largetBBox[2]
    scale_x = cut_width*1.0/width;
    scale_y = cut_height*1.0/height;
    point[0::2]=[float(j * scale_x + largetBBox[0]) for j in facepoint[0::2]] 
    point[1::2]=[float(j * scale_y + largetBBox[2]) for j in facepoint[1::2]]
    return point

def show_image(img, facepoint, bboxs, headpose):
    plt.figure(figsize=(20,10))
    for faceNum in range(0,facepoint.shape[0]):
        cv2.rectangle(img, (int(bboxs[faceNum,0]), int(bboxs[faceNum,2])), (int(bboxs[faceNum,1]), int(bboxs[faceNum,3])), (0,0,255), 2)
        for p in range(0,3):
            plt.text(int(bboxs[faceNum,0]), int(bboxs[faceNum,2])-p*30,
                '{:s} {:.2f}'.format(pose_name[p], headpose[faceNum,p]),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=12, color='white')
        for i in range(0,facepoint.shape[1]/2):
            cv2.circle(img,(int(round(facepoint[faceNum,i*2])),int(round(facepoint[faceNum,i*2+1]))),1,(0,255,0),2)
    height = img.shape[0]
    width = img.shape[1]
    if height > system_height or width > system_width:
        height_radius = system_height*1.0/height
        width_radius = system_width*1.0/width
        radius = min(height_radius,width_radius)
        img = cv2.resize(img, (0,0), fx=radius, fy=radius)

    #img = img[:,:,[2,1,0]]
    #plt.imshow(img)
    #plt.show()
    cv2.imshow("img", img)
    cv2.waitKey(1000)


def recoverPart(point,bbox,left,right,top,bottom,img_height,img_width,height,width):
    largeBBox = getCutSize(bbox,left,right,top,bottom)
    retiBBox = retifyBBoxSize(img_height,img_width,largeBBox)
    recover = recover_coordinate(retiBBox,point,height,width)
    recover=recover.astype('float32')
    return recover


def getRGBTestPart(bbox,left,right,top,bottom,im,height,width):
    img = im.copy()
    largeBBox = getCutSize(bbox,left,right,top,bottom)
    print("largeBBox")
    print(largeBBox)
    retiBBox = retifyBBox(img,largeBBox)
    print("retiBBox")
    print(retiBBox)
    cv2.rectangle(img, (int(retiBBox[0]), int(retiBBox[2])), (int(retiBBox[1]), int(retiBBox[3])), (0,0,255), 2)
    #cv2.imshow('f',img)
    #cv2.waitKey(0)
    face=img[int(retiBBox[2]):int(retiBBox[3]),int(retiBBox[0]):int(retiBBox[1]),:]
    face = cv2.resize(face,(height,width),interpolation = cv2.INTER_AREA)
    face=face.astype('float32')
    return face

def batchRecoverPart(predictPoint,totalBBox,totalSize,left,right,top,bottom,height,width):
    recoverPoint = np.zeros(predictPoint.shape)
    for i in range(0,predictPoint.shape[0]):
        recoverPoint[i] = recoverPart(predictPoint[i],totalBBox[i],left,right,top,bottom,totalSize[i,0],totalSize[i,1],height,width)
    return recoverPoint



def retifyBBox(img,bbox):
    img_height = np.shape(img)[0] - 1
    img_width = np.shape(img)[1] - 1
    if bbox[0] <0:
        bbox[0] = 0
    if bbox[1] <0:
        bbox[1] = 0
    if bbox[2] <0:
        bbox[2] = 0
    if bbox[3] <0:
        bbox[3] = 0
    if bbox[0] > img_width:
        bbox[0] = img_width
    if bbox[1] > img_width:
        bbox[1] = img_width
    if bbox[2]  > img_height:
        bbox[2] = img_height
    if bbox[3]  > img_height:
        bbox[3] = img_height
    return bbox

def retifyBBoxSize(img_height,img_width,bbox):
    if bbox[0] <0:
        bbox[0] = 0
    if bbox[1] <0:
        bbox[1] = 0
    if bbox[2] <0:
        bbox[2] = 0
    if bbox[3] <0:
        bbox[3] = 0
    if bbox[0] > img_width:
        bbox[0] = img_width
    if bbox[1] > img_width:
        bbox[1] = img_width
    if bbox[2]  > img_height:
        bbox[2] = img_height
    if bbox[3]  > img_height:
        bbox[3] = img_height
    return bbox

def getCutSize(bbox,left,right,top,bottom):   #left, right, top, and bottom

    box_width = bbox[1] - bbox[0]
    box_height = bbox[3] - bbox[2]
    cut_size=np.zeros((4))
    cut_size[0] = bbox[0] + left * box_width
    cut_size[1] = bbox[1] + (right - 1) * box_width
    cut_size[2] = bbox[2] + top * box_height
    cut_size[3] = bbox[3] + (bottom-1) * box_height
    return cut_size


def getFaceImage(image,bboxs,left,right,top,bottom,height,width):
    num = bboxs.shape[0]
    faces = np.zeros((num,channels,height,width))
    for i in range (0,num):
        faces[i] = getTestPart(bboxs[i],left,right,top,bottom,image,height,width)/255.0
        print faces[i].shape
        # cv2.imshow('f',faces[i][0])
        #  cv2.waitKey(0)
    return faces

def detectFace_dlib(img):
    detector = dlib.get_frontal_face_detector()
    dets = detector(img,1)
    bboxs = np.zeros((len(dets),4))
    for i, d in enumerate(dets):
        bboxs[i,0] = d.left();
        bboxs[i,1] = d.right();
        bboxs[i,2] = d.top();
        bboxs[i,3] = d.bottom();
    return bboxs;


def predictImage(filename):
    vgg_point_MODEL_FILE = 'models/68point_vgg_with_pose_deploy.prototxt'
    vgg_point_PRETRAINED = 'models/68point_vgg_with_pose.caffemodel'
    mean_filename='models/VGG_mean.binaryproto'
    threshold = [0.6,0.6,0.7]
    vgg_point_net=caffe.Net(vgg_point_MODEL_FILE,vgg_point_PRETRAINED,caffe.TEST)
    # caffe.set_mode_cpu()
    caffe.set_mode_gpu()
    caffe.set_device(0)
    f = open(filename)
    line = f.readline()
    index = 0
    proto_data = open(mean_filename, "rb").read()
    a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
    mean = caffe.io.blobproto_to_array(a)[0]

    while line:
        print index
        line = line.strip()
        info = line.split(' ')
        imgPath = info[0]
        print imgPath
        num = 1
        colorImage = cv2.imread(imgPath)
        bboxs = detectFace(imgPath, threshold)
        faceNum = bboxs.shape[0]
        faces = np.zeros((1,3,vgg_height,vgg_width))
        predictpoints = np.zeros((faceNum,pointNum*2))
        predictpose = np.zeros((faceNum,3))
        imgsize = np.zeros((2))
        imgsize[0] = colorImage.shape[0]-1
        imgsize[1] = colorImage.shape[1]-1
        TotalSize = np.zeros((faceNum,2))
        for i in range(0,faceNum):
            TotalSize[i] = imgsize
        for i in range(0,faceNum):
            bbox = bboxs[i]
            print(bbox)
            print("bbox")
            colorface = getRGBTestPart(bbox,M_left,M_right,M_top,M_bottom,colorImage,vgg_height,vgg_width)
            normalface = np.zeros(mean.shape)
            normalface[0] = colorface[:,:,0]
            normalface[1] = colorface[:,:,1]
            normalface[2] = colorface[:,:,2]
            normalface = normalface - mean
            faces[0] = normalface

            blobName = '68point'
            data4DL = np.zeros([faces.shape[0],1,1,1])
            vgg_point_net.set_input_arrays(faces.astype(np.float32),data4DL.astype(np.float32))
            vgg_point_net.forward()
            predictpoints[i] = vgg_point_net.blobs[blobName].data[0]

            blobName = 'poselayer'
            pose_prediction = vgg_point_net.blobs[blobName].data
            predictpose[i] = pose_prediction * 50

        predictpoints = predictpoints * vgg_height/2 + vgg_width/2
        level1Point = batchRecoverPart(predictpoints,bboxs,TotalSize,M_left,M_right,M_top,M_bottom,vgg_height,vgg_width)

        show_image(colorImage, level1Point, bboxs, predictpose)
        line = f.readline()
        index = index + 1


if __name__ == '__main__':
    filename = "testList.txt"
    predictImage(filename)