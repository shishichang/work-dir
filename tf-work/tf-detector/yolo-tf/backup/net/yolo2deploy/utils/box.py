import numpy as np
import math


class BoundBox:
    def __init__(self, classes):
        self.x, self.y = float(), float()
        self.w, self.h = float(), float()
        self.c = float()
        self.class_num = classes
        self.probs = np.zeros((classes,))

def overlap(x1,w1,x2,w2):
    l1 = x1 - w1 / 2.;
    l2 = x2 - w2 / 2.;
    left = max(l1, l2)
    r1 = x1 + w1 / 2.;
    r2 = x2 + w2 / 2.;
    right = min(r1, r2)
    return right - left;

def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w);
    h = overlap(a.y, a.h, b.y, b.h);
    if w < 0 or h < 0: return 0;
    area = w * h;
    return area;

def box_union(a, b):
    i = box_intersection(a, b);
    u = a.w * a.h + b.w * b.h - i;
    return u;

def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b);

def prob_compare(box):
    return box.probs[box.class_num]

def prob_compare2(boxa, boxb):
    if (boxa.pi < boxb.pi):
        return 1
    elif(boxa.pi == boxb.pi):
        return 0
    else:
        return -1


def explit_c_mine(x):
    y = 1.0/(1.0 + math.exp(-x))
    return y

def NMS(final_probs, final_bbox):
    print("final_probs.shape")
    print(final_probs.shape)
    print("final_bbox.shape")
    print(final_bbox.shape)
    boxes = list()
    indices = set()
    
    pred_length = final_bbox.shape[0]
    class_length = final_probs.shape[1]
    for class_loop in range(class_length):
        for index in range(pred_length):
            if final_probs[index, class_loop] == 0: continue
            for index2 in range(index+1, pred_length):
                if final_probs[index2, class_loop] == 0: continue
                if index==index2: continue
                a = BoundBox(0)
                b = BoundBox(0) 
                a.x = final_bbox[index,0]
                a.y = final_bbox[index,1]
                a.w = final_bbox[index,2]
                a.h = final_bbox[index,3]
                b.x = final_bbox[index2,0]
                b.y = final_bbox[index2,1]
                b.w = final_bbox[index2,2]
                b.h = final_bbox[index2,3]
                if box_iou(a, b) >= 0.4:
                    if final_probs[index2, class_loop] > final_probs[index, class_loop]:
                        final_probs[index, class_loop] = 0
                        break
                    final_probs[index2, class_loop] = 0
            if index not in indices:
                bb = BoundBox(class_length)
                bb.x = final_bbox[index, 0]
                bb.y = final_bbox[index, 1]
                bb.w = final_bbox[index, 2]
                bb.h = final_bbox[index, 3]
                bb.probs = np.asarray(final_probs[index, :])
                boxes.append(bb)
                indices.add(index)
    return boxes

def box_constructor(meta, net_out_in):
    threshold = meta['thresh']
    anchors = np.asarray(meta['anchors'])
    H, W, _ = meta['out_size']
    
    C = int(meta['classes'])
    B = int(meta['num'])
    print("net_out_in.shape:")
    print(net_out_in.shape)
    net_out = net_out_in.reshape([H, W, B, int(net_out_in.shape[2]/B)])
    Classes = net_out[:,:,:,5:]
    Bbox_pred = net_out[:,:,:,:5]
    probs = np.zeros((H,W,B,C), dtype=np.float32)

    for row in range(H):
        for col in range(W):
            for box_loop in range(B):
                arr_max = 0
                sum = 0
                Bbox_pred[row, col, box_loop, 4] = explit_c_mine(Bbox_pred[row, col, box_loop, 4])
                Bbox_pred[row, col, box_loop, 0] = (col + explit_c_mine(Bbox_pred[row, col, box_loop, 0])) / W
                Bbox_pred[row, col, box_loop, 1] = (row + explit_c_mine(Bbox_pred[row, col, box_loop, 1])) / H
                Bbox_pred[row, col, box_loop, 2] = math.exp(Bbox_pred[row, col, box_loop, 2]) * anchors[2*box_loop + 0] / W
                Bbox_pred[row, col, box_loop, 3] = math.exp(Bbox_pred[row, col, box_loop, 3]) * anchors[2*box_loop + 1] / H

                for class_loop in range(C):
                    arr_max = max(arr_max, Classes[row, col, box_loop, class_loop])

                for class_loop in range(C):
                    Classes[row, col, box_loop, class_loop] = math.exp(Classes[row,col,box_loop,class_loop] - arr_max)
                    sum += Classes[row,col,box_loop,class_loop]

                for class_loop in range(C):
                    tempc = Classes[row, col, box_loop, class_loop] * Bbox_pred[row,col,box_loop,4]/sum
                    if(tempc > threshold):
                        probs[row, col, box_loop, class_loop] = tempc

    #NMS
    return NMS(np.ascontiguousarray(probs).reshape(H*W*B,C), np.ascontiguousarray(Bbox_pred).reshape(H*B*W, 5))
