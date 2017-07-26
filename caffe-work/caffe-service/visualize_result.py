#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function


def print_batch_results(response):
    print('ObjectBatchReply:')
    print('img_num: {}'.format(response.img_num))
#    print('batch_name: {}'.format(response.batch_id))
    for k, v in enumerate(response.category_detail):
        print('category_name:{}, times: {}'.format(v.category, v.times))
    for k, v in enumerate(response.frame_object):
        print('image_id:{}'.format(v.img_id))
        print('obj_num:{}'.format(v.obj_num))
        for h, w in enumerate(v.object_info):
            print('label:{}'.format(w.label))
            print('category:{}'.format(w.category))
            print('score:{}'.format(w.score))
            print('x1:{}'.format(w.x1))
            print('x2:{}'.format(w.x2))
            print('y1:{}'.format(w.y1))
            print('y2:{}'.format(w.y2))

