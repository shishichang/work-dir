from . import yolo2deploy  
from os.path import basename

class framework(object):
    
    def __init__(self, meta):
        model = basename(meta['model'])
        model = '.'.join(model.split('.')[:-1])
        meta['name'] = model
        
        self.meta = meta

    def is_inp(self, file_name):
        return True

class YOLOv2(framework):
    constructor = yolo2deploy.constructor
    preprocess = yolo2deploy.predict.preprocess
    is_inp = yolo2deploy.misc.is_inp
    postprocess = yolo2deploy.predict.postprocess
    resize_input = yolo2deploy.predict.resize_input
    findboxes = yolo2deploy.predict.findboxes
    process_box = yolo2deploy.predict.process_box

"""
framework factory
"""
types = {
    '[region]': YOLOv2
}

def create_framework(meta):
    net_type = meta['type']
    print(net_type)
    this = types.get(net_type, framework)
    return this(meta)
