# txt_to_xml.py
# txt_to_xml.py
# encoding:utf-8
# ����һ��������XML Schema��ʹ��DOM������ʽ�ӿհ��ļ�����һ��XML
from xml.dom.minidom import Document
import cv2
import os

def generate_xml(name,split_lines,img_size,class_ind):
    doc = Document()  # ����DOM�ĵ�����

    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    title = doc.createElement('folder')
    title_text = doc.createTextNode('CITY')
    title.appendChild(title_text)
    annotation.appendChild(title)

    img_name=name+'.jpg'

    title = doc.createElement('filename')
    title_text = doc.createTextNode(img_name)
    title.appendChild(title_text)
    annotation.appendChild(title)

    source = doc.createElement('source')
    annotation.appendChild(source)

    title = doc.createElement('database')
    title_text = doc.createTextNode('The CITY Database')
    title.appendChild(title_text)
    source.appendChild(title)

    title = doc.createElement('annotation')
    title_text = doc.createTextNode('CITY')
    title.appendChild(title_text)
    source.appendChild(title)

    size = doc.createElement('size')
    annotation.appendChild(size)

    title = doc.createElement('width')
    title_text = doc.createTextNode(str(img_size[1]))
    title.appendChild(title_text)
    size.appendChild(title)

    title = doc.createElement('height')
    title_text = doc.createTextNode(str(img_size[0]))
    title.appendChild(title_text)
    size.appendChild(title)

    title = doc.createElement('depth')
    title_text = doc.createTextNode(str(img_size[2]))
    title.appendChild(title_text)
    size.appendChild(title)

    for split_line in split_lines:
        line=split_line.strip().split()
        if line[0] in class_ind:
            object = doc.createElement('object')
            annotation.appendChild(object)

            title = doc.createElement('name')
            title_text = doc.createTextNode(line[0])
            title.appendChild(title_text)
            object.appendChild(title)

            bndbox = doc.createElement('bndbox')
            object.appendChild(bndbox)
            title = doc.createElement('xmin')
            title_text = doc.createTextNode(str(int(float(line[1]))))
            title.appendChild(title_text)
            bndbox.appendChild(title)
            title = doc.createElement('ymin')
            title_text = doc.createTextNode(str(int(float(line[2]))))
            title.appendChild(title_text)
            bndbox.appendChild(title)
            title = doc.createElement('xmax')
            title_text = doc.createTextNode(str(int(float(line[3]))))
            title.appendChild(title_text)
            bndbox.appendChild(title)
            title = doc.createElement('ymax')
            title_text = doc.createTextNode(str(int(float(line[4]))))
            title.appendChild(title_text)
            bndbox.appendChild(title)

    # ��DOM����docд���ļ�
    f = open('Annotations/'+name+'.xml','w')
    f.write(doc.toprettyxml(indent = ''))
    f.close()

if __name__ == '__main__':
    class_ind=('Pedestrian', 'Car', 'Truck','Cyclist')
    cur_dir=os.getcwd()
    labels_dir=os.path.join(cur_dir,'Labels')
    for parent, dirnames, filenames in os.walk(labels_dir): # �ֱ�õ���Ŀ¼����Ŀ¼�͸�Ŀ¼���ļ�   
        for file_name in filenames:
            full_path=os.path.join(parent, file_name) # ��ȡ�ļ�ȫ·��
            f=open(full_path)
            split_lines = f.readlines()
            name= file_name[:-4] # ����λ����չ��.txt��ֻȡǰ����ļ���
            img_name=name+'.jpg' 
            img_path=os.path.join('/home/its/data/CITYdevkit/CITY/JPEGImages/',img_name) # ·����Ҫ�����޸�            
            img_size=cv2.imread(img_path).shape
            generate_xml(name,split_lines,img_size,class_ind)
print('all txts has converted into xmls')