# generate_label_1.py
# encoding:utf-8
file=open('/home/mx/tempfile/labels.txt','r') # ԭʼlabels.txt�ĵ�ַ
for eachline in file:
    data=eachline.strip().split(',')
    filename=data[4]
    filename=filename[:-4]
    txt_path='/home/mx/tempfile/label_txt/'+filename+'.txt' # ���ɵ�txt��ע�ļ���ַ
    txt=open(txt_path,'a')
    # new_line=data[5]+' '+data[0]+' '+data[1]+' '+data[2]+' '+data[3] ��ʹ��ԭʼͼƬ�ߴ磬�þ�ȡ��ע��
    # new_line=data[5]+' '+str(int(data[0])/2)+' '+str(int(data[1])/2)+' '+str(int(data[2])/2)+' '+str(int(data[3])/2) ��ʹ��1/4ͼƬ�ߴ磬�þ�ȡ��ע��
    txt.writelines(new_line)
    txt.write('\n')
    txt.close()
file.close()
print('generate label success')