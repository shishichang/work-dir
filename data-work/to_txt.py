# generate_label_1.py
# encoding:utf-8
file=open('/home/mx/tempfile/labels.txt','r') # 原始labels.txt的地址
for eachline in file:
    data=eachline.strip().split(',')
    filename=data[4]
    filename=filename[:-4]
    txt_path='/home/mx/tempfile/label_txt/'+filename+'.txt' # 生成的txt标注文件地址
    txt=open(txt_path,'a')
    # new_line=data[5]+' '+data[0]+' '+data[1]+' '+data[2]+' '+data[3] 如使用原始图片尺寸，该句取消注释
    # new_line=data[5]+' '+str(int(data[0])/2)+' '+str(int(data[1])/2)+' '+str(int(data[2])/2)+' '+str(int(data[3])/2) 如使用1/4图片尺寸，该句取消注释
    txt.writelines(new_line)
    txt.write('\n')
    txt.close()
file.close()
print('generate label success')