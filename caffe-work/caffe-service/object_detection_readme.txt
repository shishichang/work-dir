客户端可调用服务器的接口有两个：
Judge：传入单张图像，返回单张图像结果
JudgeBatch：支持传入多张图像，并行批量处理，返回全部结果

Judge传入数据格式为ObjectReq, 返回JudgeRep
JudgeBatch传入数据为ObjectReqList, 返回JudgeRepList
详细数据格式定义见Object.proto
输入图像预先变换尺寸到500X500，传入多张图像时，一次最大传入数量为50
建议使用JudgeBatch，批量处理平均用时46ms，串行处理方式平均用时58ms

可以一次传送最大图像数量：99张。

服务器端检测模型分别支持20，80，200类物体检测，目前设定模型为200类

返回检测结果数据格式：
 
judge_rep_list {
    batch_name: test.mp4            #传入批量的名称
    img_num: 2                      #图片数量

    batch_state: 1                  #服务返回状态
    
	judge_rep {                     
      img_name: "image/001150.jpg"  #图像名称
      state: 1                      #图像返回状态
	  obj_num: 2                    #图像中检测到物体数目
      object_info {
        label: 58                   #物体类别编号
        category: "dog"             #物体类别名称
        score: 0.99240642786        #检测得分
        x1: 0.341052383184          #bounding box左上角坐标
        y1: 0.222472250462
        x2: 0.798786044121          #bounding box右下角坐标
        y2: 0.998941481113
      }
      object_info {                 #该图像中检测到的另一个物体信息
        label: 124
        category: "person"
        score: 0.105609208345
        x1: 0.0919093936682
    	y1: 0.0
        x2: 0.502002418041
        y2: 0.439175218344
      }
    }
			 
    judge_rep {                     #另一张图像的检测结果
      img_name: "image/test.jpg"    
      state: -1                    
	  obj_num: 0                    
    }
}
