CC=g++
CCFLAG=-std=c++0x -O3 -Wall
#CCFLAG=-std=c++0x -g -O0 -Wall

cuda_include=/usr/local/cuda/include
cuda_lib=/usr/local/cuda/lib

caffe_path=/root/ssc/caffe-ssd
caffe_include=$(caffe_path)/include
caffe_build_include=$(caffe_path)/build/include
caffe_lib=$(caffe_path)/build/lib

opencv_include=/usr/local/include/opencv

include=-I/usr/local/include -I/usr/include/atlas -I$(cuda_include) -I$(opencv_include) -I$(caffe_include) -I$(caffe_build_include)
lib=-lprotobuf -lglog -lpthread -lboost_system -lboost_filesystem -lopencv_highgui -lopencv_core -lopencv_imgproc -lopencv_calib3d $(caffe_path)/build/lib/libcaffe.so -lrt -Wl,-rpath,/usr/local/lib:$(caffe_path)/build/lib:$(cuda_lib)

object1=detection.o main.o


all:$(object1)
	$(CC) -o detection $(object1) $(lib)

detection.o:detection.cpp detection.h
	$(CC) -DUSE_OPENCV -c detection.cpp $(CCFLAG) $(include)

main.o:main.cpp detection.h time_helper.h
	$(CC) -c -DUSE_OPENCV main.cpp $(CCFLAG) $(include)

clean:
	rm -f detection
	rm -f *.o
