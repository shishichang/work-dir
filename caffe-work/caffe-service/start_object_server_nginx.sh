#!/bin/sh
pid=`ps -ef|grep nginx|grep -v grep|grep -v start_object_server_nginx|awk '{print $2}'`
if [ "$pid"x != ""x ]; then
    kill -9 $pid
fi

num=`ps -ef|grep grpc_server|grep py|awk '{print $2}'`
if [ "$num"x != ""x ];then
    echo $num > server.pid
    while read pid
    do
        echo "kill -9 $pid"
        kill -9 $pid
    done < server.pid
    ps -ef|grep grpc_server|grep py
    num=`ps -ef|grep grpc_server|grep py|awk '{print $2}'`
    if [ "$num"x != ""x ];then
        echo "kill grpc_server err"
    fi
else
    echo "no need to kill grpc_server.py"
fi

sleep 2
cd /usr/local/nginx/sbin
./nginx > /tmp/nginx.log 2>&1 &
cd /proj/liuhz/image_process/ssd-detector/
#cd /root/ssc/git-lab/image_process/ssd-detector/
for i in 0 1 
do
    nohup python grpc_server.py 4500${i} $i > /tmp/grpc_server_4500${i}.log 2>&1 &
done
