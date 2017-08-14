#########################################################################
# File Name: ./get_loss.sh
# Author: ma6174
# mail: ma6174@163.com
# Created Time: Fri Aug 11 01:58:52 2017
#########################################################################
#!/bin/bash
cat finetune.log | grep "Train net output" |awk '{print $11}' |tee''""
