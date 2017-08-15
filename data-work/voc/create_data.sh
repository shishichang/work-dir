cur_dir=/data1/shishch/work-dir/data-work/voc
root_dir=/data1/shishch/caffe-snow

cd $root_dir

redo=1
data_root_dir="/data2/datasets-obj"
my_dir="/VOC/VOCdevkit"
dataset_name="VOC"
mapfile="$cur_dir/labelmap_voc.prototxt"
anno_type="detection"
label_type="txt"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0

extra_cmd="--encode-type=jpg --encoded"
if [ $redo ]
then
  extra_cmd="$extra_cmd --redo"
fi
for subset in test train
do
  python $root_dir/scripts/create_annoset.py --anno-type=$anno_type --label-type=$label_type --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir $data_root_dir/$my_dir/$subset.txt $data_root_dir/$dataset_name/$db/$dataset_name"_"$subset"_"$db $cur_dir/$dataset_name
done
