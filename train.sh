PORT=30001
GPU=$1
CFG=$2
PTH=$3
TAG=${4:-'default'}

torchrun --nproc_per_node $GPU \
	--master_port $PORT main.py \
	--cfg $CFG \
	--data-path $PTH \
	--amp --tag $TAG
