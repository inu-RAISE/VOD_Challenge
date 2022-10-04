set -ex

DATAROOT=$1
FILE_NAME=$2
MODE=$3

if [[ $MODE == "train" ]]; then
	python CUT/train.py --dataroot $DATAROOT --name $FILE_NAME --CUT_mode CUT --phase train --n_epochs 100 --batch 2 --gpu_ids 0,1 --preprocess resize --num_threads 0
elif [[ $MODE == "test" ]]; then
	python CUT/test.py --dataroot $DATAROOT --name $FILE_NAME --CUT_mode CUT --phase test --gpu_ids 0,1 --preprocess resize --num_threads 0 --num_test 10000000
fi