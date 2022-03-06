CFG=$1
CKPT=$2
EPOCH=$3
RES=$4
GPU=$5
CSVRES=$6

CUDA_VISIBLE_DEVICES=$GPU python test_MATNet.py -gpu_id 0 -cfg_file $CFG -ckpt_path $CKPT -ckpt_epoch $EPOCH -result_dir $RES -use_flip
CUDA_VISIBLE_DEVICES=$GPU python MoCA_eval.py --masks_dir ${RES}/MATNet_epoch${EPOCH}/ --out_dir $CSVRES --MoCA_dir /local/riemann/home/msiam/MoCA_filtered2/ --MoCA_csv /local/riemann/home/msiam/MoCA/annotations.csv --thresh 0.1
