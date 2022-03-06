#!/bin/sh

# This is the script to perform training, the goal is that code in
# this script can be safely preempted. Jobs in slurm queue are scheduled
# and preempted according to their priorities. Note that even the job with
# deadline queue can be preempted over time so it is crucial that you
# checkpoint your program state for every fixed interval: e.g 10 mins.

# Vector provides a fast parallel filesystem local to the GPU nodes,  dedicated
# for checkpointing. It is mounted under /checkpoint. It is strongly
# recommended that you keep your intermediary checkpoints under this directory
# i.e. /checkpoint/${USER}/${SLURM_JOB_ID}

# We also recommend users to create a symlink of the checkpoint dir so your
# training code stays the same with regards to different job IDs and it would
# be easier to navigate the checkpoint directory under your job's working directory
#ln -sfn /checkpoint/${USER}/${SLURM_JOB_ID} $PWD/checkpoint


# In the future, the checkpoint directory will be removed immediately after the
# job has finished. If you would like the file to stay longer, and create an
# empty "delay purge" file as a flag so the system will delay the removal for
# 48 hours
#touch /checkpoint/${USER}/${SLURM_JOB_ID}/DELAYPURGE

# prepare the environment, here I am using environment modules, but you could
# select the method of your choice (but note that code in ~/.bash_profile or
# ~/.bashrc will not be executed with a new job)
source ~/.bashrc
conda activate moca
#module purge && module load  vector_cv_project

# Then we run our training code, using the checkpoint dir provided the code
# demonstrates how to perform checkpointing in pytorch, please navigate to the
# file for more information.
#python main.py --cfg cfg/ucf24.yaml --save_dir /checkpoint/${USER}/2550120 \
python train_MATNet.py -gpu_id 0 1 -num_workers 8 -lr 0.001 -batch_size 6 \
-ckpt_path two_stream_frame_10_sample_1 -wandb_run_name two_stream_frame_10_sample_1 \
-cfg_file configs/two_stream/r101_frame_10_sample_1.yaml