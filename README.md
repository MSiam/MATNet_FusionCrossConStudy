##  Fusion Tyes and Cross Connection Study
Official Implementation used for training MATNet variants in our CVPR2022 Work.
We provide the best model on MoCA improving over state of the art performance with 3%.

## Installation

The training and testing experiments are conducted using Python 3.7 PyTorch 1.9 with multi GPU support.
Other minor Python modules can be installed by running

```bash
pip install -r requirements.txt
```

## Train

### Clone
```git clone --recursive https://github.com/tfzhou/MATNet.git```

### Download Datasets
In the paper, we use the following two public available dataset for training. Here are some steps to prepare the data:
- [DAVIS-17](https://davischallenge.org/davis2017/code.html): we use all the data in the train subset of DAVIS-16. 
    However, please download DAVIS-17 to fit the code. It will automatically choose the subset of DAVIS-16 for training. 
- [YoutubeVOS-2018](https://youtube-vos.org/dataset/): we sample the training data every 10 frames in YoutubeVOS-2018. We use the dataset version with 6fps rather than 30fps.
- Create soft links:

    ```cd data; ln -s your/davis17/path DAVIS2017; ln -s your/youtubevos/path YouTubeVOS_2018;```
    
### Prepare Edge Annotations
I have provided some matlab scripts to generate edge annotations from mask. Please run ```data/run_davis2017.m``` 
and ```data/run_youtube.m```.

### Prepare HED Results
I have provided the pytorch codes to generate HED results for the two datasets (see ```3rdparty/pytorch-hed```).
Please run ```run_davis.py``` and ```run_youtube.py```. 

The codes are borrowed from https://github.com/sniklaus/pytorch-hed. 

### Prepare Optical Flow
I have provided the pytorch codes to generate optical flow results for the two datasets (see ```3rdparty/pytorch-pwc```).
Please run ```run_davis_flow.py``` and ```run_youtubevos_flow.py```. 

The codes are borrowed from https://github.com/sniklaus/pytorch-pwc. 
Please follow the [setup](https://github.com/sniklaus/pytorch-pwc#setup) section to install ```cupy```. 

`warning: Total size of optical flow results of Youtube-VOS is more than 30GB.`

### Train
Once all data is prepared, please run for training.
```
python train_MATNet.py -gpu_id GPU -vis_port VISDOMPORT -num_workers NWORKERS -batch_size BATCH -ckpt_path CKPT_PATH -frame_nb 3DMATNET
```

* Train multigpu
```
CUDA_VISIBLE_DEVICES=2,3 python train_MATNet.py -gpu_id 0 1 -vis_port 1025 -num_workers 2 -batch_size 2 -ckpt_path multigpu_reproduce/
```

## Test
```
python test_MATNet.py -ckpt_epoch 16 -ckpt_path multigpu_reproduce/ -result_dir
```


## Segmentation Results

1. The segmentation results on DAVIS-16 and Youtube-objects can be downloaded from [Google Drive](https://drive.google.com/file/d/1d23TGBtrr11g8KFAStwewTyxLq2nX4PT/view?usp=sharing).
2. The segmentation results on DAVIS-17 __val__ can be downloaded from [Google Drive](https://drive.google.com/open?id=1GTqjWc7tktw92tBNKln2eFmb9WzdcVrz). We achieved __58.6__ in terms of _Mean J&F_.
3. The segmentation results on DAVIS-17 __test-dev__ can be downloaded from [Google Drive](https://drive.google.com/file/d/1Ood-rr0d4YRFSrGGh6yVpYvOvE_h0tVK/view?usp=sharing). We achieved __59.8__ in terms of _Mean J&F_. The method also achieved the second place in DAVIS-20 unsupervised object segmentation challenge. Please refer to [paper](https://davischallenge.org/challenge2020/papers/DAVIS-Unsupervised-Challenge-2nd-Team.pdf) for more details of our challenge solution.

## Pretrained Models

The pre-trained model can be downloaded from [Google Drive](https://drive.google.com/file/d/1XlenYXgQjoThgRUbffCUEADS6kE4lvV_/view?usp=sharing).

## References

This repository heavily relies on [MATNet](https://github.com/tfzhou/MATNet) repo.
