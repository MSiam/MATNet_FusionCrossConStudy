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

### Download Datasets
We follow MATNet and use the following two public available dataset for training. Here are some steps to prepare the data:
- [DAVIS-17](https://davischallenge.org/davis2017/code.html): we use all the data in the train subset of DAVIS-16. 
    However, please download DAVIS-17 to fit the code. It will automatically choose the subset of DAVIS-16 for training. 
- [YoutubeVOS-2018](https://youtube-vos.org/dataset/): we sample the training data every 10 frames in YoutubeVOS-2018. We use the dataset version with 6fps rather than 30fps.
- Create soft links:
    ```cd data; ln -s your/davis17/path DAVIS2017; ln -s your/youtubevos/path YouTubeVOS_2018;```
    
### Prepare Edge Annotatios, HED Results and Optical Flow
Use MATNet instructions 

### Train
* Choose the right config:
    * Original matnet: configs/two_stream.yaml
    * reciprocal version with gated fusion: configs/two_stream_coatt_gating_recip.yaml

* Set correct checkpoint path in CONFIG you choose

* Run for training, it runs on two 1080TI GPUs.
```
CUDA_VISIBLE_DEVICES=2,3 python train_MATNet.py -cfg_file CONFIG -gpu_id 0 1 -wandb_run_name WANDB_RUN
```

## Test
```
python test_MATNet.py -ckpt_epoch BEST_EPOCH -ckpt_path CKPT_PATH -result_dir RESULT_DIR
```

## Inference and Evaluation on MoCA
```
bash scripts/eval_MoCA.sh CFG CKPT BEST_EPOCH MASK_RESULT_DIR GPU_ID CSV_RESULT_DIR
```
## Trained Models
For original MATNet use their provided models, for the reciprocal version with gated fusion that achieved best MoCA results use this [model]().

## References

* This repository heavily relies on [MATNet](https://github.com/tfzhou/MATNet) repo.
