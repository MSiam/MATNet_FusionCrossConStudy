##  Fusion Tyes and Cross Connection Study
Official Implementation used for training MATNet variants in our CVPR2022 Work.
We provide the best model on MoCA improving over state of the art performance with 3%.

## MoCA Results
Our paper uses threshold 0.2 and [SOA](https://charigyang.github.io/motiongroup/) MoCA comparison uses threshold 0.1.

* MATNet variants results without additiona YouTube-VOS data (NoYTB) and without Boundary Aware Refinement module (NoBAR)
_____
Method | th | Flip | mIoU | SR_0.5 | SR_0.6 | SR_0.7 | SR_0.8 | SR_0.9 | mSR |
FusionSeg Modified | 0.2 | No | 42.3 | 47.9 | 43.6 | 35.9 | 24.2 | 9.4 | 39.2|
RTNet | 0.2 | No | 60.7 | 67.9 | 62.4 | 53.6 | 43.4 | 23.9 | 50.2 |
MATNet reproduced | 0.1 | No | 66.9 | 76.5 | 70.8 | 60.9 | 46.0 | 23.5 | 55.5|
MATNet NoBAR | 0.2 | No | 65.1 | 73.6 | 68.0 | 58.9 | 44.7 | 21.5 |  53.3|
MATNet NoYTB | 0.2 | No | 54.7 | 59.9 | 53.5 | 44.0 | 31.0 | 13.4 | 40.3 |
_____

* Training reciprocal cross connections (cross connections similar to RTNet) with gated fusion (fusion similar to MATNet), which achieves best performance on MoCA.
_____
Method | th | Flip | mIoU | SR_0.5 | SR_0.6 | SR_0.7 | SR_0.8 | SR_0.9 | mSR|
NonRecip CC + Gated Fusion | 0.1 | Yes |
NonRecip CC + Gated Fusion | 0.2 | Yes |
Recip CC + Gated Fusion | 0.2 | Yes | 70.6 | 81.2 | 75.5 | 65.0 | 48.1 | 23.0 |
Recip CC + Gated Fusion | 0.1 | Yes | 67.6 | 77.9 | 70.1 | 59.1 | 40.7 | 16.8 |
_____

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
For original MATNet use their provided models, for the reciprocal version with gated fusion that achieved best MoCA results use this [model](https://www.dropbox.com/s/8eoqdbf1d6kaxli/ckpt_cctype_coatt_gating_recip.zip?dl=0).

## References

* This repository heavily relies on [MATNet](https://github.com/tfzhou/MATNet) repo.
