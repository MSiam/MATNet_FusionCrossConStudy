##  Fusion Types and Cross Connection Study
Official Implementation used for training MATNet variants in our CVPR2022 Work.
We provide the best models on MoCA with 70.6% mean IoU.

## MoCA Results
Our paper uses threshold 0.2 and [SOA](https://charigyang.github.io/motiongroup/) MoCA comparison uses threshold 0.1. We recommend for better reporting on MoCA to compute area under the curve for different thresholds but is out of our current work scope.

* MATNet variants results without additional YouTube-VOS data (NoYTB) and without Boundary Aware Refinement module (NoBAR)

|Method   | Th  | Flip  | mIoU  | SR_0.5  | SR_0.6  | SR_0.7  | SR_0.8  | SR_0.9  | mSR  |
|---|---|---|---|---|---|---|---|---|---|
|FusionSeg Modified | 0.2 | No | 42.3 | 47.9 | 43.6 | 35.9 | 24.2 | 9.4 | 39.2|
|RTNet | 0.2 | No | 60.7 | 67.9 | 62.4 | 53.6 | 43.4 | 23.9 | 50.2 |
|MATNet reproduced | 0.2 | No | 67.3 | 75.9 | 70.8 | 61.9 | 48.6 | 26.0 | 56.6|
|MATNet NoBAR | 0.2 | No | 65.1 | 73.6 | 68.0 | 58.9 | 44.7 | 21.5 |  53.3|
|MATNet NoYTB | 0.2 | No | 54.7 | 59.9 | 53.5 | 44.0 | 31.0 | 13.4 | 40.3 |

* In the main submission we found RTNet with reciprocal cross connections heavily static biased. Our additional experiments here shows that Reciprocal connections (motion-to-appearance and appearance-to-motion) can encourage dynamics if trained with proper fusion and training data without pretraining towards saliency. Training reciprocal cross connections (cross connections similar to RTNet) with gated fusion (fusion similar to MATNet), achieves best performance on MoCA and shows increase in dynamic bias unlike original RTNet. RTNet convex combination gated fusion has shown to cause accuracy degradation on the other hand.

|Method   | Th  | Flip  | mIoU  | SR_0.5  | SR_0.6  | SR_0.7  | SR_0.8  | SR_0.9  | mSR  |
|---|---|---|---|---|---|---|---|---|---|
NonRecip CC + Gated Fusion | 0.1 | Yes | 70.2| 79.4 | 74.1 | 64.6 | 49.0 | 23.8| 58.2|
NonRecip CC + Gated Fusion | 0.2 | Yes | 68.5| 77.3| 72.2| 63.5| 50.6| 27.0 | 58.1|
Recip CC + Gated Fusion | 0.2 | Yes | 70.6 | 81.2 | 75.5 | 65.0 | 48.1 | 23.0 | 58.6|
Recip CC + Gated Fusion | 0.1 | Yes | 67.6 | 77.9 | 70.1 | 59.1 | 40.7 | 16.8 | 52.9|

* Results showing the static dynamic bias for the new Recip CC + Gated Fusion for the final fusion layer w.r.t other models.
<div align="center">
<img src="https://github.com/MSiam/MATNet_FusionCrossConStudy/blob/main/figures/static_dynamic_recip.png" width="40%" height="40%"><br><br>
</div>

* Results showing the mIoU on MoCA when masking top-K units per factor. It aligns with the previous results that fusion layer 2 is dynamic biased, while fusion layers 3,4 and 5 are static biased. In case of sampling random units, we select the (K+5%) of the least units that are biased towards the significant (i.e. dynamic in fusion layer 2 and static in the rest) and then randomly select within these. We do that to ensure random selection especially with higher percentages does not sample some of the units biased towards the corresponding significant factor either static or dyanmic.
<div align="center">
<img src="https://github.com/MSiam/MATNet_FusionCrossConStudy/blob/main/figures/static_dynamic_masking_units.png" width="40%" height="40%"><br><br>
</div>

## Installation

The training and testing experiments are conducted using Python 3.7 PyTorch 1.9 with multi GPU support.
Other minor Python modules can be installed by running

```bash
pip install -r requirements.txt
```

## Datasets

### Download Datasets
We follow MATNet and use the following two public available dataset for training. Here are some steps to prepare the data:
- [DAVIS-17](https://davischallenge.org/davis2017/code.html): we use all the data in the train subset of DAVIS-16. 
    However, please download DAVIS-17 to fit the code. It will automatically choose the subset of DAVIS-16 for training. 
- [YoutubeVOS-2018](https://youtube-vos.org/dataset/): we sample the training data every 10 frames in YoutubeVOS-2018. We use the dataset version with 6fps rather than 30fps.
- Create soft links:
    ```cd data; ln -s your/davis17/path DAVIS2017; ln -s your/youtubevos/path YouTubeVOS_2018;```
    
### Prepare Edge Annotatios, HED Results and Optical Flow
Use MATNet instructions from [here](https://github.com/tfzhou/MATNet)

### Prepare MoCA for Evaluation
Follow motiongrouping instructions from [here](https://github.com/charigyang/motiongrouping/)

## Train
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
For original MATNet use their provided models and for the reciprocal version with gated fusion that achieved best MoCA results use this [model](https://www.dropbox.com/s/8eoqdbf1d6kaxli/ckpt_cctype_coatt_gating_recip.zip?dl=0).

## BibTeX
If you find this repository useful, please consider citing our work :t-rex:


      @InProceedings{kowal2022deeper,
       title={A Deeper Dive Into What Deep Spatiotemporal Networks Encode: Quantifying Static vs. Dynamic Information},
       author={Kowal, Matthew and Siam, Mennatullah and Islam, Md Amirul and Bruce, Neil and Wildes, Richard P. and Derpanis, Konstantinos G.},
       booktitle={Conference on Computer Vision and Pattern Recognition},
       year={2022}
     }
     
## References

* This repository heavily relies on [MATNet](https://github.com/tfzhou/MATNet) repo.
