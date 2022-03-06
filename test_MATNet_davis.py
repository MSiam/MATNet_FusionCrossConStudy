import torch
from torch.utils import data
from torchvision import transforms
import torch.nn as nn

import os
import sys
import time
import random
import numpy as np
import glob
from tqdm import tqdm

import torch.backends.cudnn as cudnn
from modules.MATNet import MATNet
from args import get_parser
from utils.utils import make_dir, check_parallel, init_or_resume_wandb_run
from utils.utils import get_optimizer, read_and_merge_cfg
from dataloader.dataset_utils import get_dataset_davis_youtube_ehem
from utils.utils import save_checkpoint_epoch, load_checkpoint_epoch, load_last_checkpoint, clean_checkpoint_dir
from utils.objectives import WeightedBCE2d
from measures.jaccard import db_eval_iou_multi
import pathlib
import wandb

def init_dataloaders(args):
    loaders = {}

    # init dataloaders for training and validation
    split = 'val'
    batch_size = args.batch_size
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    image_transforms = transforms.Compose([to_tensor, normalize])
    target_transforms = transforms.Compose([to_tensor])

    dataset = get_dataset_davis_youtube_ehem(
        args, split=split, image_transforms=image_transforms,
        target_transforms=target_transforms,
        augment=args.augment and split == 'train',
        inputRes=(473, 473), stylized=args.mot_biased)

    shuffle = False
    loader = data.DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=args.num_workers,
                             drop_last=True)

    return loader

def infer(args):
    print(args)

    model_dir = os.path.join(args.ckpt_path, args.model_name)
    model_dir = model_dir
    make_dir(model_dir)

    if len(glob.glob(model_dir + '/*')) > 3:
        epoch_resume, encoder_dict, decoder_dict, enc_opt_dict, dec_opt_dict, load_args = \
            load_last_checkpoint(model_dir, use_gpu=args.use_gpu)

        model = MATNet(args=args)
        encoder_dict, decoder_dict = check_parallel(encoder_dict, decoder_dict)
        model.encoder.load_state_dict(encoder_dict)
        model.decoder.load_state_dict(decoder_dict)

    if args.use_gpu:
        model = nn.DataParallel(model, device_ids=args.gpu_id)
        device = torch.device("cuda")
        model.to(device)

    loader = init_dataloaders(args)

    ious = []

    start = time.time()
    model.train(False)

    for batch_idx, (image, flow, mask, bdry, negative_pixels, stylizedimg, path) in enumerate(tqdm(loader)):
        if args.frame_nb > 1:
            flow = flow.permute(0, 2, 1, 3, 4)
            image = image.permute(0, 2, 1, 3, 4)

        image, flow, mask, bdry, negative_pixels = \
            image.to(device), flow.to(device), mask.to(device), bdry.to(device),\
            negative_pixels.to(device)

        with torch.no_grad():
            mask_pred, p1, p2, p3, p4, p5, _ = model(image, flow)

        iou = db_eval_iou_multi(mask.cpu().detach().numpy(),
                                mask_pred.cpu().detach().numpy())

        ious.append(iou)

    print('mIoU ', np.mean(ious))

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    args = read_and_merge_cfg(args)

    args.ckpt_path = 'ckpts/' + args.ckpt_path

    args.model_name = 'MATNet'
    args.max_epoch = 25
    args.year = '2016'

    gpu_id = args.gpu_id
    print('gpu_id: ', gpu_id)
    print('use_gpu: ', args.use_gpu)
    if args.use_gpu and len(gpu_id) == 1:
        gpu_id = gpu_id[0]
        torch.cuda.set_device(device=gpu_id)
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    infer(args)
