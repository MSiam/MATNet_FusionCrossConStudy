import torch
from torch.utils import data
from torchvision import transforms

from tqdm import tqdm
import cv2
import os
import sys
import time
import random
import numpy as np

from modules.MATNet import Encoder, Decoder
from args import get_parser
from utils.utils import get_optimizer
from utils.utils import make_dir, check_parallel
from dataloader.dataset_utils import get_dataset_davis_youtube_ehem
from utils.utils import save_checkpoint_epoch, load_checkpoint_epoch
from utils.objectives import WeightedBCE2d
from measures.jaccard import db_eval_iou_multi
from visualizer import Visualizer

def init_dataloaders(args):
    loaders = {}

    # init dataloaders for training and validation
    for split in ['train', 'val']:
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

        shuffle = True if split == 'train' else False
        loaders[split] = data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=shuffle,
                                         num_workers=args.num_workers,
                                         drop_last=True)

    return loaders

def mask_denorm(mask):
    mask_ = mask.cpu().repeat(3,1,1) * 255
    return mask_.permute(1,2,0)

def denorm(img):
    mean= [0.485, 0.456, 0.406]
    scale = [0.229, 0.224, 0.225]
    img = img.permute(1,2,0)
    img = img * torch.tensor(scale) + torch.tensor(mean)
    img = img.permute(2,0,1).cpu().numpy()
    img = np.asarray(img*255, np.uint8)
    return img.transpose(1,2,0)

def debug_loader(args):
    print(args)

    loaders = init_dataloaders(args)

    if not os.path.exists('tmp'):
        os.mkdir('tmp')

    split = 'train'
    for idx, (image, flow, mask, bdry, negative_pixels, stylizedimg) in tqdm(enumerate(loaders[split])):
        image, flow, mask, bdry, negative_pixels = \
            image.cuda(), flow.cuda(), mask.cuda(), bdry.cuda(),\
            negative_pixels.cuda()

        for batch_idx in range(image.shape[0]):
            img_vis = denorm(image[batch_idx].cpu())

            flow_vis = []
            for frameid in range(flow.shape[1]):
                flow_vis.append(denorm(flow[batch_idx, frameid].cpu()))
            mask_vis = mask_denorm(mask[batch_idx])
            neg_vis = mask_denorm(negative_pixels[batch_idx])

            concat_img_mask = np.concatenate((img_vis, mask_vis, neg_vis), axis=1)
            concat_flow = np.concatenate(flow_vis, axis=1)

            cv2.imwrite('tmp/concat_img_%05d_%02d.png'%(idx, batch_idx), concat_img_mask)
            cv2.imwrite('tmp/concat_flow_%05d_%02d.png'%(idx, batch_idx), concat_flow)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    args.model_name = 'MATNet'
    args.batch_size = 2
    args.max_epoch = 25
    args.year = '2016'

    gpu_id = args.gpu_id
    print('gpu_id: ', gpu_id)
    print('use_gpu: ', args.use_gpu)
    if args.use_gpu:
        torch.cuda.set_device(device=gpu_id)
        torch.cuda.manual_seed(args.seed)
    debug_loader(args)
