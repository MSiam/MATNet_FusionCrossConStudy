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
    return mask_

def denorm(img):
    mean= [0.485, 0.456, 0.406]
    scale = [0.229, 0.224, 0.225]
    img = img.permute(1,2,0)
    img = img * torch.tensor(scale) + torch.tensor(mean)
    img = img.permute(2,0,1).cpu().numpy()
    img = np.asarray(img*255, np.uint8)
    return img

def trainIters(args):
    print(args)

    model_dir = os.path.join(args.ckpt_path, args.model_name)
    model_dir = model_dir
    make_dir(model_dir)

    # wandb init
    if args.wandb_run_name != '':
        run_name = args.wandb_run_name
        wandb_id_file_path = pathlib.Path(model_dir + '/' + run_name + '.txt')
        config = init_or_resume_wandb_run(wandb_id_file_path,
                                          entity_name=args.wandb_user,
                                          project_name=args.wandb_project,
                                          run_name=run_name,
                                          config=args)

    print("Center = ", args.center)
    epoch_resume = 0
    # if args.auto_resume:
    if len(glob.glob(model_dir + '/*')) > 3:
        epoch_resume, encoder_dict, decoder_dict, enc_opt_dict, dec_opt_dict, load_args = \
            load_last_checkpoint(model_dir, use_gpu=args.use_gpu)

        model = MATNet(args=args)
        encoder_dict, decoder_dict = check_parallel(encoder_dict, decoder_dict)
        model.encoder.load_state_dict(encoder_dict)
        model.decoder.load_state_dict(decoder_dict)
    elif hasattr(args, 'pretrained_weights'):
        args.pretrained_weights
        encoder_dict, decoder_dict, enc_opt_dict, dec_opt_dict, load_args = \
            load_checkpoint_epoch(args.model_name, args.pretrained_epoch, args.use_gpu,
                                  ckpt_path=args.pretrained_weights)

        model = MATNet(args=args)
        encoder_dict, decoder_dict = check_parallel(encoder_dict, decoder_dict)
        model.encoder.load_state_dict(encoder_dict, strict=False)
        model.decoder.load_state_dict(decoder_dict, strict=False)

    elif args.resume:
        encoder_dict, decoder_dict, enc_opt_dict, dec_opt_dict, load_args = \
            load_checkpoint_epoch(args.model_name, args.epoch_resume,
                                  args.use_gpu)

        epoch_resume = args.epoch_resume

        model = MATNet(args=args)

        encoder_dict, decoder_dict = check_parallel(encoder_dict, decoder_dict)
        model.encoder.load_state_dict(encoder_dict)
        model.decoder.load_state_dict(decoder_dict)
    else:
        model = MATNet(args=args)

    criterion = WeightedBCE2d()

    if args.use_gpu:
        model = nn.DataParallel(model, device_ids=args.gpu_id)
        device = torch.device("cuda")
        model.to(device)
        criterion.to(device)

    encoder_params = model.module.encoder.parameters()
    decoder_params = list(model.module.decoder.parameters())
    if type(encoder_params) == dict:
        if args.newweights_treated == 1:
            # Use higher weights for new params in Encoder
            decoder_params += encoder_params['new_weights']
            encoder_params = encoder_params['old_weights']
            enc_opt = get_optimizer(args.optim_cnn, args.lr_cnn, encoder_params,
                                args.weight_decay_cnn)

        elif args.newweights_treated == 2:
            # Train only new weights in Encoder
            for p in decoder_params:
                p.requires_grad = False
            for p in encoder_params['old_weights']:
                p.requires_grad = False

            decoder_params = encoder_params['new_weights']

            enc_opt = None
        else:
            # Original setup 2stream att + warping
            encoder_params = encoder_params['old_weights'] + encoder_params['new_weights']
            enc_opt = get_optimizer(args.optim_cnn, args.lr_cnn, encoder_params,
                                args.weight_decay_cnn)
    else:
        # Original setup Other models
        encoder_params = list(encoder_params)
        enc_opt = get_optimizer(args.optim_cnn, args.lr_cnn, encoder_params,
                            args.weight_decay_cnn)

    dec_opt = get_optimizer(args.optim, args.lr, decoder_params,
                            args.weight_decay)

    loaders = init_dataloaders(args)

    best_iou = 0
    cur_itrs = 0

    start = time.time()
    for e in range(epoch_resume, args.max_epoch):
        print("Epoch", e)
        epoch_losses = {'train': {'total': [], 'iou': [],
                                  'mask_loss': [], 'bdry_loss': []},
                        'val': {'total': [], 'iou': [],
                                'mask_loss': [], 'bdry_loss': []}}

        for split in ['train', 'val']:
            if split == 'train':
                model.train(True)
            else:
                model.train(False)

            for batch_idx, (image, flow, mask, bdry, negative_pixels, stylizedimg, path) in\
                    enumerate(loaders[split]):

                cur_itrs += 1
                image, flow, mask, bdry, negative_pixels = \
                    image.to(device), flow.to(device), mask.to(device), bdry.to(device),\
                    negative_pixels.to(device)

                if split == 'train':
                    mask_pred, p1, p2, p3, p4, p5 = model(image, flow)

                    mask_loss = criterion(mask_pred, mask, negative_pixels)
                    bdry_loss = criterion(p1, bdry, negative_pixels) + \
                                criterion(p2, bdry, negative_pixels) + \
                                criterion(p3, bdry, negative_pixels) + \
                                criterion(p4, bdry, negative_pixels) + \
                                criterion(p5, bdry, negative_pixels)
                    loss = mask_loss + 0.2 * bdry_loss

                    iou = db_eval_iou_multi(mask.cpu().detach().numpy(),
                                            mask_pred.cpu().detach().numpy())

                    if cur_itrs % args.vis_freq == 0:
                        bidx = 0
                        mask_vis = mask_denorm(mask[bidx])
                        mask_pred_vis = mask_denorm(mask_pred[bidx].detach())
                        neg_vis = mask_denorm(negative_pixels[bidx])

                        flow_vis = denorm(flow[bidx].cpu())
                        img_vis = denorm(image[bidx].cpu())
                        concat_img_mask = np.concatenate((img_vis, mask_vis, mask_pred_vis,
                                                          neg_vis), axis=2)

                        bdry_vis = mask_denorm(bdry[bidx])
                        p1_vis = mask_denorm(p1[bidx].detach())
                        p2_vis = mask_denorm(p2[bidx].detach())
                        p3_vis = mask_denorm(p3[bidx].detach())
                        p4_vis = mask_denorm(p4[bidx].detach())
                        p5_vis = mask_denorm(p5[bidx].detach())
                        concat_img_bdry = np.concatenate((bdry_vis, p1_vis, p2_vis, p3_vis, p4_vis,
                                                          p5_vis), axis=2)
                        if args.wandb_run_name != '':
                            wandb.log({'Mask Loss': mask_loss.detach().cpu(),
                                       'Boundary Loss': bdry_loss.detach().cpu(),
                                       'Total Loss': loss.detach().cpu(),
                                       'Train IoU': iou,
                                       'RGB | Mask GT | Mask Pred | Neg Pred': wandb.Image(concat_img_mask.transpose(1,2,0)),
                                       'Flow Input': wandb.Image(flow_vis.transpose(1,2,0)),
                                       'Image Boundaries': wandb.Image(concat_img_bdry.transpose(1,2,0)),
                                       })

                    dec_opt.zero_grad()
                    if enc_opt is not None:
                        enc_opt.zero_grad()
                    loss.backward()
                    if enc_opt is not None:
                        enc_opt.step()
                    dec_opt.step()
                else:
                    with torch.no_grad():
                        mask_pred, p1, p2, p3, p4, p5, _ = model(image, flow)

                        mask_loss = criterion(mask_pred, mask, negative_pixels)
                        bdry_loss = criterion(p1, bdry, negative_pixels) + \
                                    criterion(p2, bdry, negative_pixels) + \
                                    criterion(p3, bdry, negative_pixels) + \
                                    criterion(p4, bdry, negative_pixels) + \
                                    criterion(p5, bdry, negative_pixels)
                        loss = mask_loss + 0.2 * bdry_loss

                    iou = db_eval_iou_multi(mask.cpu().detach().numpy(),
                                            mask_pred.cpu().detach().numpy())

                epoch_losses[split]['total'].append(loss.data.item())
                epoch_losses[split]['mask_loss'].append(mask_loss.data.item())
                epoch_losses[split]['bdry_loss'].append(bdry_loss.data.item())
                epoch_losses[split]['iou'].append(iou)

                if (batch_idx) % args.print_every == 0:
                    mt = np.mean(epoch_losses[split]['total'])
                    mmask = np.mean(epoch_losses[split]['mask_loss'])
                    mbdry = np.mean(epoch_losses[split]['bdry_loss'])
                    miou = np.mean(epoch_losses[split]['iou'])

                    te = time.time() - start
                    print('Epoch: [{}/{}][{}/{}]\tTime {:.3f}s\tLoss: {:.4f}'
                          '\tMask Loss: {:.4f}\tBdry Loss: {:.4f}'
                          '\tIOU: {:.4f}'.format(e, args.max_epoch, batch_idx,
                                                 len(loaders[split]), te, mt,
                                                 mmask, mbdry, miou))

                    start = time.time()

                # break

        if args.wandb_run_name != '':
            wandb.log({'Ave %s Total Loss'%split: np.mean(epoch_losses[split]['total']),
                       'Ave %s Mask Loss'%split: np.mean(epoch_losses[split]['mask_loss']),
                        'Ave %s Boundary Loss'%split: np.mean(epoch_losses[split]['bdry_loss']),
                        'Ave %s mIoU'%split: np.mean(epoch_losses[split]['iou'])})


        miou = np.mean(epoch_losses['val']['iou'])
        if args.wandb_run_name != '':
            wandb.log({'Ave Val mIoU': miou})

        eps = 0.005
        if (miou - best_iou) > eps:
            best_iou = miou
            save_checkpoint_epoch(args, model,
                                  enc_opt, dec_opt, e, True)
            clean_checkpoint_dir(model_dir, e)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    args = read_and_merge_cfg(args)

    args.ckpt_path = args.ckpt_path

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


    trainIters(args)
