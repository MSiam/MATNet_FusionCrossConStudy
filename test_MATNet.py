import torch
from torchvision import transforms

import os
import glob
from tqdm import tqdm
from PIL import Image
#from scipy.misc import imresize

from modules.MATNet import MATNet
from utils.utils import check_parallel
from utils.utils import load_checkpoint_epoch
from args import get_parser
from dataloader.inference_dataset import InferenceDataset
from utils.utils import get_optimizer, read_and_merge_cfg

def flip(x, dim):
    if x.is_cuda:
        return torch.index_select(x, dim, torch.arange(x.size(dim) - 1, -1, -1).long().cuda(0))
    else:
        return torch.index_select(x, dim, torch.arange(x.size(dim) - 1, -1, -1).long())


parser = get_parser()
args = parser.parse_args()
args = read_and_merge_cfg(args)

inputRes = (473, 473)
use_flip = True

to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
image_transforms = transforms.Compose([to_tensor, normalize])

model_name = 'MATNet' # specify the model name
epoch = args.ckpt_epoch # specify the epoch number
result_dir = args.result_dir

encoder_dict, decoder_dict, enc_opt_dict, dec_opt_dict, load_args =\
    load_checkpoint_epoch(model_name, epoch, True, False, args=args)
model = MATNet(args=args)
encoder_dict, decoder_dict = check_parallel(encoder_dict, decoder_dict)
model.encoder.load_state_dict(encoder_dict)
model.decoder.load_state_dict(decoder_dict, strict=False)

model.cuda()

model.train(False)

val_set = '/local/riemann/home/msiam/MoCA_filtered2/val.txt'
root_dir = '/local/riemann/home/msiam/MoCA_filtered2/JPEGImages/'
flow_dir = '/local/riemann/home/msiam/MoCA_filtered2/FlowImages_gap1/'

#val_set = 'data/DAVIS2017/val.txt'
#root_dir = 'data/DAVIS2017/JPEGImages/480p/'
#flow_dir = 'data/DAVIS2017/davis2017-flow/'


#val_set = '/mnt/zeta_share_1/public_share/Datasets/MoCA/MoCA_filtered/val.txt'
#root_dir = '/mnt/zeta_share_1/public_share/Datasets/MoCA/MoCA_filtered/JPEGImages/'
#flow_dir = '/mnt/zeta_share_1/public_share/Datasets/MoCA/MoCA_filtered/FlowImages_gap1/'
if not hasattr(args, 'imgseq'):
    args.imgseq = False


dataset = InferenceDataset(inputRes=inputRes, args=args, val_set=val_set, root_dir=root_dir, flow_dir=flow_dir,
                           image_transforms=image_transforms)

with torch.no_grad():
    for image, flow, width, height, video, imagefile in tqdm(dataset):
        image = image.unsqueeze(0)
        flow = flow.unsqueeze(0)

        image, flow = image.cuda(), flow.cuda()
        mask_pred, bdry_pred, p2, p3, p4, p5 = model(image, flow)

        if use_flip:
            flip_dim = 3

            image_flip = flip(image, 3)

            flow_flip = flip(flow, flip_dim)
            mask_pred_flip, bdry_pred_flip, p2, p3, p4, p5 =\
                model(image_flip, flow_flip)

            mask_pred_flip = flip(mask_pred_flip, 3)
            bdry_pred_flip = flip(bdry_pred_flip, 3)

            mask_pred = (mask_pred + mask_pred_flip) / 2.0
            bdry_pred = (bdry_pred + bdry_pred_flip) / 2.0

        mask_pred = mask_pred[0, 0, :, :]
        mask_pred = Image.fromarray(mask_pred.cpu().detach().numpy() * 255).convert('L')

        save_folder = '{}/{}_epoch{}/{}'.format(result_dir,
                                                model_name, epoch, video)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        save_file = os.path.join(save_folder,
                                 os.path.basename(imagefile)[:-4] + '.png')
        mask_pred = mask_pred.resize((width, height))
        mask_pred.save(save_file)
