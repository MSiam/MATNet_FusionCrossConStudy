"""
Code from https://github.com/sssdddwww2/CVPR2021_VSPW_Implement
"""
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
#from models.sync_batchnorm import SynchronizedBatchNorm2d
#BatchNorm2d = SynchronizedBatchNorm2d
from torch.nn import BatchNorm2d as BatchNorm2d
from RAFT_core.raft import RAFT
from RAFT_core.utils.utils import InputPadder
from collections import OrderedDict

def flowwarp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        grid = grid.to(x.device)
    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)
    output = nn.functional.grid_sample(x, vgrid,align_corners=False)

    return output



def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )
class FlowCNN(nn.Module):
    def __init__(self):
        super(FlowCNN,self).__init__()
        self.conv1 = conv3x3_bn_relu(11,16)
        self.conv2 = conv3x3_bn_relu(16,32)
        self.conv3 = conv3x3_bn_relu(32,2)
        self.conv4 = conv3x3_bn_relu(4,2)
    def forward(self,img1,img2,flow):
        x = torch.cat([flow,img1,img2,img2-img1],1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.cat([flow,x],1)
        x = self.conv4(x)
        return x



class NetWarp(nn.Module):
    def __init__(self, args):
        super(NetWarp, self).__init__()

        self.raft = RAFT()
        to_load = torch.load('./RAFT_core/raft-things.pth')
        new_state_dict = OrderedDict()
        for k, v in to_load.items():
            name = k[7:] # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
            new_state_dict[name] = v #新字典的key值对应的value为一一对应的值。
        self.raft.load_state_dict(new_state_dict)
        ####
        self.mean=torch.FloatTensor([0.485, 0.456, 0.406])
        self.std=torch.FloatTensor([0.229, 0.224, 0.225])
        ####
        self.args= args
        self.warp_config = args.warp_config

        self.flowcnn=FlowCNN()
        channels = {'res2': 256, 'res3': 512, 'res4': 1024, 'res5': 2048}
        self.indices = {'res2': 0, 'res3': 1, 'res4': 2, 'res5': 3}

        for layer, chs in channels.items():
            setattr(self, 'w0_0_%s'%layer, nn.Parameter(torch.FloatTensor(chs), requires_grad=True))
            getattr(self, 'w0_0_%s'%layer).data.fill_(1.0)

            setattr(self, 'w0_1_%s'%layer, nn.Parameter(torch.FloatTensor(chs), requires_grad=True))
            getattr(self, 'w0_1_%s'%layer).data.fill_(0.0)

    def compute_flow(self, images):
        c_img = images[:, :, -1]
        warping_flows = []
        for i in range(images.shape[2]):
            c_pre_img = images[:, :, i]
            mean= self.mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            mean = mean.to(c_img.device)
            mean = mean.expand_as(c_img)
            std = self.std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            std = std.to(c_img.device)
            std = std.expand_as(c_img)

            h, w = c_img.shape[-2:]
            c_img_f = ((c_img*std)+mean)*255.
            c_pre_img_f = (c_pre_img*std+mean)*255.
            with torch.no_grad():
                self.raft.eval()
                padder = InputPadder((h,w))
                c_img_f_ = padder.pad(c_img_f)
                c_pre_img_f_ = padder.pad(c_pre_img_f)
                _,flow = self.raft(c_img_f_,c_pre_img_f_,iters=20, test_mode=True)
                flow = padder.unpad(flow)

            #########
            flow = self.flowcnn(c_img_f, c_pre_img_f, flow)
            warping_flows.append(flow.unsqueeze(2))
        return torch.stack(warping_flows, dim=2).squeeze(3)

    def forward(self, flow, feats, layer=None):
        clip_num = feats.shape[2]
        flow = F.interpolate(flow, feats.shape[-2:], mode='bilinear', align_corners=True)
        new_feats = flowwarp(feats, flow)
        if self.warp_config.get('residual_warp', False):
            assert layer is not None
            w0_0 = getattr(self, 'w0_0_%s'%layer)
            w0_1 = getattr(self, 'w0_1_%s'%layer)
            new_feats = w0_0.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand_as(feats).to(feats.device) * feats + \
                            w0_1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand_as(new_feats).to(new_feats.device) * new_feats

        return new_feats
