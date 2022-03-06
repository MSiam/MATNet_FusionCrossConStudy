from torch.utils import data
import torch
from dataloader.utils import _sample_flow_seq

import os
import glob
import numpy as np
from PIL import Image

class InferenceDataset(data.Dataset):
    '''
    Dataset for Inference
    '''

    def __init__(self, args, inputRes, image_transforms=None, val_set='', root_dir='', flow_dir=''):
        self.image_transforms = image_transforms
        self.inputRes = inputRes
        self.args = args

        self.frame_nb = args.frame_nb - 1 if args.frame_nb > 1 else 1
        self.sampling_rate = args.sampling_rate

        self.imagefiles = []
        self.flowfiles = []

        self.metainfo = []
        self.imgsbyvid = {}

        self.imagefiles, self.flowfiles, self.videos = self._load_dataset(val_set, root_dir, flow_dir)

    def _load_dataset(self, val_set, root_dir, root_flow_dir):
        with open(val_set) as f:
            seqs = f.readlines()
            seqs = [seq.strip().split(' ')[0].split('/')[-2] for seq in seqs]
        seqs = sorted(list(set(seqs)))

        self.metainfo = []
        self.imgsbyvid = {}

        imagefiles, flowfiles, videos = [], [], []
        for video in seqs:
            image_dir = os.path.join(root_dir, video)
            flow_dir = os.path.join(root_flow_dir, video)

            ifiles = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))[:-1]
            imagefiles += ifiles
            ffiles = sorted(glob.glob(os.path.join(flow_dir, '*.png')))
            flowfiles += ffiles

            self.imgsbyvid[video] = (ffiles, ifiles)
            self.metainfo += [(video, i) for i in range(len(ffiles))]
            videos += [video]*len(ffiles)
        return imagefiles, flowfiles, videos

    def __len__(self):
        return len(self.imagefiles)

    def __getitem__(self, index):
        imagefile = self.imagefiles[index]
        flowfile = self.flowfiles[index]
        video = self.videos[index]

        image = Image.open(imagefile).convert('RGB')
        flow = Image.open(flowfile).convert('RGB')
        if self.frame_nb > 1:
            flow_seq, img_seq = _sample_flow_seq(
                self.metainfo[index], self.imgsbyvid, self.frame_nb, self.sampling_rate
            )

        width, height = image.size

        image = image.resize(self.inputRes)
        flow = flow.resize(self.inputRes)

        image = self.image_transforms(image)
        flow = self.image_transforms(flow)

        if self.frame_nb > 1:
            for i in range(self.frame_nb):
                flow_seq[i] = np.array(flow_seq[i].resize(self.inputRes))
                flow_seq[i] = self.image_transforms(flow_seq[i])

                img_seq[i] = np.array(img_seq[i].resize(self.inputRes))
                img_seq[i] = self.image_transforms(img_seq[i])

            flow_seq.append(flow)
            img_seq.append(image)

            flow = torch.stack(flow_seq)
            if self.args.imgseq:
                image = torch.stack(img_seq)

        return image, flow, width, height, video, imagefile
