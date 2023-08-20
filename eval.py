import argparse
import os
import numpy as np

import torch
import torch.nn.functional as F

import models.resnet_cbn as resnet_cbn
import models.resnet_orig as resnet_orig
import models.moco as moco

from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets.photo_sketch_dataset import PhotoSketchDataset
from models.PSCNet import PSCNet
from utils.pck import proj_kps, compute_pck

############################
# Argument parser
parser = argparse.ArgumentParser(description='PyTorch Training')

# data
parser.add_argument('--csv-path', metavar='DIR',
                    help='root path to csv files')
parser.add_argument('--data-path', metavar='DIR',
                    help='root path to dataset')

# job
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')

# model arch
parser.add_argument('--arch', metavar='ARCH', default='resnet18',
                    choices=['resnet18', 'resnet50', 'resnet101'],
                    help='model architecture')
parser.add_argument('--layer', default=[2, 3], nargs='*', type=int,
                    help='resnet blocks used for similarity measurement')
parser.add_argument('--no-cbn', action='store_false', dest='cbn',
                    help='not use conditional batchnorm')

# checkpoint
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint; resume the entire model')

args = parser.parse_args()

############################
# Initialization

test_csv = os.path.join(args.csv_path, "test_pairs_ps.csv")

dataset = PhotoSketchDataset(test_csv, args.data_path, mode="test")
dataloader = DataLoader(dataset, batch_size=1, num_workers=4)

print("Dataset loaded.")

# import the original or the conditional BN version of ResNet
if args.cbn:
    resnet = resnet_cbn
else:
    resnet = resnet_orig

model = PSCNet(moco.MoCo, resnet.__dict__[args.arch], dim=128, K=8192, corr_layer=args.layer).cuda()

checkpoint = torch.load(args.checkpoint)
state_dict = checkpoint['state_dict']

for k in list(state_dict.keys()):
    if "module." in k:
        state_dict[k[len("module."):]] = state_dict[k]
        del state_dict[k]

msg = model.load_state_dict(state_dict, strict=False)
assert len(msg.missing_keys) == 0 and len(msg.unexpected_keys) == 0
model = model.cuda().eval()

print("Model loaded.")

############################
# Computation

image_size = 256
with torch.no_grad():
    pck05_list = []
    pck10_list = []

    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        photo, sketch, photo_kps, sketch_kps = data

        photo = photo.cuda(non_blocking=True)
        sketch = sketch.cuda(non_blocking=True)

        # get feature maps
        _, photo_res = model.encoder_q(photo, cond=0, return_map=True)
        _, sketch_res = model.encoder_q(sketch, cond=1, return_map=True)

        # estimate displacement field
        fwd_flow, bwd_flow = model.forward_stn(photo_res, sketch_res)
        fwd_flow = F.interpolate(fwd_flow.permute(0, 3, 1, 2), (image_size, image_size),
                                 mode="bilinear", align_corners=True).permute(0, 2, 3, 1).cpu()
        bwd_flow = F.interpolate(bwd_flow.permute(0, 3, 1, 2), (image_size, image_size),
                                 mode="bilinear", align_corners=True).permute(0, 2, 3, 1).cpu()

        # project keypoints & compute error
        pred_sketch_kps = proj_kps(bwd_flow, photo_kps, image_size)
        pck10, pck05 = compute_pck(sketch_kps, pred_sketch_kps, image_size)
        pck10_list.append(pck10)
        pck05_list.append(pck05)

        pred_photo_kps = proj_kps(fwd_flow, sketch_kps, image_size)
        pck10, pck05 = compute_pck(photo_kps, pred_photo_kps, image_size)
        pck10_list.append(pck10)
        pck05_list.append(pck05)

pck10_list = np.concatenate(pck10_list, axis=0).mean()
pck05_list = np.concatenate(pck05_list, axis=0).mean()

print("pck@0.10:", pck10_list)
print("pck@0.05:", pck05_list)
