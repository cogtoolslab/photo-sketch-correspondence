import torch
import numpy as np


def proj_kps(flows, kps, image_size):
    """
    Project src keypoints to the dst image using estimated flows.
    :param flows: Shape: N, H, W, 2
    :param kps: Shape: N, 8, 2
    :param image_size: Size of image. Default: 256.
    :return: The projected keypoints.
    """
    N, H, W, _ = flows.shape
    flows = (flows + 1) * (image_size - 1) / 2
    flows = flows.reshape(N, H * W, 2)
    kps = torch.round(kps).long()
    kps = kps.reshape(N, 8, 2)
    kps = kps[:, :, 0] * image_size + kps[:, :, 1]
    kps = kps.unsqueeze(-1).expand(-1, -1, 2)
    dst = torch.gather(flows, 1, kps).flip(dims=(-1,))
    return dst


def compute_pck(gt_kps, pred_kps, image_size):
    """
    Compute PCK@15, PCK@0.10, PCK@0.05
    :param gt_kps: groundtruth keypoints annotation
    :param pred_kps: predicted keypoints
    :param image_size: size of image
    :return: PCK@10, PCK@5
    """
    diff = np.sqrt(np.sum(np.square(np.array(gt_kps) - np.array(pred_kps)), axis=-1))  # N, 8
    pck10 = np.sum(diff <= image_size * 0.1, axis=1) / 8
    pck05 = np.sum(diff <= image_size * 0.05, axis=1) / 8
    return pck10, pck05
