import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskMSE(nn.Module):
    """
    A masked version of MSE, designed for flow.
    It only cares about valid pixels, and ignores errors at pixels with out-of-bound values.
    Ignore conditions: value == 0: typically caused by zero padding in grid_sample
                       value > 1.0 or value < -1.0: typically happened when flow indices went out of range

    """
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        mse = F.mse_loss(input, target, reduction="none")

        mask = torch.ones(mse.shape, dtype=torch.bool).to(mse.device)
        input_mask = input != 0
        target_mask = target != 0
        mask = mask & input_mask & target_mask

        input_mask = torch.abs(input) <= 1
        target_mask = torch.abs(target) <= 1
        mask = mask & input_mask & target_mask
        mask = mask.detach().flatten()

        mse = mse.flatten()
        return mse[mask].mean()
