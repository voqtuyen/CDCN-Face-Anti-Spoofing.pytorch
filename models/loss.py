import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


def contrast_depth_conv(input, device):
    ''' compute contrast depth in both of (out, label) '''
    '''
        input  32x32
        output 8x32x32
    '''

    kernel_filter_list =[
                        [[1,0,0],[0,-1,0],[0,0,0]], [[0,1,0],[0,-1,0],[0,0,0]], [[0,0,1],[0,-1,0],[0,0,0]],
                        [[0,0,0],[1,-1,0],[0,0,0]], [[0,0,0],[0,-1,1],[0,0,0]],
                        [[0,0,0],[0,-1,0],[1,0,0]], [[0,0,0],[0,-1,0],[0,1,0]], [[0,0,0],[0,-1,0],[0,0,1]]
                        ]
    
    kernel_filter = np.array(kernel_filter_list, np.float32)
    
    kernel_filter = torch.from_numpy(kernel_filter.astype(np.float)).float().to(device)
    # weights (in_channel, out_channel, kernel, kernel)
    kernel_filter = kernel_filter.unsqueeze(dim=1)
    
    input = input.unsqueeze(dim=1).expand(input.shape[0], 8, input.shape[1],input.shape[2])
    
    contrast_depth = F.conv2d(input, weight=kernel_filter, groups=8)  # depthwise conv
    
    return contrast_depth


class ContrastDepthLoss(nn.Module):    # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self, device):
        super(ContrastDepthLoss, self).__init__()
        self.device = device


    def forward(self, out, label): 
        '''
        compute contrast depth in both of (out, label),
        then get the loss of them
        tf.atrous_convd match tf-versions: 1.4
        '''
        contrast_out = contrast_depth_conv(out, device=self.device)
        contrast_label = contrast_depth_conv(label, device=self.device)

        criterion_MSE = nn.MSELoss()
    
        loss = criterion_MSE(contrast_out, contrast_label)
    
        return loss


class DepthLoss(nn.Module):
    def __init__(self, device):
        super(DepthLoss, self).__init__()
        self.criterion_absolute_loss = nn.MSELoss()
        self.criterion_contrastive_loss = ContrastDepthLoss(device=device)


    def forward(self, predicted_depth_map, gt_depth_map):
        absolute_loss = self.criterion_absolute_loss(predicted_depth_map, gt_depth_map)
        contrastive_loss = self.criterion_contrastive_loss(predicted_depth_map, gt_depth_map)
        return absolute_loss + contrastive_loss
