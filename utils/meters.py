import os
import numpy as np
import torch


class AvgMeter():
    def __init__(self, writer, name, num_iter_per_epoch, per_iter_vis=False):
        self.writer = writer
        self.name = name
        self.num_iter_per_epoch = num_iter_per_epoch
        self.per_iter_vis = per_iter_vis


    def reset(self, epoch):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.epoch = epoch

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count if self.count !=0 else 0
        if self.per_iter_vis:
            self.writer.add_scalar(self.name, self.avg, self.epoch * self.num_iter_per_epoch + self.count - 1)
        else:
            if self.count == self.num_iter_per_epoch - 1:
                self.writer.add_scalar(self.name, self.avg, self.epoch)