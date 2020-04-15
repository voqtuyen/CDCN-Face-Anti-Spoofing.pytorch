import os
import torch
from torchvision import transforms, models
from torch.utils.tensorboard import SummaryWriter
from datasets.FASDataset import FASDataset
from utils.transform import RandomGammaCorrection
from utils.utils import read_cfg, get_optimizer, get_device, build_network


cfg = read_cfg(cfg_file="config/CDCNpp_adam_lr1e-3.yaml")

device = get_device(cfg)

network = build_network(cfg)

optimizer = get_optimizer(cfg)

writer = SummaryWriter(cfg['log_dir'])

dump_input = torch.randn((1, 3, cfg['model']['input_size'][0], cfg['model']['input_size'][1]))

writer.add_graph(network, dump_input)

train_transform = transforms.Compose([
    RandomGammaCorrection(max_gamma=cfg['dataset']['augmentation']['gamma_correction'][1],
                            min_gamma=cfg['dataset']['augmentation']['gamma_correction'][0]),
    transforms.RandomResizedCrop(cfg['model']['input_size'][0]),
    transforms.ColorJitter(
        cfg['dataset']['augmentation']['brightness'],
        cfg['dataset']['augmentation']['contrast'],
        cfg['dataset']['augmentation']['saturation'],
        cfg['dataset']['augmentation']['hue']
    ),
    transforms.RandomRotation(cfg['dataset']['augmentation']['rotation_range']),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(cfg['model']['input_size']),
    transforms.ToTensor(),
    transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma'])
])


val_transform = transforms.Compose([
    transforms.Resize(cfg['model']['input_size']),
    transforms.ToTensor(),
    transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma'])
])

trainset = FASDataset(
    root_dir=cfg['dataset']['root'],
    csv_file=cfg['dataset']['train_set'],
    depth_map_size=cfg['model']['depth_map_size'],
    transform=train_transform,
    smoothing=cfg['train']['smoothing']
)

valset = FASDataset(
    root_dir=cfg['dataset']['root'],
    csv_file=cfg['dataset']['val_set'],
    depth_map_size=cfg['model']['depth_map_size'],
    transform=val_transform,
    smoothing=cfg['train']['smoothing']
)