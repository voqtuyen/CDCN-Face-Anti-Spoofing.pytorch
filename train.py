import os
import torch
from torchvision import transforms, models
from torch.utils.tensorboard import SummaryWriter
from datasets.FASDataset import FASDataset
from utils.transform import RandomGammaCorrection
from utils.utils import read_cfg, get_optimizer, get_device, build_network
from trainer.FASTrainer import FASTrainer
from models.loss import DepthLoss
from torch.optim.lr_scheduler import StepLR


cfg = read_cfg(cfg_file="config/CDCNpp_adam_lr1e-3.yaml")

device = get_device(cfg)

network = build_network(cfg)

optimizer = get_optimizer(cfg, network)

lr_scheduler = StepLR(optimizer=optimizer, step_size=30, gamma=0.1)

criterion = DepthLoss(device=device)

writer = SummaryWriter(cfg['log_dir'])

dump_input = torch.randn((1, 3, cfg['model']['input_size'][0], cfg['model']['input_size'][1]))

writer.add_graph(network, dump_input)

train_transform = transforms.Compose([
    RandomGammaCorrection(max_gamma=cfg['dataset']['augmentation']['gamma_correction'][1],
                            min_gamma=cfg['dataset']['augmentation']['gamma_correction'][0]),
    transforms.RandomResizedCrop(cfg['model']['input_size'][0]),
    # transforms.ColorJitter(
    #     brightness=cfg['dataset']['augmentation']['brightness'],
    #     contrast=cfg['dataset']['augmentation']['contrast'],
    #     saturation=cfg['dataset']['augmentation']['saturation'],
    #     hue=cfg['dataset']['augmentation']['hue']
    # ),
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

trainloader = torch.utils.data.DataLoader(
    dataset=trainset,
    batch_size=cfg['train']['batch_size'],
    shuffle=True,
    num_workers=2
)

valloader = torch.utils.data.DataLoader(
    dataset=valset,
    batch_size=cfg['val']['batch_size'],
    shuffle=True,
    num_workers=2
)

trainer = FASTrainer(
    cfg=cfg, 
    network=network,
    optimizer=optimizer,
    criterion=criterion,
    lr_scheduler=lr_scheduler,
    device=device,
    trainloader=trainloader,
    valloader=valloader,
    writer=writer
)

trainer.train()

writer.close()