import yaml
import torch
from torch import optim
from models.CDCNs import CDCN, CDCNpp


def read_cfg(cfg_file):
    """
    Read configurations from yaml file
    Args:
        cfg_file (.yaml): path to cfg yaml
    Returns:
        (dict): configuration in dict
    """
    with open(cfg_file, 'r') as rf:
        cfg = yaml.safe_load(rf)
        return cfg


def get_optimizer(cfg, network):
    """ Get optimizer based on the configuration
    Args:
        cfg (dict): a dict of configuration
        network: network to optimize
    Returns:
        optimizer 
    """
    optimizer = None
    if cfg['train']['optimizer'] == 'adam':
        optimizer = optim.Adam(network.parameters(), lr=cfg['train']['lr'])
    else:
        raise NotImplementedError

    return optimizer


def get_device(cfg):
    """ Get device based on configuration
    Args: 
        cfg (dict): a dict of configuration
    Returns:
        torch.device
    """
    device = None
    if cfg['device'] == '':
        device = torch.device("cpu")
    elif cfg['device'] == '0':
        device = torch.device("cuda:0")
    elif cfg['device'] == '1':
        device = torch.device("cuda:1")
    else:
        raise NotImplementedError
    return device


def build_network(cfg):
    """ Build the network based on the cfg
    Args:
        cfg (dict): a dict of configuration
    Returns:
        network (nn.Module) 
    """
    network = None

    if cfg['model']['base'] == 'CDCNpp':
        network = CDCNpp()
    elif cfg['model']['base'] == 'CDCN':
        network = CDCN()
    else:
        raise NotImplementedError

    return network