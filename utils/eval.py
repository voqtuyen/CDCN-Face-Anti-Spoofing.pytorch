import os
import torch
from torchvision import transforms
from PIL import ImageDraw


def add_visualization_to_tensorboard(cfg, epoch, img_batch, preds, targets, score, writer):
    """ Do the inverse transformation
    x = z*sigma + mean
      = (z + mean/sigma) * sigma
      = (z - (-mean/sigma)) / (1/sigma),
    Ref: https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/6
    """
    mean = [-cfg['dataset']['mean'][i] / cfg['dataset']['sigma'][i] for i in range(len(cfg['dataset']['mean']))]
    sigma = [1 / cfg['dataset']['sigma'][i] for i in range(len(cfg['dataset']['sigma']))]
    img_transform = transforms.Compose([
        transforms.Normalize(mean, sigma),
        transforms.ToPILImage()
    ])

    ts_transform = transforms.ToTensor()

    for idx in range(img_batch.shape[0]):
        vis_img = img_transform(img_batch[idx].cpu())
        ImageDraw.Draw(vis_img).text((0,0), 'pred: {} vs gt: {}'.format(int(preds[idx]), int(targets[idx])), (255,0,255))
        ImageDraw.Draw(vis_img).text((20,20), 'score {}'.format(score[idx]), (255,0,255))
        tb_img = ts_transform(vis_img)
        writer.add_image('Prediction visualization/{}'.format(idx), tb_img, epoch)


def predict(depth_map, threshold=0.5):
    """
    Convert depth_map estimation to true/fake prediction
    Args
        - depth_map: 32x32 depth_map
        - threshold: threshold between 0 and 1
    Return
        Predicted score
    """
    with torch.no_grad():
        score = torch.mean(depth_map, axis=(1,2))
        preds = (score >= threshold).type(torch.FloatTensor)

        return preds, score


def calc_accuracy(preds, targets):
    """
    Compare preds and targets to calculate accuracy
    Args
        - preds: batched predictions
        - targets: batched targets
    Return
        a single accuracy number
    """
    with torch.no_grad():
        equals = torch.mean(preds.eq(targets).type(torch.FloatTensor))
        return equals.item()