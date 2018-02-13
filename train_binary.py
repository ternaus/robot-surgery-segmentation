import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

import torch.backends.cudnn

from unet_models import UNet11, Loss
import utils
import cv2


class BinaryDataset(Dataset):
    def __init__(self, file_names: list, to_augment=False):
        self.file_names = file_names
        self.to_augment = to_augment

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img = load_image(str(self.file_names[idx]))
        mask = load_mask(str(self.file_names[idx]).replace('images', 'binary_masks'))

        if self.to_augment:
            img, mask = augment(img, mask)

        return utils.img_transform(img), torch.from_numpy(np.expand_dims(mask, 0)).float()


def augment(img, mask):
    if np.random.random() < 0.5:
        img = np.fliplr(img)
        mask = np.fliplr(mask)

    if np.random.random() < 0.5:
        img = np.flipud(img)
        mask = np.flipud(mask)

    if np.random.random() < 0.5:
        if np.random.random() < 0.5:
            img = randomHueSaturationValue(img,
                                           hue_shift_limit=(-50, 50),
                                           sat_shift_limit=(-5, 5),
                                           val_shift_limit=(-15, 15))

    return img.copy(), mask.copy()


def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255)):
    """
    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-50, 50),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))

    :param image:
    :param hue_shift_limit:
    :param sat_shift_limit:
    :param val_shift_limit:
    :param u:
    :return:
    """

    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(image)
    hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
    h = cv2.add(h, hue_shift)
    sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
    s = cv2.add(s, sat_shift)
    val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
    v = cv2.add(v, val_shift)
    image = cv2.merge((h, s, v))
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

    return image


def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path):
    return (cv2.imread(str(path), 0) > 0).astype(np.uint8)


def validation(model: nn.Module, criterion, valid_loader) -> Dict[str, float]:
    model.eval()
    losses = []

    jaccard = []

    for inputs, targets in valid_loader:
        inputs = utils.variable(inputs, volatile=True)
        targets = utils.variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.data[0])
        jaccard += [get_jaccard(targets, (outputs > 0).float()).data[0]]

    valid_loss = np.mean(losses)  # type: float

    valid_jaccard = np.mean(jaccard)

    print('Valid loss: {:.5f}, jaccard: {:.5f}'.format(valid_loss, valid_jaccard))
    metrics = {'valid_loss': valid_loss, 'jaccard_loss': valid_jaccard}
    return metrics


def get_jaccard(y_true, y_pred):
    epsilon = 1e-15
    intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1)
    union = y_true.sum(dim=-2).sum(dim=-1) + y_pred.sum(dim=-2).sum(dim=-1)

    return (intersection / (union - intersection + epsilon)).mean()


def get_file_names(fold: int):
    folds = {0: [1, 3],
             1: [2, 5],
             2: [4, 8],
             3: [6, 7]}

    train_path = Path('data') / 'cropped_train'

    train_file_names = []
    val_file_names = []

    for instrument_id in range(1, 9):
        if instrument_id in folds[fold]:
            val_file_names += list((train_path / ('instrument_dataset_' + str(instrument_id)) / 'images').glob('*'))
        else:
            train_file_names += list((train_path / ('instrument_dataset_' + str(instrument_id)) / 'images').glob('*'))

    return train_file_names, val_file_names


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--jaccard-weight', type=float)
    arg('--device-ids', type=str, default='0', help='For example 0,1 to run on two GPUs')
    arg('--fold', type=int, help='fold', default=0)
    arg('--root', default='runs/debug', help='checkpoint root')
    arg('--batch-size', type=int, default=1)
    arg('--n-epochs', type=int, default=100)
    arg('--lr', type=float, default=0.0001)
    arg('--workers', type=int, default=8)

    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    model = UNet11(pretrained='vgg')

    if torch.cuda.is_available():
        if args.device_ids:
            device_ids = list(map(int, args.device_ids.split(',')))
        else:
            device_ids = None
        model = nn.DataParallel(model, device_ids=device_ids).cuda()

    loss = Loss()

    def make_loader(file_names, to_augment=False, shuffle=False):
        return DataLoader(
            dataset=BinaryDataset(file_names, to_augment=to_augment),
            shuffle=shuffle,
            num_workers=args.workers,
            batch_size=args.batch_size,
            pin_memory=torch.cuda.is_available()
        )

    train_file_names, val_file_names = get_file_names(args.fold)

    train_loader = make_loader(train_file_names, to_augment=True, shuffle=True)
    valid_loader = make_loader(val_file_names)

    root.joinpath('params.json').write_text(
        json.dumps(vars(args), indent=True, sort_keys=True))

    utils.train(
        init_optimizer=lambda lr: Adam(model.parameters(), lr=lr),
        args=args,
        model=model,
        criterion=loss,
        train_loader=train_loader,
        valid_loader=valid_loader,
        validation=validation,
        fold=args.fold
    )


if __name__ == '__main__':
    main()
