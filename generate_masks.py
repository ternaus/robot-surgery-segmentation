"""
Script generates predictions, splitting original images into tiles, and assembling prediction back together
"""
import argparse
from prepare_train_val import get_split
from dataset import data_path, RoboticsDataset
import cv2
from models import UNet16, LinkNet34, UNet11
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
import utils
from torch.utils.data import DataLoader
from torch.nn import functional as F
from prepare_data import (original_height,
                          original_width,
                          h_start, w_start
                          )

from transforms import (ImageOnly,
                        Normalize,
                        DualCompose)

img_transform = DualCompose([
    ImageOnly(Normalize())
])


def get_model(model_path, model_type='unet11'):
    """

    :param model_path: 'UNet16', 'UNet11', 'LinkNet34'
    :param model_type:
    :return:
    """
    if model_type == 'UNet16':
        model = UNet16()
    elif model_type == 'UNet11':
        model = UNet11()
    elif model_type == 'LinkNet34':
        model = LinkNet34(num_classes=1)

    state = torch.load(str(model_path))
    state = {key.replace('module.', ''): value for key, value in state['model'].items()}
    model.load_state_dict(state)

    if torch.cuda.is_available():
        return model.cuda()

    model.eval()

    return model


def predict(model, from_file_names, batch_size: int, to_path):
    loader = DataLoader(
        dataset=RoboticsDataset(from_file_names, transform=img_transform, mode='predict', problem_type='binary'),
        shuffle=False,
        batch_size=batch_size,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available()
    )

    for batch_num, (inputs, stems) in enumerate(tqdm(loader, desc='Predict')):
        inputs = utils.variable(inputs, volatile=True)
        outputs = F.sigmoid(model(inputs))
        mask = (outputs.data.cpu().numpy() * 255).astype(np.uint8)

        for i, image_name in enumerate(stems):
            t_mask = mask[i, 0]

            h, w = t_mask.shape

            full_mask = np.zeros((original_height, original_width))
            full_mask[h_start:h_start + h, w_start:w_start + w] = t_mask

            cv2.imwrite(str(to_path / (stems[i] + '.png')), full_mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model_path', type=str, default='data/models/linknet34_60', help='path to model folder')
    arg('--model_type', type=str, default='UNet11', help='network architecture',
        choices=['UNet11', 'UNet16', 'LinkNet34'])
    arg('--output_path', type=str, help='path to save images', default='.')
    arg('--batch-size', type=int, default=4)
    arg('--fold', type=int, default=0)
    arg('--workers', type=int, default=8)

    args = parser.parse_args()

    _, file_names = get_split(args.fold)
    model = get_model(str(Path(args.model_path).joinpath('model_{fold}.pt'.format(fold=args.fold))),
                      model_type='LinkNet34')

    print('num file_names = {}'.format(len(file_names)))

    output_path = Path(args.output_path) / str(args.fold)
    output_path.mkdir(exist_ok=True, parents=True)

    predict(model, file_names, args.batch_size, output_path)
