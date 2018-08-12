from pathlib import Path
import argparse
import cv2
import numpy as np
from tqdm import tqdm

from prepare_data import height, width, h_start, w_start


def general_dice(y_true, y_pred):
    result = []

    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    for instrument_id in set(y_true.flatten()):
        if instrument_id == 0:
            continue
        result += [dice(y_true == instrument_id, y_pred == instrument_id)]

    return np.mean(result)


def general_jaccard(y_true, y_pred):
    result = []

    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    for instrument_id in set(y_true.flatten()):
        if instrument_id == 0:
            continue
        result += [jaccard(y_true == instrument_id, y_pred == instrument_id)]

    return np.mean(result)


def jaccard(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)


def dice(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--train_path', type=str, default='data/cropped_train',
        help='path where train images with ground truth are located')
    arg('--target_path', type=str, default='predictions/unet11', help='path with predictions')
    arg('--problem_type', type=str, default='parts', choices=['binary', 'parts', 'instruments'])
    args = parser.parse_args()

    result_dice = []
    result_jaccard = []

    if args.problem_type == 'binary':
        for instrument_id in tqdm(range(1, 9)):
            instrument_dataset_name = 'instrument_dataset_' + str(instrument_id)

            for file_name in (
                    Path(args.train_path) / instrument_dataset_name / 'binary_masks').glob('*'):
                y_true = (cv2.imread(str(file_name), 0) > 0).astype(np.uint8)

                pred_file_name = (Path(args.target_path) / 'binary' / instrument_dataset_name / file_name.name)

                pred_image = (cv2.imread(str(pred_file_name), 0) > 255 * 0.5).astype(np.uint8)
                y_pred = pred_image[h_start:h_start + height, w_start:w_start + width]

                result_dice += [dice(y_true, y_pred)]
                result_jaccard += [jaccard(y_true, y_pred)]

    elif args.problem_type == 'parts':
        for instrument_id in tqdm(range(1, 9)):
            instrument_dataset_name = 'instrument_dataset_' + str(instrument_id)
            for file_name in (
                    Path(args.train_path) / instrument_dataset_name / 'parts_masks').glob('*'):
                y_true = cv2.imread(str(file_name), 0)

                pred_file_name = Path(args.target_path) / 'parts' / instrument_dataset_name / file_name.name

                y_pred = cv2.imread(str(pred_file_name), 0)[h_start:h_start + height, w_start:w_start + width]

                result_dice += [general_dice(y_true, y_pred)]
                result_jaccard += [general_jaccard(y_true, y_pred)]

    elif args.problem_type == 'instruments':
        for instrument_id in tqdm(range(1, 9)):
            instrument_dataset_name = 'instrument_dataset_' + str(instrument_id)
            for file_name in (
                    Path(args.train_path) / instrument_dataset_name / 'instruments_masks').glob('*'):
                y_true = cv2.imread(str(file_name), 0)

                pred_file_name = Path(args.target_path) / 'instruments' / instrument_dataset_name / file_name.name

                y_pred = cv2.imread(str(pred_file_name), 0)[h_start:h_start + height, w_start:w_start + width]

                result_dice += [general_dice(y_true, y_pred)]
                result_jaccard += [general_jaccard(y_true, y_pred)]

    print('Dice = ', np.mean(result_dice), np.std(result_dice))
    print('Jaccard = ', np.mean(result_jaccard), np.std(result_jaccard))
