"""
[1] Merge masks with different instruments into one binary mask
[2] Crop black borders from images and masks
"""
from pathlib import Path

from tqdm import tqdm
import cv2
import numpy as np

data_path = Path('data')

train_path = data_path / 'train'

cropped_train_path = data_path / 'cropped_train'

height, width = 1024, 1280
h_start, w_start = 28, 320

for instrument_index in range(1, 9):
    instrument_folder = 'instrument_dataset_' + str(instrument_index)

    (cropped_train_path / instrument_folder / 'images').mkdir(exist_ok=True, parents=True)
    (cropped_train_path / instrument_folder / 'binary_masks').mkdir(exist_ok=True, parents=True)

    mask_folders = (train_path / instrument_folder / 'ground_truth').glob('*')
    mask_folders = [x for x in mask_folders if 'Other' not in str(mask_folders)]

    for file_name in tqdm(list((train_path / instrument_folder / 'left_frames').glob('*'))):
        img = cv2.imread(str(file_name))
        old_h, old_w, _ = img.shape

        img = img[h_start: h_start + height, w_start: w_start + width]
        cv2.imwrite(str(cropped_train_path / instrument_folder / 'images' / file_name.name), img)

        mask = np.zeros((old_h, old_w))

        for mask_folder in mask_folders:
            mask += cv2.imread(str(mask_folder / file_name.name), 0)

        mask = (mask[h_start: h_start + height, w_start: w_start + width] > 0).astype(np.uint8) * 255

        cv2.imwrite(str(cropped_train_path / instrument_folder / 'binary_masks' / file_name.name), mask)
