import cv2
from os.path import join, isdir
import numpy as np
from os import listdir


MODES = ["ORIGINAL", "BINARY", "PARTS", "INSTRUMENTS"]
MODELS = ["GROUND_TRUTH", "LINKNET", "UNET", "UNET11", "UNET16"]

SCALE = 0.125
GAP = 5
LEFT, TOP = 320, 28
RIGHT, BOTTOM = LEFT + 1280, TOP + 1024

N_ROWS = len(MODES)
N_COLUMNS = len(MODELS)

W = int((RIGHT - LEFT) * SCALE)
H = int((BOTTOM - TOP) * SCALE)

THRESHOLD = 128
WHITE, RED, GREEN, BLUE, YELLOW, PURPLE, CYAN, LIME, MAGENTA = (255, 255, 255), (0, 0, 255), (0, 255, 0), \
                                                               (255, 0, 0), (0, 255, 255), (128, 0, 128), \
                                                               (255, 255, 0), (0, 255, 0), (255, 0, 255)
# type_labels = {
#     "Bipolar": MAGENTA,
#     "Prograsp": GREEN,
#     "Needle": BLUE,
#     "Sealer": YELLOW,
#     "Grasp": PURPLE,
#     "Scissors": LIME,
#     "Other": CYAN,
# }
type_labels = {
    "Bipolar": MAGENTA,
    "Prograsp": WHITE,
    "Needle": PURPLE,
    "Sealer": YELLOW,
    "Grasp": BLUE,
    "Scissors": LIME,
    "Other": RED,
}
type_labels_preds = {
    1 * 32: type_labels["Bipolar"],
    2 * 32: type_labels["Prograsp"],
    3 * 32: type_labels["Needle"],
    4 * 32: type_labels["Sealer"],
    5 * 32: type_labels["Grasp"],
    6 * 32: type_labels["Scissors"],
    7 * 32: type_labels["Other"],
}
part_labels = {
    10: BLUE,
    20: RED,
    30: MAGENTA,
    # 40: GREEN,
}
part_labels_preds = {
    1 * 85: part_labels[10],
    2 * 85: part_labels[20],
    3 * 85: part_labels[30],
}
binary_label = YELLOW


def get_picture(mode, model):
    if mode == "ORIGINAL":
        image = cv2.imread(join(PHOTO_DIR, FRAME))
    else:
        image = None
        if model == "GROUND_TRUTH":
            for l_d, v in zip(LABEL_DIRS, type_labels.values()):
                ovr = cv2.imread(join(GROUND_TRUTH_DIR, l_d, FRAME))
                if image is None: image = np.zeros_like(ovr)
                if mode == "BINARY":
                    image[np.all(ovr > 0, axis=-1)] = binary_label
                elif mode == "INSTRUMENTS":
                    lbl = next(k for k in type_labels if k in l_d)
                    image[np.all(ovr > 0, axis=-1)] = type_labels[lbl]
                elif mode == "PARTS":
                    for label_code in part_labels:
                        image[np.all(ovr == label_code, axis=-1)] = part_labels[label_code]
        else:
            ovr = cv2.imread(join(PREDICTIONS_DIR, model.lower(), mode.lower(), DATASET, FRAME))
            image = np.zeros_like(ovr)
            if mode == "BINARY":
                image[np.all(ovr > THRESHOLD, axis=-1)] = binary_label
            elif mode == "INSTRUMENTS":
                for label_code in type_labels_preds:
                    image[np.all(ovr == label_code, axis=-1)] = type_labels_preds[label_code]
            elif mode == "PARTS":
                for label_code in part_labels_preds:
                    image[np.all(ovr == label_code, axis=-1)] = part_labels_preds[label_code]
    image = image[TOP: BOTTOM, LEFT: RIGHT]
    return cv2.resize(image, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)


DATASET_NUM = 1
FRAME_NUM = 1
for DATASET_NUM in range(1, 9):
    for FRAME_NUM in range(1, 226, 20):
        FRAME = "frame{:03d}.png".format(FRAME_NUM)
        DATASET = "instrument_dataset_" + str(DATASET_NUM)
        PHOTO_DIR = join("data/train", DATASET, "left_frames")
        GROUND_TRUTH_DIR = join("data/train", DATASET, "ground_truth")
        LABEL_DIRS = [l_d for l_d in listdir(GROUND_TRUTH_DIR) if isdir(join(GROUND_TRUTH_DIR, l_d))]
        PREDICTIONS_DIR = "predictions"

        grid = np.ones((H * N_ROWS + GAP * (N_ROWS - 1), W * N_COLUMNS + GAP * (N_COLUMNS - 1), 3)) * 255

        for j, model in enumerate(MODELS):
            for i, mode in enumerate(MODES):
                x = j * (W + GAP)
                y = i * (H + GAP)
                grid[y: y + H, x: x + W] = get_picture(mode, model)

        cv2.imwrite("images/grid-{}-{}.png".format(DATASET_NUM, FRAME_NUM), grid)


