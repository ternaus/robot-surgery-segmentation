import cv2
from os.path import join, isdir, exists
from os import mkdir
import numpy as np
from os import listdir


ROOT = "data/train/instrument_dataset_2"
PHOTO_DIR = join(ROOT, "left_frames")
GROUND_TRUTH_DIR = join(ROOT, "ground_truth")
LABEL_DIRS = [l_d for l_d in listdir(GROUND_TRUTH_DIR) if isdir(join(GROUND_TRUTH_DIR, l_d))]

ALPHA = 0.5
SCALE = 0.25
MODES = ["ORIGINAL", "PARTS", "TYPE", "BINARY"]

RED, GREEN, BLUE, YELLOW, PURPLE, CYAN, LIME = (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255,255), \
                                                (128, 0, 128), (255, 255, 0), (0, 255, 0)
type_labels = {
    "Bipolar": RED,
    "Prograsp": GREEN,
    "Needle": BLUE,
    "Sealer": YELLOW,
    "Grasp": PURPLE,
    "Scissors": LIME,
    "Other": CYAN,
}
part_labels = {
    10: BLUE,
    20: YELLOW,
    30: RED,
    40: GREEN,
}
binary_label = GREEN
LEFT, TOP = 328, 37
RIGHT, BOTTOM = 1591, 1046

for MODE in MODES:
    OUTPUT_DIR = join(ROOT, "output_" + MODE.lower())
    if not exists(OUTPUT_DIR): mkdir(OUTPUT_DIR)
    for FRAME_NUM in range(225):
        FRAME = "frame{:03d}.png".format(FRAME_NUM)

        image = cv2.imread(join(PHOTO_DIR, FRAME))
        overlay = np.zeros_like(image)

        if MODE != "ORIGINAL":
            for l_d, (_, v) in zip(LABEL_DIRS, type_labels.items()):
                ovr = cv2.imread(join(GROUND_TRUTH_DIR, l_d, FRAME))
                if MODE == "BINARY":
                    overlay[np.all(ovr > 0, axis=-1)] = binary_label
                elif MODE == "TYPE":
                    overlay[np.all(ovr > 0, axis=-1)] = v
                elif MODE == "PARTS":
                    for label_code in part_labels:
                        overlay[np.all(ovr == label_code, axis=-1)] = part_labels[label_code]

        mask = np.ones(overlay.shape[:-1] + (1,)) * ALPHA
        mask[np.all(overlay == 0, axis=-1)] = 0
        out = image * (1 - mask) + overlay * mask
        out = out.astype(np.uint8)
        out = out[TOP: BOTTOM + 1, LEFT: RIGHT + 1]
        if SCALE != 1:
            out = cv2.resize(out, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(join(OUTPUT_DIR, FRAME), out)

        # cv2.putText(overlay, "PyImageSearch: alpha={}".format(alpha),
        #     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        # cv2.imshow("Output", out)
        # cv2.waitKey(0)
