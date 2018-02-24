# robots
[endovissub2017](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/)
The training dataset contatins 8 instrument datasets, while test datasets contains 10 instrument datasets. The whole project tree looks like:

```
├── input
│   ├── test
│   │   ├── instrument_dataset_1
│   │   │   ├── left_frames
│   │   │   └── right_frames
│   └── train
│       ├── instrument_dataset_1
│       │   ├── ground_truth
│       │   │   ├── Left_Prograsp_Forceps_labels
│       │   │   ├── Maryland_Bipolar_Forceps_labels
│       │   │   ├── Other_labels
│       │   │   └── Right_Prograsp_Forceps_labels
│       │   ├── left_frames
│       │   └── right_frames
├── notebooks
├── src
└── predictions
```
the notebooks folder contains Jupyter notebooks that helps us to unilize visually the train and test images. The predictions folder contains weights of several networks

![Alt Text](https://github.com/ternaus/robots/blob/master/images/gifs/dataset4/original.gif) ![Alt Text](https://github.com/ternaus/robots/blob/master/images/gifs/dataset4/binary.gif)

![Alt Text](https://github.com/ternaus/robots/blob/master/images/gifs/dataset4/parts.gif) ![Alt Text](https://github.com/ternaus/robots/blob/master/images/gifs/dataset4/type.gif)

### Binary Segmentation

| Model            |Mean IOU   | Mean pix. accuracy | Inference time (512x512) | Model Download Link |
|------------------|-----------|--------------------|--------------------------|---------------------|
| U-Net            | 96.0      | in prog.           | 28 ms.                   | [Dropbox](https://drive.google.com/)|
| Ternaus-Net11    | in prog.  | in prog.           | 50 ms.                   | [google drive](https://drive.google.com/drive/folders/1dFzqsFosU04oyAb_5DaieDsdKYXrgXfS)            |
| Ternaus-Net16    | in prog.  | in prog.           | 50 ms.                   | [google drive](https://drive.google.com/drive/folders/1M9SeI7T49HjdmCFVbeCM9mEt4edwyNFQ?usp=sharing)            |
| Link-Net         | in prog.  | in prog.           | 50 ms.                   | in prog.            |
| PSP-Net          | in prog.  | in prog.           | 50 ms.                   | in prog.            |


### 3-class Segmentation (part of instruments)

| Model            |Mean IOU   | Mean pix. accuracy | Inference time (512x512) | Model Download Link |
|------------------|-----------|--------------------|--------------------------|---------------------|
| U-Net            | 96.0      | in prog.           | 28 ms.                   | [Dropbox](https://www.dropbox.com/)|
| Ternaus-Net11    | in prog.  | in prog.           | 50 ms.                   | in prog.            |
| Ternaus-Net16    | in prog.  | in prog.           | 50 ms.                   | in prog.            |
| Link-Net         | in prog.  | in prog.           | 50 ms.                   | in prog.            |
| PSP-Net          | in prog.  | in prog.           | 50 ms.                   | in prog.            |

### 3-class Segmentation (different instruments)

| Model            |Mean IOU   | Mean pix. accuracy | Inference time (512x512) | Model Download Link |
|------------------|-----------|--------------------|--------------------------|---------------------|
| U-Net            | 96.0      | in prog.           | 28 ms.                   | [Dropbox](https://www.dropbox.com/)|
| Ternaus-Net11    | in prog.  | in prog.           | 50 ms.                   | in prog.            |
| Ternaus-Net16    | in prog.  | in prog.           | 50 ms.                   | in prog.            |
| Link-Net         | in prog.  | in prog.           | 50 ms.                   | in prog.            |
| PSP-Net          | in prog.  | in prog.           | 50 ms.                   | in prog.            |
