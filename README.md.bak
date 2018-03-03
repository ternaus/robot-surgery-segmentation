# robots
[endovissub2017](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/)
The training dataset contatins 8 instrument datasets, while test datasets contains 10 instrument datasets. The whole project tree looks like:

```
├── data
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

To run the calculations as a first step you need to prepare data and run 
```
python prepare_data.py
```
it removed unnessery part of images and make appropriate image resolutions.

![Alt Text](https://github.com/ternaus/robots/blob/master/images/gifs/dataset6/original.gif) ![Alt Text](https://github.com/ternaus/robots/blob/master/images/gifs/dataset6/binary.gif)

![Alt Text](https://github.com/ternaus/robots/blob/master/images/gifs/dataset6/parts.gif) ![Alt Text](https://github.com/ternaus/robots/blob/master/images/gifs/dataset6/types.gif)

### Binary Segmentation

| Model            |Mean IOU   | Mean Dice          | Inference time (1280x1024)| Model Download Link |
|------------------|-----------|--------------------|---------------------------|---------------------|
| U-Net            | 75.44     | 84.37              | 93 ms.                    | [Dropbox](https://drive.google.com/)|
| Ternaus-Net11    | 81.14     | 88.07              | 142 ms.                   | [google drive](https://drive.google.com/drive/folders/1PfQ-0QDURIvf6WpvllC_3sm0JInMRB4O)            |
| Ternaus-Net16    | 83.60     | 90.01              | 184 ms.                   |             |
| Link-Net         | 82.36     | 88.87              | 88 ms.                    | [google drive](https://drive.google.com/drive/folders/12OXFy82Z_x1Y1Ly1EKa43r6Jd468m6SE)      |


### multi-class Segmentation (3 parts of an instrument)

| Model            |Mean IOU   | Mean Dice          | Inference time (1280x1024)| Model Download Link |
|------------------|-----------|--------------------|---------------------------|---------------------|
| U-Net            | 48.41     | 60.75              | 106 ms.                   | [Dropbox](https://www.dropbox.com/)|
| Ternaus-Net11    | 63.23     | 74.25              | 157 ms.                   | in prog.            |
| Ternaus-Net16    | 65.50     | 75.97              | 202 ms.                   | in prog.            |
| Link-Net         | 34.55     | 41.26              | 97 ms.                    | in prog.            |

### multi-class Segmentation (7 different instruments)

| Model            |Mean IOU   | Mean Dice          | Inference time (1280x1024)| Model Download Link |
|------------------|-----------|--------------------|---------------------------|---------------------|
| U-Net            | 15.80     | 23.59              | 122 ms.                   | [Dropbox](https://www.dropbox.com/)|
| Ternaus-Net11    | 34.61     | 45.86              | 173 ms.                   | in prog.            |
| Ternaus-Net16    | 33.78     | 44.95              | 275 ms.                   | in prog.            |
| Link-Net         | 22.47     | 24.71              | 177 ms.                   | in prog.            |
