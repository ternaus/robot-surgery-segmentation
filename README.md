# robots


[endovissub2017](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/)

![Alt Text](https://github.com/ternaus/robots/blob/master/images/gifs/dataset4/binary.gif)

![Alt Text](https://github.com/ternaus/robots/blob/master/images/gifs/dataset4/parts.gif)

![Alt Text](https://github.com/ternaus/robots/blob/master/images/gifs/dataset4/type.gif)

### Binary Segmentation

| Model            | Test data |Mean IOU | Mean pix. accuracy | Pixel accuracy|Inference time (512x512 px. image) | Model Download Link |
|------------------|-----------|---------|--------------------|----------------|----|---------------------|
| U-Net   | RV-VOC12  | 96.0   | in prog.           | in prog.       |28 ms.| [Dropbox](https://www.dropbox.com/)            |
| Ternaus-Net   | RV-VOC12  | in prog.   | in prog.           | in prog.  | 50 ms.  | in prog.            |
| Link-Net   | RV-VOC12  | in prog.   | in prog.           | in prog.  | 50 ms.  | in prog.            |
| PSP-Net   | RV-VOC12  | in prog.   | in prog.           | in prog.  | 50 ms.  | in prog.            |


Qualitative results (on validation sequence):

![Alt text](pytorch_segmentation_detection/recipes/endovis_2017/segmentation/validation_binary.gif?raw=true "Title")

### Multi-class Segmentation

| Model            | Test data |Mean IOU | Mean pix. accuracy | Pixel accuracy|Inference time (512x512 px. image) | Model Download Link |
|------------------|-----------|---------|--------------------|----------------|----|---------------------|
| Resnet-18-8s   | RV-VOC12  | 81.0   | in prog.           | in prog.       |28 ms.| [Dropbox](https://www.dropbox.com/s/p9ey655mmzb3v5l/resnet_18_8s_multiclass_best.pth?dl=0)            |
| Resnet-34-8s   | RV-VOC12  | in prog.   | in prog.           | in prog.  | 50 ms.  | in prog            |
