# Dual Refinement Underwater Object Detection Network

[This is the unofficial implementation of Dual Refinement Underwater Object Detection Network](https://link.springer.com/chapter/10.1007/978-3-030-58565-5_17) (ECCV2020).

### SETUP

This implementation is based on [mmdetection](https://github.com/open-mmlab/mmdetection).

Follow the installation of [mmdetection](https://github.com/open-mmlab/mmdetection), and copy the code of this repository to the [mmdetection](https://github.com/open-mmlab/mmdetection) file.

### ENVIRONMENT

Python == 3.7.6

Pytorch == 1.5.1

torchvision == 0.6.1

numpy == 1.18.1

pillow == 7.0.0

### DATASET

Since the authors of FERNet haven't open source the UWD dataset, we use dataset UTDAC2020, the download link of which is shown as follows.

It is recommended to symlink the dataset file to the root.

```
FERNet
├── data
│   ├── UTDAC2020
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── annotations
```

### TRAIN

```python
python tools/train.py configs/FERNet300_coco.py
```

### TEST

```
python tools/test.py configs/FERNet300_coco.py path/to/checkpoints
```

### DETAILS OF THE IMPLEMENTATION

Since the official paper omits lots of details of the implementation, I decide these details to reimplement the code.

- In CCB, the paper selects the $150 \times 150$, $75 \times 75$ and  $38 \times 38$ characteristic layers on the lead backbone, but haven't clear point out which layer for feature fusion (for features of many layer have the same resolutions), we use the features when the channel sizes just reach 64, 256 and 512 respectively in VGG16 for feature fusion. For ResNet-50, the feature after the first conv (before the first maxpooling), features of Stage1 and Stage2 for feature fusion.
- It seems that the original SSD do the downsampling in the extra layers. If RFAM also do the downsampling, the prediction feature maps' sizes won't fit any more. So I cancel the stride in the RFAM block.
- In PRS, the paper mentions to do the dilation in DCN, but does not mention the dilation rate. I set it as 3.
- The paper says that pre-processing phase is doing binary classification and refinement phase is fine-tuned. I maintain the pre-processing phase as multi-classification, and sum the softmax logit except background.
- In PRS, for each anchors of certain pixel has its own offset ($\Delta x, \Delta y$), we use group DCN and set  $group == deform\_group == num_anchors$. 
- Instead of directly input offset ($\Delta x, \Delta y$) to DCN, I use a FC layer to process the offset, which seems to obtain higher performance.
- Since mmdetection have no implementation of randomly warming up, I still follow the pre-defined schedule_2x setting in mmdetection.

### SOMETHING TO SAY ABOUT THE EXPERIMENTS

- Although I only train the model for 24 epochs, it seems that neither of PRS and CCB improves performance. Thus, I need your help to point out the mistake of my implementation and assist me to improve the this implementation. If the author of the paper can see this repository, please open source the code as soon as possible. 
- In PRS, if you use the feature of after the pre-processing phase ($X_{end}$) for DCN, the performance is much lower than just using $X_{out}$. 
- You can realize fine-tune refinement phase by inputting feat.detach() and offset.detach() to DCN. In this way, DCN won't affect the training of upstream parameters.
- In RFAM, the first branch and the input do not have any spatial downsampling while the other two branches do downsampling, which will lead to mismatch of the size. Besides, the authors do not mention the padding method for the dilation, so I use zero padding.
