# Cascaded-approach-using-SE-ResNeXt-ResNet-and-Feature-Pyramid-Network-for-kidney-tumor-segmentation
This project employs a cascaded approach utilizing SE_ResNeXt, ResNet, and Feature Pyramid Network for kidney tumor segmentation. The process involves two related stages aimed at achieving accurate segmentation results.

## Project Overview

### Dependencies Installation:
run pip install -r requirements.txt to install all the dependencies

### Data Preprocessing:
The KiTS-19 dataset was used for this training -> https://github.com/neheller/kits19
Extract slices from each scan and rename them using the rename_dataset.py script.
Use the load_images function to preprocess the data while loading the slices.
Call change_masks_background to convert kidney tumors to the foreground and other kidney parts to the background.

### Dataset Preparation:
Run group_dataset.py to rename and group all images into one folder and all tumors into another folder. This restructuring is suitable for the PyTorch DataLoader.

### First Training:
Execute load_dataset.py to load the dataset for the initial training.
Perform the first training stage using the coarse_segmentation script.

### Second Training with FPN and ResNet:
Execute load_dataset.py to load the dataset for the final training with the right path.
Run fpn_resnet.py for the second training stage, utilizing Feature Pyramid Network (FPN) and ResNet using the results of the first training.

### Metrics Measurement:
Measure and evaluate metrics for the second training using the appropriate evaluation script.
