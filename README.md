# Soccer Field Segmentation and Camera Calibration
Official implementation of Spiideo's contribution to the 2023
[SoccerNet Camera Calibration challange](https://www.soccer-net.org/tasks/camera-calibration).

![Example segmentatoions](docs/segmentations.jpg)

## Install
Install requirements using

    pip install -r requirements.txt

## Train
To train the segmentation model, use

    python train.py

## Estimate Cameras
To use the trained segmentation model to estimate camera parameters, use

    python estimate_cameras.py