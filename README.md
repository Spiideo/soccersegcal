# Soccer Pitch Segmentation and Camera Calibration
Official implementation of Spiideo's contribution to the 2023
[SoccerNet Camera Calibration challange](https://www.soccer-net.org/tasks/camera-calibration).

![Example segmentatoions](docs/segmentations.jpg)

It includes a modified version of the
[SoccerNet Camera Calibration Development Kit](https://github.com/SoccerNet/sn-calibration)
in [sncalib](https://github.com/Spiideo/soccersegcal/tree/main/sncalib).

## Install
Install requirements using

    pip install -r requirements.txt

## Train
To train the segmentation model, use

    python train.py

## Estimate Cameras
To use the trained segmentation model to estimate camera parameters, use

    python estimate_cameras.py