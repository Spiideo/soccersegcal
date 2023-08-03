# Soccer Pitch Segmentation and Camera Calibration
Official implementation of Spiideo's contribution to the 2023
[SoccerNet Camera Calibration challange](https://www.soccer-net.org/tasks/camera-calibration).

![Example segmentatoions](docs/segmentations.jpg)
![Example optimization](docs/demo.gif)


It includes a modified version of the
[SoccerNet Camera Calibration Development Kit](https://github.com/SoccerNet/sn-calibration)
in [sncalib](https://github.com/Spiideo/soccersegcal/tree/main/sncalib).

## Install
Install pytorch3d following it's
[installation instaructions](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md), for example

    python -mpip install --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1120/download.html

Install remaining requirements and setup python to run the modules from the checked
out source

    python -mpip install -r requirements.txt
    python setup.py develop --user

## Data
Download the SoccerNet into `data/SoccerNet/calibration-2023`

```python
from SoccerNet.Downloader import SoccerNetDownloader as SNdl
soccerNetDownloader = SNdl(LocalDirectory="data/SoccerNet/calibration-2023")
soccerNetDownloader.downloadDataTask(task="calibration-2023", split=["train", "valid", "test", "challenge"])
```

Run the dataloader to display the images and generated segmentations:

    python soccersegcal/dataloader.py

## Train Segmentation
Train the segmentation model (add `--help` to se availible options):

    python soccersegcal/train.py

To monitor training progress, compare different runs and get hold of the resulting checkpoint.ckpt:

    mlflow ui

The checkpoint can also be found by digging through the `mlruns` dir.

## Estimate Cameras
To use the trained segmentation model to estimate camera parameters for the first two samples (index 0 and 1) in the validation set while visualizing the optimization:

    python soccersegcal/estimate_cameras.py -c path/to/segmentation/checkpoint.ckpt -i [0,1] -s

To estimate all the cameras in the test set without visualisation (faster):

    python soccersegcal/estimate_cameras.py -c path/to/segmentation/checkpoint.ckpt -p test

To se other options:

    python soccersegcal/estimate_cameras.py --help

The estimated cameras will be saved in the `cams_out` directory. To run the SoccerNet evaluation on them:

    python sncalib/evaluate_camera.py -s data/SoccerNet/calibration-2023/ --split test -p cams_out/

## Pretrained weights
Pretrained weights can be downloaded from the table below. It also lists hyperparameters with non-default values.

| Hyperparameters | Combined Metric | Accuracy@5 | Completeness | |
| --- | --- | --- | --- | --- |
| epochs=27 | 0.53 | 52.95 | 99.96 | [snapshot.ckpt](https://github.com/Spiideo/soccersegcal/releases/download/SoccerNetChallenge2023/snapshot.ckpt) |
