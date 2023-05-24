# Soccer Pitch Segmentation and Camera Calibration
Official implementation of Spiideo's contribution to the 2023
[SoccerNet Camera Calibration challange](https://www.soccer-net.org/tasks/camera-calibration).

![Example segmentatoions](docs/segmentations.jpg)

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
Download the SoccerNet into `data/SoccerNet/calibration-2023`.

```python
from SoccerNet.Downloader import SoccerNetDownloader as SNdl
soccerNetDownloader = SNdl(LocalDirectory="data/SoccerNet/calibration-2023")
soccerNetDownloader.downloadDataTask(task="calibration-2023", split=["train", "valid", "test", "challenge"])
```

Run the dataloader to display the images and generated segmentations:

    python soccersegcal/dataloader.py

## Train
To train the segmentation model, use

    python soccersegcal/train.py

## Estimate Cameras
To use the trained segmentation model to estimate camera parameters, use

    python soccersegcal/estimate_cameras.py