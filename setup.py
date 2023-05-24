from setuptools import setup

setup(
    name='soccersegcal',
    description='Soccer Field Segmentation and Camera Calibration',
    long_description='Soccer pitch segmentation and camera calibration in two steps. Step 1, pixelwise segmentation of an broacast image of a soccer game into six different clases defined by the line markings. Step 2, a differential-rendering optimizer that tries to estimate camera parameters from such segementations.Trained on SoccerNet.',
    version='1.0',
    packages=['soccersegcal', 'sncalib'],
    zip_safe=False,
    url='https://github.com/Spiideo/soccersegcal/',
    author='Hakan Ardo',
    author_email='hakan.ardo@spiideo.com',
)