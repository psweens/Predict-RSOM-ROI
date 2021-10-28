# Predicting Tumour Region-of-Interest from 3D Mesoscopic Photoacoustic Images

This Python package contains a trained 3D convolutional neural network based on a [U-Net](https://arxiv.org/abs/1505.04597) architecture to segment tumour region-of-interest (ROI) from raster-scan optoacoustic mesoscopy (RSOM) 3D image volumes. Segmentation allows delineation of tumour ROIs from surrounding tissue to provide an estimate of the tumour boundary and consequently tumour volume.

## Prerequisites
The 3D CNN was trained using:
* Python 3.6.
* Tensorflow-GPU 1.14.0.

A package list for a Python environment has been provided and can be installed using the method described below.

## Installation
The ROI package is compatible with Python3, and has been tested on Ubuntu 18.04 LTS. 
Other distributions of Linux, macOS, Windows should work as well.

To install the package from source, download zip file on GitHub page or run the following in a terminal:
```bash
git clone https://github.com/psweens/Predict-RSOM-ROI.git
```

The required Python packages can be found [here](https://github.com/psweens/Predict-RSOM-ROI/blob/main/REQUIREMENTS.txt). The package list can be installed, for example, using creating a Conda environment by running:
```bash
conda create --name <env> --file REQUIREMENTS.txt
```
