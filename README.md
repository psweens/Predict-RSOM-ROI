# Predicting Tumour Region-of-Interests from 3D Mesoscopic Photoacoustic Images

This Python package contains a trained 3D convolutional neural network based on a [U-Net](https://arxiv.org/abs/1505.04597) architecture to segment tumour region-of-interest (ROI) from raster-scan optoacoustic mesoscopy (RSOM) 3D image volumes. Segmentation allows delineation of tumour ROIs from surrounding tissue to provide an estimate of the tumour boundary and consequently tumour volume.

Hyperparameters were optimised and evaluated using [Talos](https://github.com/autonomio/talos) integrated with [Keras](https://keras.io/). A random search optimisation strategy was deployed using the quantum random method. A probabilistic reduction scheme was used to reduce the number of parameter permutations by removing poorly performing hyperparameter configurations from the remaining search space after a predefined interval.

The scripts to train your own 3D CNN from scratch or perform additional training can be found [here](https://github.com/psweens/3D-CNN). More detailed information of the design, training and application can be found here.

## CNN Architecture
The network architecture consists of five convolutional layers with dropout in the 3rd, 4th and 5th layers to reduce segmentation bias and ensure that segmentation is performed utilising high-level features that may not have been considered in our semi-manual ROI annotations used as ground truth.

![alt text](https://github.com/psweens/Predict-RSOM-ROI/blob/main/CNN_Architecture.jpg)

## Training

![alt text](https://github.com/psweens/Predict-RSOM-ROI/blob/main/ROI_Analysis%20layout%202.jpg)

## Prerequisites
The 3D CNN was trained using:
* Python 3.6.
* Keras 2.3.1.
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
