# Course-Project-reproduce-Results-

**DPDNet: Defocus Deblurring using Dual-Pixel Sensors:**
This repository contains the implementation of the DPDNet model for defocus deblurring using dual-pixel sensors. The implementation includes both the training and testing code.

**Overview**
DPDNet is a deep learning model that leverages dual-pixel sensor data to deblur images affected by defocus blur. The model is trained end-to-end to estimate a sharp image directly from the left/right DP views of the defocused input image.

**Dataset**
The dataset used for training and evaluation consists of 2000 images, which include:

500 defocus-blurred images
1000 sub-aperture views from the dual-pixel sensor
500 corresponding all-in-focus images
All the images in the dataset have a full-frame resolution of 6720 x 4480 pixels. The dataset is hosted on OneDrive and can be accessed using the following link:

Please download and unzip the dataset before proceeding with the training and testing.

**Requirements**
Python 3.6+
Tensorflow==1.14.0
Keras
NumPy 1.17.2
OpenCV 3.4.2
Scikit-image 0.16.2
Scikit-learn


**Training**
To train the DPDNet model, use the training_code_DPDNet folder. Before training, ensure that the dataset is downloaded and placed in the appropriate directory.


**Testing**
To test the DPDNet model, use the testing_code_DPDNet folder. Before testing, ensure that the dataset is downloaded and placed in the appropriate directory, and that a trained model checkpoint is available.
