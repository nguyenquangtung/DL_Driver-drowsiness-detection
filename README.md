# Driver_drowsiness_system_CNN

This system is designed to detect driver drowsiness using Convolutional Neural Networks (CNN) in Python with the help of OpenCV. Its primary objective is to reduce the occurrence of accidents on the road by detecting if the driver is becoming drowsy and issuing an alarm as a warning.

The implementation involves utilizing Python, OpenCV, and Keras (with TensorFlow) to construct a system capable of extracting facial features from the driver's face. By detecting the status of the driver's eyes (open or closed), the system can identify if the driver is falling asleep. If the eyes remain closed for a continuous period of 3 seconds, an alarm is triggered to capture the driver's attention and prevent any potential accidents.

To achieve this, a CNN model is trained on a dataset that contains examples of both closed and open eyes. This trained model is then utilized by OpenCV to capture live video feed from the camera. Each frame from the video feed is passed through the CNN model for processing and classification, determining whether the eyes are open or closed in real-time.

More info in: https://youtu.be/X8ZFH_fZsIo?si=v64a-kkypdOt_J2y

## Setup

1- To set up the environment, pre-install all the required libraries by using this:
```
pip install -r requirements.txt
```
<br />
<br />1) OpenCV<br /> 2) Keras<br /> 3) Numpy<br /> 4) Pandas<br /> 5) OS<br />
2- Download the Dataset from the link given below and edit the address in the notebook accordingly.<br />
3- Optional: Run the Jupyter Notebook to train your model with other custom hyper-parameter 20.<br />
4- Run detect_drowsiness.py file for main application (change the path model to your model or use my trained model) in line 20.<br />

## The Dataset

The dataset that was used is a subnet of a dataset from(https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset)<br />
it has 4 folder which are <br />1) Closed_eyes - having 726 pictures<br /> 2) Open_eyes - having 726 pictures<br /> 3) Yawn - having 725 pictures<br /> 4) no_yawn - having 723 pictures<br />

## The Convolution Neural Network

![image](https://github.com/nguyenquangtung/DL_Driver-drowsiness-detection/assets/59195029/9b89ad67-dfca-4f23-8b4b-b5058ed5075d)


## Accuracy

We did 50 epochs, to get a good accuracy from the model i.e. 98% for training accuracy and 96% for validation accuracy.
![image](https://github.com/nguyenquangtung/DL_Driver-drowsiness-detection/assets/59195029/07efe230-00e5-478a-b96a-9f63d2e91ceb)


<!-- 
## The Output
1. Open Eyes<br />
   ![Open_eyes](https://user-images.githubusercontent.com/16632408/159187179-b557ab8e-fb8c-4408-850b-417893014f8c.png)
2. Close Eyes<br />
   Here we detect wheater the eyes are closed and count the number of frames for which the eyes were closed (which is 10 frame) greater then that the Alarm will ring and the WARNING sign is displayed.
   ![Closed_eyes](https://user-images.githubusercontent.com/16632408/159187305-68cbdee3-8325-4216-85e3-7dbb66a429fb.png) -->
