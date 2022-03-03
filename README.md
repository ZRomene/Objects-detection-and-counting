# IOT project
## Objects Detection and counting 
This IOT project consist on detecting and counting objects in real time using the webcam.
For this matter we need to install some librairies on our Ubuntu 20.04  which are: 
- Python3
- OpenCV
- YOLOv3 

[!python3.png] [!openCV.png] [!yolo.png]

## Installation 
### Installation of openCV
OpenCV (Open Source Computer Vision Library) is an open-source computer vision library with bindings for C++, Python, and Java and supports all major operating systems. It can take advantage of multi-core processing and features GPU acceleration for real-time operation.

```sh
sudo apt update
sudo apt install libopencv-dev python3-opencv
```

### Installation of Darknet
You only look once (YOLO) is a real-time object detection system. On a Pascal Titan X it processes images at 30 FPS and has a mAP of 57.9% on COCO test-dev. There are several ways of using YOLO, the original way, is through Darknet.
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.Darknet prints out the objects it detected, its confidence, and how long it took to find them.
To download a pre-trained model: 
```sh
git clone https://github.com/pjreddie/darknet
cd darknet
make
```
You already have the config file for YOLO in the cfg/ subdirectory. You will have to download the pre-trained weight file by running the following command:

```sh
wget https://pjreddie.com/media/files/yolov3.weights
```
To make sure that everything is compiled correctly, try running this 
```sh
./darknet
```
You should get the output:
> usage: ./darknet <function>


## Liens utiles
   - https://pjreddie.com/darknet/install/#cuda 
   - https://linuxize.com/post/how-to-install-opencv-on-ubuntu-20-04/
  

