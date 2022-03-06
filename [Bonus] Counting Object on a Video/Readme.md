# YOLOv3 + Deep_SORT

<img src="https://github.com/yehengchen/video_demo/blob/master/video_demo/output.gif" width="40%" height="40%"> <img src="https://github.com/yehengchen/video_demo/blob/master/video_demo/TownCentreXVID_output.gif" width="40%" height="40%">
<img src="https://github.com/yehengchen/Object-Detection-and-Tracking/blob/master/OneStage/yolo/yolo_img/output_person_315_1120_s.gif" width="40%" height="40%"> <img src="https://github.com/yehengchen/video_demo/blob/master/video_demo/output_car_143.gif" width="40%" height="40%">

__Object Tracking & Counting Demo 


## Note
This work is based on the git of Mr.yehengchen , i just fixed some problemes related to version compatibilities , and edited some line of codes to make it run with the most recent versions of the software i have on my laptop.

## Requirement
__Development Environment: 

* OpenCV 4.2.0
* sklean 
* pillow 7.0.0
* numpy 1.22.2
* tensorflow-gpu  
***

It uses:

* __Detection__: [YOLOv3](https://github.com/yehengchen/ObjectDetection/tree/master/OneStage/yolo/yolov3) to detect objects on each of the video frames. 

* __Tracking__: [Deep_SORT](https://github.com/nwojke/deep_sort) to track those objects over different frames.

*This repository contains code for Simple Online and Realtime Tracking with a Deep Association Metric (Deep SORT). We extend the original SORT algorithm to integrate appearance information based on a deep appearance descriptor. See the [arXiv preprint](https://arxiv.org/abs/1703.07402) for more information.*

## Quick Start

__0.Requirements__

    pip install -r req.txt
    
__1. Download the code to your computer.__
    
    git clone https://github.com/yehengchen/Object-Detection-and-Tracking.git
    
__2. Download [[yolov3.weights]](https://pjreddie.com/media/files/yolov3.weights)__ and place it in `deep_sort_yolov3/model_data/`

*Here you can download my trained [[yolo-spp.h5]](https://pan.baidu.com/s/1DoiifwXrss1QgSQBp2vv8w&shfl=shareset) - `t13k` weights for detecting person/car/bicycle,etc.*

__3. Convert the Darknet YOLO model to a Keras model:__
```
$ python convert.py model_data/yolov3.cfg model_data/yolov3.weights model_data/yolo.h5
``` 
__4. Run the YOLO_DEEP_SORT:__

```
$ python main.py -c [CLASS NAME] -i [INPUT VIDEO PATH]

$ python main.py -c person -i people.mp4
```

__5. Can change [deep_sort_yolov3/yolo.py] `__Line 100__` to your tracking object__

*DeepSORT pre-trained weights using people-ReID datasets only for person*
```
    if predicted_class != args["class"]:
               continue
    
    if predicted_class != 'person' and predicted_class != 'car':
               continue
```

    
## Reference
#### Github:deep_sort_yolov3@[yehengchen](https://github.com/yehengchen/Object-Detection-and-Tracking/tree/master/OneStage/yolo/deep_sort_yolov3)
#### Github:deep_sort@[Nicolai Wojke nwojke](https://github.com/nwojke/deep_sort)
#### Github:deep_sort_yolov3@[Qidian213 ](https://github.com/Qidian213/deep_sort_yolov3)


