# Real Time Object Detection and Counting
This is an explantation of the program test.py 
## STEP1 : Load the YOLO network

1- First we download the pre-trained YOLO weight file (237 MB): 
```bash
https://pjreddie.com/media/files/yolov3.weight
```
2- then the YOLO configuration file named yolov3.cfg
3- And the 80 COCO class names.

4- You can now load the YOLO network model from the harddisk into OpenCV:
```bash
net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg")
```
The YOLO neural network has 254 components, 
```bash
ln = net.getLayerNames()
```
Extract the 82, 94 , 106 layers , as they are what characterize Yolo3 ALgorithm
```bash
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
```
5-The input to the network is a called blob object , this function transforms the image into a blob
```bash
blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
```
the image to transform
the scale factor (1/255 to scale the pixel values to [0..1])    
the size, here a 416x416 square image  
the mean value (default=0)  
the option swapBR=True (since OpenCV uses BGR)

A blob is a 4D numpy array object (images, channels, width, height).
## STEP2 : Identifiy objects

1- These two instructions calculate the network response:
```bash
net.setInput(blob)
```                                                                     
The blob object is given as input to the network:
```bash
  net.setInput(blob)
```
2- 
![image](121.png)

```bash
outs = net.forward(outputlayers)
```
The outs object are vectors of lenght 85                                               
4x the bounding box (centerx, centery, width, height)    
                                 1x box confidence                            
80x class confidence


```bash
scores = detection[5:] # scores takes classes scores
class_id = np.argmax(scores) # We extract the index of the highest score which correspends to our object
confidence = scores[class_id] # We extract our object score
```
if confidence > 0.3 ==> object detected , we add to a list called Boxes the cordinates(Center_X,Centrer_Y,Weight,Height) of all the detected objects.
another list called confidences conains all scores.
```bash
boxes.append([x,y,w,h]) #put all rectangle areas
                confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
                class_ids.append(class_id) #name of the object tha was detected
```
the list indexes contains the index of each object detected , the legnth of indexes is the number of total objects detected
```bash
indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)
```

## STEP3 : Counting objects
For counting objets , we created a new vector without repteated  values .
we use this new vector to count how much an object appears in one frame

```bash
pap=1
for you in m:
count=np.count_nonzero(arr == you) # counts how much an object appears in one frame
cv2.putText(frame,str(classes[you])+" "+str(count),(10,150+pap*30),font,2,(0,0,0),1) # object name , and how many is there
pap=pap+1
```


