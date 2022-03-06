
# Check this link to understand
#https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html	

import cv2
import numpy as np
import time

net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg") # Original yolov3
classes = []

#Reading classes from (our object database) coco.names 

with open("coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]
	
#print(classes) displaying 80 classes 

layer_names = net.getLayerNames() #The YOLO neural network has 254 components or layers.


outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()] # Extract the 82, 94 , 106 layers , as they are what characterize Yolo3 ALgorithm 


colors= np.random.uniform(0,255,size=(len(classes),3)) #Choosing random color for each class

cap = cv2.VideoCapture(0) #Start video capture
font = cv2.FONT_HERSHEY_PLAIN
starting_time= time.time() #Start counting time
frame_id = 0

while True:
    _,frame= cap.read() #Collect frame by frame from the video
    frame_id+=1
    
    #height,width,channels = frame.shape
    frame = cv2.resize(frame,(800,600),fx=0,fy=0, interpolation = cv2.INTER_CUBIC) # resizing the frame
    height,width,channels = frame.shape

    #detecting objects
    #A blob is a 4D numpy array object (images, channels, width, height).
    
    blob = cv2.dnn.blobFromImage(frame,0.00392,(320,320),(0,0,0),swapRB=True,crop=False) #reduce 416 to 320    
    
    #The blob object is given as input to the network:
    #These two instructions calculate the network response:
    
    net.setInput(blob)
    
    #The size of outs is 3
    #each outputs object are vectors of lenght 85
    #4x the bounding box (centerx, centery, width, height)
    #1x box confidence
    #80x class confidence
    
    outs = net.forward(outputlayers)
    
    
    #xx=np.array(outs)
   
    

    #Showing info on screen/ get confidence score of algorithm in detecting an object in blob
    class_ids=[]
    confidences=[]
    boxes=[]
    
    #https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b
    
    for out in outs: # We have three layers to work with , each object will be detect using the three layers
        for detection in out: # We take each layer , and try to detect our object 
            scores = detection[5:] # scores takes classes scores
            class_id = np.argmax(scores) # We extract the index of the highest score which correspends to our object
            confidence = scores[class_id] # We extract our object score
            if confidence > 0.3:
                #object detected
                # for each object detect you have (centerx, centery, width, height)
                #print(type(detection[0]))
                center_x= int(detection[0]*width) #detection[0] is  centerx 
                center_y= int(detection[1]*height) #detection[1] is centery
                w = int(detection[2]*width) #detection[2] is  width 
                h = int(detection[3]*height) ##detection[3] is  height

                #cv2.circle(frame,(center_x,center_y),10,(0,255,0),2) # to draw the center of the box
                #rectangle co-ordinaters
                x=int(center_x - w/2)
                y=int(center_y - h/2)
                #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2) #nothing

                boxes.append([x,y,w,h]) #put all rectangle areas
                confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
                class_ids.append(class_id) #name of the object tha was detected
                
                
                
    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6) #   To delete ovearlaping boxes ,Perform non maximum suppression to eliminate redundant overlapping boxes with
    
    
    
    arr=np.array((class_ids))  # convert to array , arrays are easy to manipulate
    m = list(dict.fromkeys(class_ids)) 
    m=np.array(m) # m is an array containing all class_ids ( without duplicated ones)
   

    for i in range(len(boxes)): # for i=0 i<= nombre boxes i++
        if i in indexes:
            x,y,w,h = boxes[i] # get coordinates to draw the boxe for each object
            label = str(classes[class_ids[i]]) # get object's name
            confidence= confidences[i] # get its accuracy
            color = colors[class_ids[i]] # get the box color
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2) # draw the box
            cv2.putText(frame,label+" "+str(round(confidence,2)),(x,y+30),font,1,(255,255,255),2) #write the label
            cv2.putText(frame,"Total Object"+" "+str(len(class_ids)),(10,150),font,2,(0,0,0),1) # display Total object number
            	
            pap=1
            for you in m:
             count=np.count_nonzero(arr == you) # counts how much an object appears in one frame
             #print('count for ',classes[you],'is',str(count))
             cv2.putText(frame,str(classes[you])+" "+str(count),(10,150+pap*30),font,2,(0,0,0),1) # object name , and how many is there
             pap=pap+1
             
            
 
               
    elapsed_time = time.time() - starting_time
    fps=frame_id/elapsed_time
    cv2.putText(frame,"FPS:"+str(round(fps,2)),(10,50),font,2,(0,0,0),1)
    
    cv2.imshow("Image",frame)
    key = cv2.waitKey(1) #wait 1ms the loop will start again and we will process the next frame
    
    if key == 27: #esc key stops the process
        break;


cap.release()    
cv2.destroyAllWindows()    

