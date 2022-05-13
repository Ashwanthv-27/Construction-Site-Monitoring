import numpy as np
import argparse
import imutils
import time
import cv2

import os
import PIL
from tkinter import *
from timeit import default_timer as timer
import math 
from custom import detect
# import efficientnet.tfkeras
# from tensorflow.keras.models import load_model

from tensorflow.keras.models import load_model
from tensorflow import nn
from tensorflow.keras.backend import shape
from tensorflow.keras.layers import Dropout

class FixedDropout(Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        return tuple([shape(inputs)[i] if sh is None else sh for i, sh in enumerate(self.noise_shape)])

customObjects = {
    'swish': nn.swish,
    'FixedDropout': FixedDropout
}

from tensorflow.keras.models import model_from_json

with open('model1.json', "r") as json_file:
    model_json1 = json_file.read()
    model1 = model_from_json(model_json1,custom_objects=customObjects)

model1.load_weights('model1.h5')

with open('model2.json', "r") as json_file:
    model_json1 = json_file.read()
    model2 = model_from_json(model_json1,custom_objects=customObjects)

model2.load_weights('model2.h5')

with open('model3.json', "r") as json_file:
    model_json1 = json_file.read()
    model3 = model_from_json(model_json1,custom_objects=customObjects)

model3.load_weights('model3.h5')




font = cv2.FONT_HERSHEY_SIMPLEX
green = (0, 255, 0)
red = (0, 0, 255)
line_type = cv2.LINE_AA
IMAGE_SIZE = 224

#model1=load_model('model1')
# model2=load_model('model2')
# model3=load_model('model3')

dt=250
def distance(x1 , y1 ,w1,h1, x2 , y2,w2,h2): 
    x3=(x1+w1)/2
    y3=(y1+h1)/2
    x4=(x2+w2)/2
    y4=(y2+h2)/2



  
    return math.sqrt(math.pow(x4 - x3, 2) +
                math.pow(y4 - y3, 2) * 1.0) 

def near(x1 , y1 ,w1,h1,coord):
    for i in coord:
        x2,y2,w2,h2=i
        d=distance(x1 , y1 ,w1,h1, x2 , y2,w2,h2)
        if d<dt:
            return True
    return False        

from train1 import load_image1 as load_image1
from train2 import load_image1 as load_image2
from train3 import load_image1 as load_image3
def top_pred(img):
    image=load_image1(img)
    image=np.expand_dims(image,axis=0)/255
    pred=model1.predict(image)[0]
    # if pred>0.5:
    #     return 1
    # else:
    #     return 0   
    # return abs(np.argmax(pred)-1)
    return abs(np.argmax(pred)-1)

    

def middle_pred(img):
    image=load_image2(img)
    image=np.expand_dims(image,axis=0)/255
    
    pred=model1.predict(image)[0]
    # if pred>0.5:
    #     return 1
    # else:
    #     return 0   
    return abs(np.argmax(pred)-1)

def bot_pred(img):
    image=load_image3(img)
    image=np.expand_dims(image,axis=0)/255
    pred=model1.predict(image)[0]
    # if pred>0.5:
    #     return 1
    # else:
    #     return 0   
    return abs(np.argmax(pred)-1)

def work(frame):
    #frame=cv2.imread(path)
    frame=cv2.resize(frame,(1200,600))  
    eq_coord=detect(frame.copy())
    for i in eq_coord:
        x2,y2,w2,h2=i
        cv2.rectangle(frame, (x2, y2), (x2+ w2, y2 +h2), (255,0,0), 2)
        cv2.putText(frame, 'Equipment', (x2+10, y2-15),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)


    labelsPath = os.path.sep.join(["allmodel", "labels.names"])
    LABELS = open(labelsPath).read().strip().split("\n")
    detections=["person"]
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
            dtype="uint8")

    weightsPath = os.path.sep.join(["allmodel", "yolov4.weights"])
    configPath = os.path.sep.join(["allmodel", "yolov4.cfg"])

    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    frame1=frame.copy()
    

    
    (H, W) = frame1.shape[:2]
    
    blob = cv2.dnn.blobFromImage(frame1, 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    boxes = []
    confidences = []
    classIDs = []
    # frame1=cv2.resize(frame,(500,500))
    
    

    for output in layerOutputs:
            for detection in output:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
                    
                    if confidence > 0.2:
                            
                            box = detection[0:4] * np.array([W, H, W, H])
                            
                            (centerX, centerY, width, height) = box.astype("int")

                            x = int(centerX - (width / 2))
                            y = int(centerY - (height / 2))

                            boxes.append([x, y, int(width), int(height)])
                            confidences.append(float(confidence))
                            classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences,0.2,0.2)
    flag=0
    if len(idxs) > 0:
            for i in idxs.flatten():
                if LABELS[classIDs[i]] in detections:
                        color = [int(c) for c in COLORS[classIDs[i]]]
                        if LABELS[classIDs[i]] in ['person']:
                            try:
                                (x, y) = (boxes[i][0], boxes[i][1])
                                (w, h) = (boxes[i][2], boxes[i][3])
                                danger=near(x,y,w,h,eq_coord)
                                crop=frame[y:y+h,x:x+w]
                                h1,w1=crop.shape[:2]
                                ar=w1/h1
                                print(ar,",======")
                                if ar>0.6:
                                    pass
                                else:
                                    t1=h1//4
                                    top=crop[:t1,:]
                                    #top=cv2.cvtColor(top, cv2.COLOR_BGR2RGB)
                                    cv2.imwrite('top.png',top)
                                    top_safety=top_pred(top)
                                    middle=crop[t1:3*t1,:]
                                    cv2.imwrite('middle.png',middle)
                                    middle_safety=middle_pred(middle)
                                    bottom=crop[3*t1:,:]
                                    cv2.imwrite('bottom.png',bottom)
                                    bot_safety=bot_pred(bottom)
                                    if bot_safety:
                                        print('bottom')
                                    if middle_safety:
                                        print('middle')
                                    if top_safety:
                                        print('top')         
                                    safety_lvl=top_safety+middle_safety+bot_safety

                                    text1 = "Safety:%s"%(safety_lvl)
                                    if safety_lvl==3:
                                        color1=(255,255,255)
                                    elif safety_lvl==2:
                                        color1=(0,255,0)
                                    elif safety_lvl==1:
                                        color1=(0,0,255)
                                    elif safety_lvl==0:
                                        color1=(255,0,0)    



                                    cv2.putText(frame, text1, (x, y-10),cv2.FONT_HERSHEY_SIMPLEX, .5, color1, 2)

                                
                                    if danger:
                                        text2='Danger Zone'
                                        cv2.putText(frame, text2, (x+w-10, y-15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                                    else:
                                        text2='Safe Zone'  
                                        cv2.putText(frame, text2, (x+w-10, y-15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)  

                                    if danger and safety_lvl!=3:
                                        text3='Equipment Danger'
                                        cv2.putText(frame, text3, (x, y+h+10),cv2.FONT_HERSHEY_SIMPLEX, .5, (255,0,0), 2)
                                    else:
                                        text3='-'
                                        cv2.putText(frame, text3, (x, y+h+10),cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,0), 2)  




                                   
                                
                                      
                                    
                               
                                cv2.rectangle(frame, (x, y), (x+ w, y +h),(0,0,255), 2)
                               
                                #cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                            except Exception as e:
                                print(e)
                                pass    

     
    return frame             



def read_vd1(name,cn1,top):
    cap=cv2.VideoCapture(name)
    
    fps=int(cap.get(5))
    print(fps)
    
    
   
    while(True):
        ret, frame = cap.read()
        if ret == True:   
           
            frame2=work(frame) 
            frame2=cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                            
            photo2 = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame2))
            cn1.create_image(0, 0, image = photo2, anchor = NW)
            top.update()
            
            

           
    
    # When everything done, release the video capture and video write objects
    cap.release()
    return 





from imutils import paths
if __name__=="__main__":
    from tkinter.filedialog import askopenfilename
    path=askopenfilename()
    frame=cv2.imread(path)
    frame=work(frame)
    cv2.imshow('result',frame)
    cv2.waitKey(0)

    




                                
                            
           
           
                    


                

    
 

##showface()
