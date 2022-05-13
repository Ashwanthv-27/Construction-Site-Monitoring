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

font = cv2.FONT_HERSHEY_SIMPLEX
green = (0, 255, 0)
red = (0, 0, 255)
line_type = cv2.LINE_AA
IMAGE_SIZE = 224



def work(frame,c):
    h,w=frame.shape[:2]
    ar=w/h
    #if ar>0.2:
     #   cv2.imwrite('person1/no/%s.png'%(str(c)),frame)
      #  c+=1
    #else:
    #cv2.imwrite('person1/yes/%s.png'%(str(c)),frame)
      
    t1=h//4
    top=frame[:t1,:]
    middle=frame[t1:3*t1,:]
    bottom=frame[3*t1:,:]
    cv2.imwrite('top/%s.png'%(str(c)),top)
    cv2.imwrite('middle/%s.png'%(str(c)),middle)
    cv2.imwrite('bottom/%s.png'%(str(c)),bottom)
    c+=1

        




    
         
    return c               



from imutils import paths
if __name__=="__main__":
    c=0
    imagePaths = sorted(list(paths.list_images('person')))
    for imgp in imagePaths:
        img=cv2.imread(imgp)
        c=work(img,c)
        




                                
                            
           
           
                    


                

    
 

##showface()
