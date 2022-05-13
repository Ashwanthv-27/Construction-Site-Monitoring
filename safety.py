import tkinter as tk
from PIL import ImageTk, Image
import sqlite3,csv
from tkinter import messagebox
#from camera2 import main
from tkinter.filedialog import askopenfilename
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import messagebox,DISABLED,NORMAL
# import pymysql
import datetime
from functools import partial
from PIL import Image, ImageTk
# from testing import process
import time
title="Construction Safety"

import PIL.Image, PIL.ImageTk


def logcheck():
     global username_var,pass_var
     uname=username_var.get()
     pass1=pass_var.get()
     if uname=="" and pass1=="":
        showcheck()
     else:
        messagebox.showinfo("alert","Wrong Credentials")   



def showcheck():
    top.title(title)
    top.config(menu=menubar)
    global f,f1,f_bottom,f_top,f_b1

    f.pack_forget()
    f=Frame(top)
    f.config(bg="#000000")
    f.pack(side="top", fill="both", expand=True,padx=10,pady=10)
    
    f_bottom=Frame(f)
    f_bottom.config(bg="#000000",width=1500,height=130)
    f_bottom.pack_propagate(False)
    f_bottom.pack(side='bottom',fill='both')

    f_b1=Frame(f_bottom)
    f_b1.config(bg="#000000",width=500,height=100)
    f_b1.pack_propagate(False)
    f_b1.pack(side='top')


    f_top=Frame(f)
    f_top.config(bg="#000000",height=800,width=1500)
    f_top.pack_propagate(False)
    f_top.pack(side='bottom',fill='both')


   

    

    global f2
    f2=Frame(f_top)
    f2.pack_propagate(False)
    f2.config(bg="#000000",width=1800,height=800)
    f2.pack(side="left",fill="both")
    
    


    


    global lb1,cn1,cn2

    

    # for x in range(100):
    #     lb1.insert(END, str(x))
    b2=Button(f_b1,text="Choose video",font="Verdana 10 bold",command=detect)
    b2.pack(pady=5,padx=100)
    
    
    
    
    global cn1,c211,c212,c221,c222,c231,c232,f3a
    cn1 =Canvas(f2, width =1200, height = 800)
    cn1.pack(padx=12,side='top')
   


   
    
   

    

    
   

    

    
    
    
    

    
    
    
from os import listdir
from os.path import isfile, join
from yolo1 import read_vd1
def detect():
    global lb1,sflag,cn1,top,cn2,f4,b2,f_bottom,d_clicked,b4,b5,c231,c221,c211,forged,fc
    f=askopenfilename()
    read_vd1(f,cn1,top)

   
        
    
def delayed_insert(label,index,message):
    label.insert(0,message)  



import threading
def insert1(label,msg):
    label.insert(0,message) 
    

def delayed_insert(label,index,message):
    label.insert(0,message) 

    





import cv2

import numpy as np
# from LBP import lbp
import os

from PIL import ImageFilter





    




if __name__=="__main__":

    top = Tk()  
    top.title("Login")
    top.geometry("1900x700")
    footer = Frame(top, bg='grey', height=30)
    footer.pack(fill='both', side='bottom')

    lab1=Label(footer,text="Developed by ",font = "Verdana 8 bold",fg="white",bg="grey")
    lab1.pack()

    menubar = Menu(top)  
    # menubar.add_command(label="Home",command=showhome)  
    menubar.add_command(label="Detection",command=showcheck)

    top.config(bg="#000000",relief=RAISED)  
    f=Frame(top)
    f.config(bg="#000000")
    showcheck()
    top.mainloop()

   

