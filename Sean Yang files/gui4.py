import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
from tkinter import *
from tkinter import Tk, Label, Button, Entry, StringVar
import sys
import os
import dlib
import glob
import numpy as np
import timeit
import cv2
import datetime
#from https://solarianprogrammer.com/2018/04/21/python-opencv-show-video-tkinter-window/
class App:
    def __init__(self, window, window_title, video_source=0):        
        self.cur_folder = ""
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1000x1000") 
        self.video_source = video_source

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

      
        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()
        self.btn_text = StringVar()
        self.cur_gesture = StringVar()
        '''
        self.folder_label = Label(window, text="Click gesture you want to submit. Press enter to submit")
        self.folder_label.pack(side=TOP)

        self.text_field = Entry(window)
        self.text_field.pack(side=TOP)
        self.text_field.bind("<Return>", (lambda event: self.register_folder(self.text_field.get(),self.vid)))
        '''

        # Buttons to choose the right gesture
        self.btn_left = tkinter.Button(window, text = "Left Gesture", width=50, command=self.left)
        self.btn_left.pack(side=TOP)

        self.btn_neutral = tkinter.Button(window, text = "neutral Gesture", width=50, command=self.neutral)
        self.btn_neutral.pack(side=TOP)

        self.btn_left = tkinter.Button(window, text = "right Gesture", width=50, command=self.right)
        self.btn_left.pack(side=TOP,pady =5)

        self.folder_label = Label(window, text="selected gesture is :")
        self.folder_label.pack(side=LEFT)

        self.folder_label = Label(window, textvariable = self.cur_gesture)
        self.folder_label.pack(side=LEFT)

        # Button that lets the user take a snapshot
        self.btn_snapshot=tkinter.Button(window, textvariable=self.btn_text, width=50, command=self.changeMode)
        self.btn_snapshot.pack(side=LEFT, padx= 100)
        self.btn_text.set("Start capturing gesture")
        self.cur_gesture.set("None")

                
        
        self.redo= Button(window, text="close gesture", command=window.quit)
        self.redo.pack(side = LEFT,padx=100 )    
         # After it is called once, the update method will be automatically called every delay milliseconds
       
        self.delay = 2
        self.update()
        self.window.mainloop() 
        
       

    def changeMode(self):
        # change stop and start
        print(self.vid.is_capturing)
        if self.vid.is_capturing==True:
            self.btn_text.set("Start capturing gesture")
            self.vid.is_capturing=False
        else:
            self.btn_text.set("Stop capturing gesture")
            self.vid.is_capturing=True
        
    def right (self):
        self.vid.directory = "right"  
        print('cur gesture is right')
        self.cur_gesture.set("right")              
    
    def neutral (self):
        self.vid.directory = "neutral" 
        print('cur gesture is neutral')
        self.cur_gesture.set("neutral") 
    
    def left (self):
        self.vid.directory = "left"  
        print('cur gesture is left')    
        self.cur_gesture.set("left")         
    
    def register_folder(self,text,vid):        
        self.cur_folder = text
        vid.directory = self.cur_folder
        print(vid.directory)
        print(text)
        return None
        
    def update(self):
        # Get a frame from the video source       
        ret, frame = self.vid.get_frame()      

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)

        self.window.after(self.delay, self.update)
    


class MyVideoCapture:
    is_capturing=False
    predictor_path = os.getcwd()+"/68point.dat"   
    directory=''
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if self.directory != '':                   
            if not os.path.exists(os.getcwd()+'/gesture_test/'+self.directory):
                print('directory made sucessful')
                os.makedirs(os.getcwd()+'/gesture_test/'+self.directory)
         
            if not self.vid.isOpened():
                raise ValueError("Unable to open video source", video_source)
        
        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
   
    def get_frame(self):
        if self.directory != '':                   
            if not os.path.exists(os.getcwd()+'/gesture_test/'+self.directory):
                print('directory made sucessful')
                os.makedirs(os.getcwd()+'/gesture_test/'+self.directory)
         
            if not self.vid.isOpened():
                raise ValueError("Unable to open video source", video_source)
        if self.vid.isOpened():
            ret, img = self.vid.read()
            img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            if self.is_capturing==True:
                dets = self.detector(img, 1)
       

                for k, d in enumerate(dets):
                    forpca=None
                
                    d = dlib.rectangle(d.left(), d.top(), d.right(), int(d.bottom()))
                    print(d.left())
                    tempimg= img[d.top(): d.bottom(), d.left(): d.right()]
                    if self.directory != '':  
                        print(os.getcwd()+'/gesture_test/'+self.directory+"/"+( ''.join([x for x in str(datetime.datetime.now()) if x!=":" and x!="." ])+'.jpg'))
                        cv2.imwrite(os.getcwd()+'/gesture_test/'+self.directory+"/"+(''.join([x for x in str(datetime.datetime.now()) if x!=":" and x!="." ]))+'.jpg',tempimg,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    else:
                        print("directory not specified")                           
                    img=tempimg
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, img)
            else:
                return (ret, None)
        else:
            return (ret, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# Create a window and pass it to the Application object
App(tkinter.Tk(), "Capture script")
