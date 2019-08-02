from tkinter import Tk, Label, Button, Entry
import numpy as np
import cv2
import sys
import os
import dlib
import time
from PIL import Image, ImageTk
#https://www.pyimagesearch.com/2016/05/30/displaying-a-video-feed-with-opencv-and-tkinter/ tkitner with python video stream
class MyFirstGUI:
	def __init__(self, master):
		self.master = master
		master.title("A simple GUI")
		master.geometry("1000x1000") 

		img=dlib.load_grayscale_image("face.jpg")

		ini_image = ImageTk.PhotoImage(image = Image.fromarray(img))
		self.label = Label(master, text="Enter gesture name")
		self.label.pack()
		self.entry=Entry(master)
		self.entry.pack()

		
		self.pause = Button(master, text="Pause", command=master.quit)
		self.pause.pack()
		self.redo= Button(master, text="Redo Gesture", command=master.quit)
		self.redo.pack()
		self.next_gesture = Button(master, text="Next Gesture", command=master.quit)
		self.next_gesture.pack()
		
	def greet(self):
		print("Greetings!")



root = Tk()
my_gui = MyFirstGUI(root)
root.mainloop()
