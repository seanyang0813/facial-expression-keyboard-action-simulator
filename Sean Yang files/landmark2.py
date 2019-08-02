import numpy as np
import cv2
import sys
import os
import dlib
import time
#read in gray scale
#file_name=input("input image name: ")
def haar():
	global img
	global gray
	global face_cascade

	
	faces = face_cascade.detectMultiScale(gray, 1.1, 5)
	counter=0
	if len(faces)>0:
		for (x,y,w,h) in faces:
			if counter==0:
				counter+=1;		
				if (y+int(h*1.2)<(gray.shape[0]-1)):				    
					img= img[y:y+int(h*1.2), x:x+w]
				else:
					img= img[y:gray.shape[0]-1, x:x+w]
					
			    
  
		  

def keypoint():
	global img
	global gray
	global predictor
	global win
	global shape
	d=dlib.rectangle(0,0,img.shape[1]-1,img.shape[0]-1)
	shape = predictor(img, d)
	
	for x in range(68):
		#cv2.circle(img,(shape.part(x)[0], shape.part(x)[1]), 2, (0,255,0), -1)
		print(shape.part(x))
shape=None
face_cascade = cv2.CascadeClassifier(os.getcwd()+'/lbp.xml')
predictor = dlib.shape_predictor(os.getcwd()+'/68point.dat')   
win = dlib.image_window()
file_name="face.jpg"

img = cv2.imread(file_name,1)
print(img)
img=cv2.resize(img,(int(img.shape[1]/10),int(img.shape[0]/10)),cv2.INTER_AREA) #https://docs.opencv.org/3.3.0/d7/d8b/tutorial_py_face_detection.html
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
t0 = time.time()
haar()   #does haar cascade for daace detect

keypoint()
print( time.time()-t0)
#cv2.imwrite("modified"+file_name,img)
#https://stackoverflow.com/questions/4661557/pil-rotate-image-colors-bgr-rgb
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



win.clear_overlay()
win.add_overlay(shape)
win.set_image(rgb)
win.wait_until_closed();

print('here')
