import numpy as np
import cv2
import sys
import os
import dlib
import time
import datetime
#read in gray scale
#file_name=input("input image name: ")
directory=sys.argv[1]
if not os.path.exists(os.getcwd()+'/gestures/'+directory):
	os.makedirs(os.getcwd()+'/gestures/'+directory)
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
				if (y+int(h*1.3)<(gray.shape[0]-1)):				    
					img= img[y:y+int(h*1.3), x:x+w]
				else:
					img= img[y:gray.shape[0]-1, x:x+w]
		return True
	else:
		return False
def side():
	global img
	global gray
	global side_cascade


	
	faces = side_cascade.detectMultiScale(gray, 1.1, 5)
	counter=0
	if len(faces)>0:
		for (x,y,w,h) in faces:
			if counter==0:
				counter+=1;		
				if (y+int(h*1.3)<(gray.shape[0]-1)):				    
					img= img[y:y+int(h*1.3), x:x+w]
				else:
					img= img[y:gray.shape[0]-1, x:x+w]
		return True
	else:
		return False
										
			    
  
		  

def keypoint():
	global img
	global gray
	global predictor
	global win
	global shape
	d=dlib.rectangle(0,0,img.shape[1]-1,img.shape[0]-1)
	shape = predictor(img, d)
	
	#for x in range(68):
	#	#cv2.circle(img,(shape.part(x)[0], shape.part(x)[1]), 2, (0,255,0), -1)
	#	print(shape.part(x))
shape=None
face_cascade = cv2.CascadeClassifier(os.getcwd()+'/lbp.xml')
side_cascade = cv2.CascadeClassifier(os.getcwd()+'/lbpside.xml')
predictor = dlib.shape_predictor(os.getcwd()+'/68point.dat')   
win = dlib.image_window()
file_name="face.jpg"

img = cv2.imread(file_name,1)
#print(img)


cap = cv2.VideoCapture(0)
count=0
while(True):
	count+=1
	#tries+=1
	ret, img = cap.read()

	img=cv2.resize(img,(int(img.shape[1]),int(img.shape[0])),cv2.INTER_AREA) #https://docs.opencv.org/3.3.0/d7/d8b/tutorial_py_face_detection.html
	#img=np.rot90(img)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	t0 = time.time()
	ha=haar()   #does haar cascade for daace detect

	win.clear_overlay()
	
	if ha==True:
		print('front')
		keypoint()
		
	#print( time.time()-t0)
	#cv2.imwrite("modified"+file_name,img)
	#https://stackoverflow.com/questions/4661557/pil-rotate-image-colors-bgr-rgb
		rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	
	
	
		
		win.add_overlay(shape)
		
		
		cv2.imwrite(os.getcwd()+'/gestures/'+directory+'/'+str(datetime.datetime.now())+'.jpg',img)
	else:

		see=side()
		if see:
			print('side')
			rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			cv2.imwrite(os.getcwd()+'/gestures/'+directory+'/'+str(datetime.datetime.now())+'.jpg',img)
	

		
	win.set_image(img)
	#win.wait_until_closed();
	#cv2.imshow("image", img);
	#cv2.waitKey(0);
	#print('here')
	
