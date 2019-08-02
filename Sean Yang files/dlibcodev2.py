import sys
import os
import dlib
import glob
import numpy as np
import timeit
import cv2
import time

def distance(p1,p2):
	return np.array([p1.x-p2.x,p1.y-p2.y]) 
def distance2(p1,p2):
	return (p1.x-p2.x,p1.y-p2.y) 

cap = cv2.VideoCapture(0)
predictor_path = os.getcwd()+"/68point.dat"


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
win = dlib.image_window()



#img = dlib.load_rgb_image(os.getcwd()+"/face.jpg")




#win.set_image(img)

# Ask the detector to find the bounding boxes of each face. The 1 in the
# second argument indicates that we should upsample the image 1 time. This
# will make everything bigger and allow us to detect more faces.
work=list()
while(True):
	ret, img = cap.read()
	img = cv2.resize(img, dsize=(int(img.shape[1]),int(img.shape[0])), interpolation=cv2.INTER_CUBIC)
	
	work=list()
	win.set_image(img)
	dets = detector(img, 1)
	
	print("Number of faces detected: {}".format(len(dets)))
	win.clear_overlay()

	for k, d in enumerate(dets):
		
		shape = predictor(img, d)
		
		comp=shape.part(30)

		for x in range(68):
			a=distance(shape.part(x),comp)
			work.append(a[0])
			work.append(a[1])
				# Draw the face landmarks on the screen.
		win.add_overlay(shape)

	print(work)
	#win.add_overlay(dets)
	#time.sleep(0.1)

	



