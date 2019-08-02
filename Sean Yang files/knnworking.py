import sys
import os
import dlib
import glob
import numpy as np
import timeit
import cv2
import time
import math
import keyboard
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
def get_angle(top_pt,bottom_pt):
	refpt=dlib.point(bottom_pt.x,bottom_pt.y-1)
	s1=1
	s2=distance(top_pt,bottom_pt)
	opposite=distance(top_pt,refpt)
	angle=math.acos((s1**2+s2**2-opposite**2)/(2*s1*s2))
	if bottom_pt.x-top_pt.x>0:		
		return angle
		#angle is positive
	else:
		#angle is negative
		return -angle
def rotate_list(angle,shape,origin):
	'''
	angle is angle to rotate
	shape is the output of the predictor
	origin is origin to rotate around

	return a list of points containing rotated points
	'''
	store=list()
	origin_x=origin.x
	origin_y=origin.y

	for num in range(68):
		
		point=shape.part(num)
		
		curx=point.x
		cury=point.y
		x=origin_x+math.cos(angle)*(curx-origin_x)-math.sin(angle)*(cury- origin_y)
		y=origin_y+math.sin(angle)*(curx-origin_x)+math.cos(angle)*(cury- origin_y)
		store.append(dlib.point(int(x),int(y)))
	
	stored=dlib.full_object_detection(dlib.rectangle(0,0,200,200),store)
	return stored
		
		
		
	


def distance(p1,p2):
	#magnitude of the vector between points
	global normalize
	
	return ((p1.x-p2.x)**2+(p1.y-p2.y)**2)**0.5
def distance2(p1,p2):
	#helper function to return vector between points
	return (p1.x-p2.x,p1.y-p2.y)
#cap = cv2.VideoCapture(0)
predictor_path = os.getcwd()+"/68point.dat"

win = dlib.image_window()
detector = dlib.get_frontal_face_detector()
#cnn_face_detector= dlib.cnn_face_detection_model_v1(os.getcwd()+"/mmod.dat")
predictor = dlib.shape_predictor(predictor_path)
xlist=list()
ylist=list()
dictionary=dict()
prev_gesture=None
gesture_count=0
count=0
tempmat=list()
for i in os.listdir(os.getcwd()+"/gestures4"):	
	print(i)
	count+=1
	for j in os.listdir(os.getcwd()+"/gestures4/"+str(i)):
		#print(os.getcwd()+"/gestures/"+str(x)+"/"+str(y))
		#tries+=1
		#ret, img = cap.read()
		
		print(os.getcwd()+"/gestures4/"+str(i)+"/"+str(j))
		img = dlib.load_grayscale_image(os.getcwd()+"/gestures4/"+i+"/"+str(j))	
		faces = [dlib.rectangle(0, 0, img.shape[1],img.shape[0])]
		win.set_image(img)
		
		
		#print("Number of faces detected: {}".format(len(dets)))
		win.clear_overlay()
		for d in faces:
			forpca=list()
			
			#d = dlib.rectangle(d.left(), d.top(), d.right(), int(d.bottom()))
			#print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
			# Get the landmarks/parts for the face in box d.
			
			shape = predictor(img, d)  # this is mmod rectangle so we need .rect
			
			#print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))
			comp=shape.part(29)
			temp=distance2(shape.part(27),shape.part(30))
			normalize=(temp[0]**2+temp[1]**2)**0.5
			angles=get_angle(shape.part(27),shape.part(30))
			print(angles)
			rotated=rotate_list(angles,shape,shape.part(30))
			win.add_overlay(rotated)



			for x in range(0,47,4):
				if x==0:
					dis=distance2(rotated.part(x),comp)
					forpca.append(dis[0]/normalize)
					forpca.append(dis[1]/normalize)
				else:
					dis=distance2(rotated.part(x),comp)
					forpca.append(dis[0]/normalize)
					forpca.append(dis[1]/normalize)
					# Draw the face landmarks on the screen.
			
			
			
			for x in range(47,68):
				#if x==47:
				#	forpca=distance(shape.part(x),comp)
				dis=distance2(rotated.part(x),comp)
				forpca.append(dis[0]*5/normalize)
				forpca.append(dis[1]*5/normalize)

			
			win.add_overlay(shape)
			#t1=timeit.timeit()
		
			if count not in dictionary:
				dictionary[count]=[forpca]
			else:
				dictionary[count].append(forpca)
			#print(forpca)
			xlist.append(forpca)
			ylist.append(i)
			
			
			#print(pca.explained_variance_ratio_)  
#print(dictionary)\
'''
for x in dictionary:
	print(dictionary[x])
	print('\nfor gesture '+str(x)+ "the average covariance are")
	print('x component:')
	print(sum([y[0] for y in dictionary[x]])/len(dictionary[x]))
	print('y component:')
	print(sum([y[1] for y in dictionary[x]])/len(dictionary[x]))
'''

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(xlist, ylist)
cap = cv2.VideoCapture(0)
prev=None
continues=0
while(True):
	#tries+=1

	ret, img = cap.read()
	#img = cv2.resize(img, dsize=(int(img.shape[1]),int(img.shape[0])), interpolation=cv2.INTER_CUBIC)
	img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	win.set_image(img)
	dets = detector(img, 1)
		
		#print("Number of faces detected: {}".format(len(dets)))
	win.clear_overlay()
	for k, d in enumerate(dets):
		forpca=list()
		#print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
		# Get the landmarks/parts for the face in box d.
		
		shape = predictor(img, d)

		
		
		
		#print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))
		comp=shape.part(29)
		temp=distance2(shape.part(27),shape.part(30))
		normalize=(temp[0]**2+temp[1]**2)**0.5
		angles=get_angle(shape.part(27),shape.part(30))
		
		rotated=rotate_list(angles,shape,shape.part(30))
		win.add_overlay(shape)



		for x in range(0,47,4):
			if x==0:
				dis=distance2(rotated.part(x),comp)
				forpca.append(dis[0]/normalize)
				forpca.append(dis[1]/normalize)
			else:
				dis=distance2(rotated.part(x),comp)
				forpca.append(dis[0]/normalize)
				forpca.append(dis[1]/normalize)
				# Draw the face landmarks on the screen.
		
		
		
		for x in range(47,68):
			#if x==47:
			#	forpca=distance(shape.part(x),comp)
			dis=distance2(rotated.part(x),comp)
			forpca.append(dis[0]*5/normalize)
			forpca.append(dis[1]*5/normalize)
			
		#win.add_overlay(shape)
		#t1=timeit.timeit()
		
		
		cur_gesture=str(neigh.predict(np.asarray(forpca).reshape(1,-1))[0])
		pred_thresh=(neigh.predict_proba(np.asarray(forpca).reshape(1,-1)))
		print(pred_thresh)
		threshhold=0.5
		cur_thresh_max=np.max([pred_thresh[0]])
		if cur_thresh_max<threshhold:
			cur_gesture='neutral'
		if cur_gesture!=prev_gesture:
			prev_gesture=cur_gesture
			gesture_count=0
		else:
			gesture_count+=1
		if gesture_count==3:
			if prev_gesture=='left':
				keyboard.press('up')
			if prev_gesture=='right':
				keyboard.press('down')
		print(cur_gesture)
		

