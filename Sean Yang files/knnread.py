import sys
import os
import dlib
import glob
import numpy as np
import timeit
import cv2
import time
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

def distance(p1,p2):
	global normalize
	
	return np.array([p1.x-p2.x,p1.y-p2.y]) 
#cap = cv2.VideoCapture(0)
predictor_path = os.getcwd()+"/68point.dat"

win = dlib.image_window()
detector = dlib.get_frontal_face_detector()
#detector = dlib.cnn_face_detection_model_v1(os.getcwd()+"/cnn.dat")
predictor = dlib.shape_predictor(predictor_path)
xlist=list()
ylist=list()
dictionary=dict()

count=0
for i in os.listdir(os.getcwd()+"/gestures"):	
	print(i)
	count+=1
	for j in os.listdir(os.getcwd()+"/gestures/"+str(i)):
		#print(os.getcwd()+"/gestures/"+str(x)+"/"+str(y))
		#tries+=1
		#ret, img = cap.read()
		
		print(os.getcwd()+"/gestures/"+str(i)+"/"+str(j))
		img = dlib.load_rgb_image(os.getcwd()+"/gestures/"+i+"/"+str(j))	
		
		#win.set_image(img)
		dets = detector(img, 1)
		
		#print("Number of faces detected: {}".format(len(dets)))
		#win.clear_overlay()
		for k, d in enumerate(dets):
			forpca=None
			
			d = dlib.rectangle(d.left(), d.top(), d.right(), int(d.bottom()))
			#print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
			# Get the landmarks/parts for the face in box d.
			
			shape = predictor(img, d)
			
			#print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))
			comp=shape.part(29)

			
			for x in range(0,47,2):
				if x==0:
					forpca=distance(shape.part(x),comp)
				else:
					forpca=np.vstack([forpca,distance(shape.part(x),comp)])
					# Draw the face landmarks on the screen.
			
			temp=distance(shape.part(18),shape.part(23))
			normalize=(temp[1:1]**2+temp[2:2])**0.5
			
			for x in range(47,68):
				#if x==47:
				#	forpca=distance(shape.part(x),comp)
				forpca=np.vstack([forpca,distance(shape.part(x),comp)*50])
				
			#win.add_overlay(shape)
			#t1=timeit.timeit()
			pca = PCA(n_components=2)			
			pca.fit(forpca)
			if count not in dictionary:
				dictionary[count]=[pca.explained_variance_ratio_]
			else:
				dictionary[count].append(pca.explained_variance_ratio_)
			xlist.append(pca.explained_variance_ratio_)
			ylist.append(i)
			
			
			#print(pca.explained_variance_ratio_)  
#print(dictionary)
for x in dictionary:
	print(dictionary[x])
	print('\nfor gesture '+str(x)+ "the average covariance are")
	print('x component:')
	print(sum([y[0] for y in dictionary[x]])/len(dictionary[x]))
	print('y component:')
	print(sum([y[1] for y in dictionary[x]])/len(dictionary[x]))
neigh = KNeighborsClassifier(n_neighbors=2)
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
		forpca=None
		d = dlib.rectangle(d.left(), d.top(), d.right(), int(d.bottom()))
		#print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
		# Get the landmarks/parts for the face in box d.
		
		shape = predictor(img, d)
		
		win.add_overlay(shape)
		
		#print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))
		comp=shape.part(29)
		
		for x in range(0,47,2):
			if x==0:
				forpca=distance(shape.part(x),comp)
			else:
				forpca=np.vstack([forpca,distance(shape.part(x),comp)])
				# Draw the face landmarks on the screen.
		
		temp=distance(shape.part(18),shape.part(23))
		normalize=(temp[1:1]**2+temp[2:2])**0.5
		
		for x in range(47,68):
			#if x==47:
			#	forpca=distance(shape.part(x),comp)
			forpca=np.vstack([forpca,distance(shape.part(x),comp)*5])
			
		#win.add_overlay(shape)
		#t1=timeit.timeit()
		pca = PCA(n_components=2)			
		pca.fit(forpca)
		print(pca.explained_variance_ratio_)
		print(neigh.predict([pca.explained_variance_ratio_]))
	
		