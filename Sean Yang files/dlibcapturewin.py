import sys
import os
import dlib
import glob
import numpy as np
import timeit
import cv2
import time
import datetime
from sklearn.decomposition import PCA

def distance(p1,p2):
	global normalize
	
	return np.array([p1.x-p2.x,p1.y-p2.y]) 
cap = cv2.VideoCapture(0)
predictor_path = os.getcwd()+"/68point.dat"
if len(sys.argv)>1:
	directory=sys.argv[1]
else:
	directory='test'
if not os.path.exists(os.getcwd()+'/gestures/'+directory):
	os.makedirs(os.getcwd()+'/gestures/'+directory)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
win = dlib.image_window()



#img = dlib.load_rgb_image(os.getcwd()+"/face.jpg")




#win.set_image(img)

# http://dlib.net/face_landmark_detection.py.html
tries=0
while(True):
	tries+=1
	ret, img = cap.read()
	img = cv2.resize(img, dsize=(int(img.shape[1]),int(img.shape[0])), interpolation=cv2.INTER_CUBIC)
	img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	
	
	dets = detector(img, 1)

	
	print("Number of faces detected: {}".format(len(dets)))
	win.clear_overlay()
	for k, d in enumerate(dets):
		forpca=None
	
		d = dlib.rectangle(d.left(), d.top(), d.right(), int(d.bottom()))
		print(d.left())
		tempimg= img[d.top(): d.bottom(), d.left(): d.right()]
		print(os.getcwd()+'/gestures/'+directory+"/"+( ''.join([x for x in str(datetime.datetime.now()) if x!=":" and x!="." ])+'.jpg'))
		cv2.imwrite(os.getcwd()+'/gestures/'+directory+"/"+(''.join([x for x in str(datetime.datetime.now()) if x!=":" and x!="." ]))+'.jpg',tempimg,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
		img=tempimg
	win.set_image(img)		

				# Get the landmarks/parts for the face in box d.
'''
		shape = predictor(img, d)
		
		print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))
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
			
		win.add_overlay(shape)
		t1=timeit.timeit()
		pca = PCA(n_components=2)
		pca.fit(forpca)
		print(pca.explained_variance_ratio_)  
		#print(timeit.timeit()-t1)
		#print(forpca)
'''

	#win.add_overlay(dets)
	#time.sleep(0.1)
	#print(forpca)

	



