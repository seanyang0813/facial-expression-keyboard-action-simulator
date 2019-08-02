import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
	ret, frame = cap.read()
	
	#detector = dlib.get_frontal_face_detector()
	#predictor = dlib.shape_predictor(args["shape_predictor"])
	factor=320/frame.shape[1] #640 length 
	
	resized_image = cv2.resize(frame, (int(frame.shape[1]*factor),int(frame.shape[0]*factor) )) 
	# Display the resulting frame
	cv2.namedWindow( "frame",CV_WINDOW_FULLSCREEN)
	cv2.imshow('frame',resized_image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
