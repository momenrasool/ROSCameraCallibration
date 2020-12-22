#!/usr/bin/env python

import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import rospy
import sys

def start_node(filename):

	#import matplotlib.image as mpimg

	cap = cv2.VideoCapture(0)

	# Check if the webcam is opened correctly
	if not cap.isOpened():
	    raise IOError("Cannot open webcam")

	# prepare object points

	nx = 7 # number of inside corners in x
	ny = 7 # number of inside corners in y 

	# Arrays to store object points and image points from all the images

	objpoints = [] # 3D points in real world space
	imgpoints = [] # 2D points in image plane

	objp = np.zeros((1,ny*nx,3), np.float32)
	objp[0,:,:2] =  np.mgrid[0:nx,0:ny].T.reshape(-1,2) # x,y coordinates

	while True:
	    ret, frame = cap.read()
	    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
	    
	    # Convert to grayscale

	    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	    
	    # Find the chessboard corners
	    
	    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
	    
	    # If found, draw corners
	    
	    if ret == True:
	    
	       # Draw and display the corners       
	       cv2.drawChessboardCorners(frame, (nx, ny), corners, ret)
	       
	    cv2.imshow('Input', frame)
	 
	    c = cv2.waitKey(1)
	    if c == 27:
	       break
	    if c == 32 and ret == True:
	       imgpoints.append(corners)
	       objpoints.append(objp)
	    
	       print('New Image Point Added. Collected {} Samples.'.format(len(imgpoints)))
	    if c == 99 and len(imgpoints) > 0:
		
	       cap.release()
	       cv2.destroyAllWindows()
	
	       img = cv2.imread(filename)
	       
	       ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1:], None, None)
	       print("Camera matrix : \n")
	       print(mtx)
	       print("dist : \n")
	       print(dist)
	       print("rvecs : \n")
	       print(rvecs)
	       print("tvecs : \n")
	       print(tvecs)
	       undist = cv2.undistort(img, mtx, dist, None, mtx)
	       
	       f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
	       f.tight_layout()
	       ax1.imshow(img)
	       ax1.set_title('Original Image', fontsize=50)
	       ax2.imshow(undist)
	       ax2.set_title('Undistorted Image', fontsize=50)
	       plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
	       plt.show()
	       break
	       
if __name__ == '__main__':
    try:
        start_node(rospy.myargv(argv=sys.argv)[1])
    except rospy.ROSInterruptException:
        pass
