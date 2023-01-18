import numpy as np
import xml.dom.minidom as md
import cv2
import glob
import json


term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# creat numpy array based on chessboard
obj_points = np.zeros((6*9,3), np.float32)
obj_points[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
real_points = []
img_points = []

# load images with glob save them in python list format 
chess_images = glob.glob("*.jpg")

for name in chess_images:
    # opencv read image file
    chess_img = cv2.imread(name)
    # BGR color image convert to GRAY format
    chess_gray = cv2.cvtColor(chess_img, cv2.COLOR_BGR2GRAY)

    # Find chessboard
    ret, corners = cv2.findChessboardCorners(chess_gray, (9,6), None)
    if ret == True:
        real_points.append(obj_points)
        corners2 = cv2.cornerSubPix(chess_gray,corners, (11,11), (-1,-1), term_criteria)
        img_points.append(corners)
        cv2.drawChessboardCorners(chess_gray, (9,6), corners2, ret)
        cv2.imshow("name",chess_gray)
       

        

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(real_points, img_points, chess_gray.shape[::-1], None, None)
img = cv2.imread('C:/Users/LENOVO/Downloads/job/New folder/opencv/calib/WIN_20230110_16_36_34_Pro.jpg')

h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

x, y, w, h = roi



XMLfile = md.parse('dict.xml')
node = XMLfile.getElementsByTagName( "framesCount" )[0].childNodes
node[0].nodeValue = len(chess_images)
node = XMLfile.getElementsByTagName( "cameraResolution" )[0].childNodes
node[0].nodeValue = h , w
node = XMLfile.getElementsByTagName( "data" )[0].childNodes
node[0].nodeValue = [list(list_i) for list_i in mtx]
node = XMLfile.getElementsByTagName( "data" )[1].childNodes
node[0].nodeValue = [list(list_i) for list_i in newcameramtx]
node = XMLfile.getElementsByTagName( "data" )[2].childNodes
node[0].nodeValue = [list(list_i) for list_i in dist]
node = XMLfile.getElementsByTagName( "data" )[3].childNodes
node[0].nodeValue = roi

with open( "dict.xml", "w" ) as fs: 
  
        fs.write( XMLfile.toxml() )
        fs.close()
