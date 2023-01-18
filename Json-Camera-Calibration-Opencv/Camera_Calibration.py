import numpy as np
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
        cv2.waitKey(500)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(real_points, img_points, chess_gray.shape[::-1], None, None)
img = cv2.imread("ENTER YOUR IMAGE PATH HERE")

h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

x, y, w, h = roi

# update calibration data to data.json file 

with open('data.json', 'r') as f:
  data = json.load(f)


data.update({"Tuple" : {"x": x,"y": y,"h": h,"w": w}})
data.update({"OldCameraMatrix" : [list(list_i) for list_i in mtx] })
data.update({"dist" :  [list(list_i) for list_i in dist]}) 
data.update({"NewCameraMatrix" : [list(list_i) for list_i in newcameramtx]})

# write in data.json file
with open('data.json', 'w') as f:
  data = json.dump(data , f)
