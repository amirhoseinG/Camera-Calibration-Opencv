from cv2 import undistort , imread , imwrite
from numpy import asarray 
from json import load


# opencv read your images 
image = imread("Enter image path")

# read .json file and load data
with open('data.json', 'r') as f:
  data = load(f)

# opencv undistort image 
calibration = undistort(image , cameraMatrix= asarray(data["OldCameraMatrix"]), distCoeffs= asarray(data["dist"]), newCameraMatrix= asarray(data["NewCameraMatrix"]) )
calibration = calibration[data["Tuple"]["y"]:data["Tuple"]["y"]+data["Tuple"]["h"] , data["Tuple"]["x"]:data["Tuple"]["x"]+data["Tuple"]["w"]]

# opencv write and save your new image 
imwrite("calibration.jpg", calibration)