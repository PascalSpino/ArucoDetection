import numpy as np
import cv2
import glob

# ---------------------------------------------------------------------------------------
#       < User Inputs >
tileLength = 96 # mm (square side length)
col = 8 # internal corners along width
row = 6 # internal corners along height
# ---------------------------------------------------------------------------------------

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# real world coordinates of internal tile corners (units of mm)
objp = tileLength * np.zeros((row*col,3), np.float32)
objp[:,:2] = np.mgrid[0:row,0:col].T.reshape(-1,2)

# arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# calibration images (collected in CalibrationImages.py)
images = glob.glob('*_cal.jpg')

for fname in images:
    # reading single image, converting to greyscale (may already be)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # finding the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (row,col),None)

    # if found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # draw and display the corners for the user
        img = cv2.drawChessboardCorners(img, (row,col), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)

# cleanup
cv2.destroyAllWindows()

# obtaining camera matrix, distortion coefficients, rotation and translation vectors
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

# save camera matrix and distortion results to a file for calibration
np.savetxt('Cam_Matrix.txt', mtx)
np.savetxt('Dist_Coeff.txt', dist)