from imutils.video import VideoStream
import imutils
import time
import cv2
import numpy as np

# ---------------------------------------------------------------------------------------
#       < User Inputs >
cam = 0 # camera to read from (1 for USB-connected)
markerLength = 100 # mm (marker dimensions, square side length)
# ---------------------------------------------------------------------------------------

# load camera calibration data
mtx = np.loadtxt('Cam_Matrix.txt')
dist = np.loadtxt('Dist_Coeff.txt')

# define the names of each possible ArUco tag that OpenCV supports
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

# load the chosen ArUCo dictionary and grab the ArUCo parameters
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT["DICT_4X4_100"])
arucoParams = cv2.aruco.DetectorParameters_create()

# initialize the video stream
vs = VideoStream(src=cam).start()
time.sleep(2.0)

# compute new camera matrix for calibration purposes
frame = vs.read()
h, w = frame.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# loop over the frames from the video stream
while True:
    # capturing a single frame
    frame = vs.read()
    # undistorted with calibration data
    frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    # setting max width to 600 pixels
    frame = imutils.resize(frame, width=600)
    # converting to greyscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detecting ArUco markers in processed frame
    (corners, ids, rejected) = cv2.aruco.detectMarkers(gray,
                                                       arucoDict, parameters=arucoParams)
    if ids is None:
        ids = []

    # verify that at least one ArUco marker was detected
    if len(ids) == 1:

        # capturing rotation and translation vectors for detected marker
        rvec, tvec, objPts = cv2.aruco.estimatePoseSingleMarkers(corners, markerLength,
                                                                 newcameramtx, dist)

        # drawing the axis of the marker
        cv2.aruco.drawAxis(frame, newcameramtx, dist, rvec, tvec, 40)

        # outlining the detected marker
        (topLeft, topRight, bottomRight, bottomLeft) = corners[0][0]
        topRight = (int(topRight[0]), int(topRight[1]))
        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
        topLeft = (int(topLeft[0]), int(topLeft[1]))
        cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
        cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
        cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
        cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)

        # displaying the distance to the marker on the frame
        distance = np.sqrt(np.square(tvec[0][0][0]) + np.square(tvec[0][0][1]) + np.square(tvec[0][0][2]))
        distance = int(distance)
        Dis_str = 'Distance: ' + str(distance) + ' mm'
        cv2.putText(frame, Dis_str,
                    (topLeft[0], topLeft[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 2)
    elif len(ids) > 1:
        print('Multiple markers detected!')


    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# cleanup
cv2.destroyAllWindows()
vs.stop()