from imutils.video import VideoStream
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from HelperFunctions import rotationMatrixToEulerAngles
from HelperFunctions import locationPlotter
from HelperFunctions import create_video
def ArucoPoseCollection(elapsed_time, image_count, w, h, ID, markerLength):
    counter = 0
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

    # initializing images
    vs = VideoStream(src=cam).start()
    time.sleep(2.0)

    # compute new camera matrix for calibration purposes
    frame = vs.read()
    # w, h = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    # for graphing ROV 2D pose
    Locations = []
    Yaw = []

    # loop over the frames from the video stream
    start_time = time.time()
    stop_time = 0
    while True:
        # capturing a single frame
        frame = vs.read()
        # undistorted with calibration data
        frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        # converting to greyscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detecting ArUco markers in processed frame
        (corners, ids, rejected) = cv2.aruco.detectMarkers(gray,
                                                           arucoDict, parameters=arucoParams)
        if ids is None:
            ids = []

        for el in range(len(ids)):
            if ids[el] == ID:
                # capturing rotation and translation vectors for detected marker
                rvec, tvec, objPts = cv2.aruco.estimatePoseSingleMarkers(corners[el], markerLength,
                                                                         newcameramtx, dist)
                cv2.aruco.drawAxis(frame, newcameramtx, dist, rvec, tvec, 40)
                # print("rvec: ", rvec)
                rotation_matrix = cv2.Rodrigues(rvec)[0]
                # print("rodrigues: ", rotation_matrix)
                euler_angles = rotationMatrixToEulerAngles(rotation_matrix)
                # cv2.decomposeProjectionMatrix(newcameramtx,)
                # plt.scatter(-1 * tvec[0][0][0], -1 * tvec[0][0][1], s=10.0, color=next(colors))
                # plt.pause(0.05)
                Locations.append((tvec[0][0][0], tvec[0][0][1]))
                break
        # show the output frame
        cv2.imshow("Frame", frame)
        cv2.imwrite('ArucoImages' + '/' + str(counter) + '.png', frame)
        counter += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_time = time.time()
            break

    # producing pose2D graph, video
    create_video(stop_time - start_time, counter + 1)
    plt.plot()
    locationPlotter(Locations)
    # cleanup
    vs.stop()
    cv2.destroyAllWindows()
    time.sleep(1)