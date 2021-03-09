import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R # for euler angles
from HelperFunctions import locationPlotter
from HelperFunctions import yawPlotter
from HelperFunctions import create_video

def ArucoPoseCollection(elapsed_time, image_count, ID, markerLength):
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

    # load in images
    images = []
    for img_num in range(image_count):
        images.append(cv2.imread("ArucoImages/" + str(img_num) + "raw.png"))
    frame = cv2.imread("ArucoImages/0raw.png")
    h, w, _ = frame.shape

    # compute new camera matrix for calibration purposes
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    # for graphing ROV 2D pose
    Locations = []
    Yaw = []

    # loop over the frames from the video stream
    for frame in images:
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
                # drawing axis and box around marker
                cv2.aruco.drawAxis(frame, newcameramtx, dist, rvec, tvec, 40)
                (topLeft, topRight, bottomRight, bottomLeft) = corners[0][0]
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))
                cv2.line(frame, topLeft, topRight, (64,64,64), 5)
                cv2.line(frame, topRight, bottomRight, (64,64,64), 5)
                cv2.line(frame, bottomRight, bottomLeft, (64,64,64), 5)
                cv2.line(frame, bottomLeft, topLeft, (64,64,64), 5)

                # capturing 2D pose information
                # rotation_matrix = cv2.Rodrigues(rvec)[0]
                # Yaw.append(R.as_euler('zyx',degrees=True))
                r = R.from_rotvec(rvec[0])
                theta = r.as_euler('zxy', degrees=True)[0][0] + 180
                x = tvec[0][0][0]
                y = tvec[0][0][1]
                # performing transform to account for offset mounting of ArUco marker
                g_Ac = [[1, 0, 50],[0, 1, 50],[0, 0, 1]]
                g_0A = [[np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta)), x],
                        [-1 * np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta)), y],
                        [0, 0, 1]]
                ROVpos = np.matmul(np.matmul(g_0A, g_Ac), [0, 0, 1])
                # storing 2D pose information
                Yaw.append(theta)
                Locations.append((ROVpos[0], ROVpos[1]))
                break
        # show the output frame
        cv2.imshow("Frame", frame)
        cv2.imwrite('ArucoImages' + '/' + str(counter) + '.png', frame)
        counter += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # producing pose2D graph, video
    create_video(elapsed_time, image_count)
    plt.figure(0)
    locationPlotter(Locations, elapsed_time)
    plt.figure(1)
    yawPlotter(Yaw, elapsed_time)
    # cleanup
    cv2.destroyAllWindows()
    time.sleep(1)
