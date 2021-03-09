import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def locationPlotter(locations, time):
    plt.title("ROV XY Location")
    plt.ylabel("Y (mm)")
    plt.xlabel("X (mm)")
    t = np.arange(len(locations))
    plt.scatter([x[0] for x in locations], [x[1] for x in locations],
                           c=t, s=20.0, cmap=cm.rainbow)
    plt.colorbar()
    # plt.show()
    plt.savefig('Path.png')

def yawPlotter(Yaw, time):
    plt.title("ROV Yaw")
    plt.ylabel("Yaw (Deg)")
    plt.xlabel("Time (s)")

    plt.scatter(np.linspace(0, time, len(Yaw)+1)[1:], Yaw, c='red', s=20.0)
    plt.savefig('Yaw.png')



def create_video(video_length, image_count):
    video_name = 'ArucoTracking.avi'

    images = []
    for img_num in range(image_count):
        images.append(cv2.imread("ArucoImages/" + str(img_num) + ".png"))
    frame = cv2.imread("ArucoImages/0.png")
    height, width, layers = frame.shape
    #
    video = cv2.VideoWriter(video_name, 0, len(images)/video_length, (width,height))

    for image in images:
        video.write(image)

    cv2.destroyAllWindows()
    video.release()