from imutils.video import VideoStream
import time
import cv2
import glob
import os
from ArucoPoseCollection import ArucoPoseCollection

# remove all previous images
prev_images = glob.glob("ArucoImages/*.png")
for img in prev_images:
    os.remove(img)

counter = 0
# initialize the video stream
vs = VideoStream(1).start()
time.sleep(2.0)

# loop over the frames from the video stream
start_time = time.time()
stop_time = 0
while True:
    # capturing a single frame
    frame = vs.read()

    # show the frame
    cv2.imshow("Frame", frame)
    cv2.imwrite('ArucoImages' + '/' + str(counter) + 'raw.png', frame)
    counter += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop_time = time.time()
        break

ArucoPoseCollection(stop_time - start_time, counter, 17, 96)
vs.stop()
cv2.destroyAllWindows()
