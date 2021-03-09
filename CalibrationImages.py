import cv2
import time

# ---------------------------------------------------------------------------------------
#       < User Inputs >
cam = 1 # camera to read from (1 for USB-connected)
# s: save image
# q: quit program
# ---------------------------------------------------------------------------------------

cap = cv2.VideoCapture(cam)
counter = 0
time.sleep(1.0)
print("camera connected")

while(True):
    # capture frame from video
    ret, frame = cap.read()
    # convert frame to greyscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # display the frame
    cv2.imshow('frame',frame)
    # Capture user input [ s: save, q: quit]
    key = cv2.waitKey(1) & 0xFF

    # saving image for calibration
    if key == ord('s'):
        name = str(counter) + "_cal.jpg"
        cv2.imwrite(name, frame)
        counter = counter + 1

    # quitting
    if key == ord('q'):
        break

# performing cleanup actions before ending
cap.release()
cv2.destroyAllWindows()