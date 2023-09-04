from picamera import PiCamera # picamera doesn't seem to work in a virtual environment
import time
import cv2

camera = PiCamera()
time.sleep(0.1)
camera.capture('opencv_imgtest.jpg')

# read image from card
image = cv2.imread('opencv_imgtest.jpg', -1)

# display image in an openCV window
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
# cv2.resize('Image', 640, 480)
cv2.imshow('Image', image)
cv2.waitKey(0)
