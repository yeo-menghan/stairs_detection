# this program uses OpenCV to capture video footage. It does not record the video
# possible use case: real-time object detection

from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

# instantiate and configure picamera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
raw_capture = PiRGBArray(camera, size=(640, 480))

# let camera module warm up
time.sleep(0.1)

# define an opencv window to display video
cv2.namedWindow("Frame")

# capture continuous frames to access video
for frame in camera.capture_continuous(raw_capture, format='bgr', use_video_port=True):
    image = frame.array
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF
    raw_capture.truncate(0)
    # if 'q' is pressed, close opencv window and end video
    if key != ord('q'):
        pass
    else:
        cv2.destroyAllWindows()
        break