from picamera import PiCamera
import time

camera = PiCamera()
camera.resolution = (640, 480) # setting resolution to reduce latency

camera.start_preview()
time.sleep(2)

# camera.capture('test.jpeg')
camera.start_recording('my_movie2.h264')
time.sleep(5) #record 5 seconds
camera.stop_recording()