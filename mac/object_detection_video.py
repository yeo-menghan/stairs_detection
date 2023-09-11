# Refer to https://github.com/chuanqi305/MobileNet-SSD for the trained dataset

import cv2
import numpy as np

# Load the pre-trained MobileNet SSD model and the prototxt file
prototxt = 'deploy.prototxt'
model = 'mobilenet_iter_73000.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Load the list of class labels (from the COCO dataset) the MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Initialize the webcam feed
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Prepare the frame for object detection
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # Perform object detection
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections and draw bounding boxes on the frame
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Only consider detections with confidence greater than a threshold (e.g., 0.2)
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with the bounding boxes
    cv2.imshow('Object Detection', frame)

    # Break the loop when the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
