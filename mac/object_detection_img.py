import cv2
import numpy as np

# List of class names known to the pre-trained model
# Note: Adjust the class names below according to the classes your MobileNet-SSD is trained on.
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Load the pre-trained MobileNet SSD model and the prototxt file
prototxt = 'deploy.prototxt'
model = 'mobilenet_iter_73000.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Load the image from file
image = cv2.imread(' ') # insert your image to test
# Get the dimensions of the image
(h, w) = image.shape[:2]

# Preprocess the image: mean subtraction, scaling, and then pass it through the neural network
blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
net.setInput(blob)
detections = net.forward()

# Loop over the detections and draw bounding boxes on the image
for i in np.arange(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    # Filter out weak detections by ensuring the confidence is greater than a minimum confidence
    if confidence > 0.2:
        # Get the index of the class label from the detections
        idx = int(detections[0, 0, i, 1])

        # Get the bounding box coordinates
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # Draw the bounding box and label on the image
        label = f"{CLASSES[idx]}: {confidence:.2f}"
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(image, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the output image
cv2.imshow('Output', image)

# Wait indefinitely until a key is pressed, then close the display window
cv2.waitKey(0)
cv2.destroyAllWindows()
