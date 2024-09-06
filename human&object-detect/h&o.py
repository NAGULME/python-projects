# curl -O https://pjreddie.com/media/files/yolov3.weights
# curl -O https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
# curl -O https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

import cv2
import numpy as np

# Load the YOLOv3 weights and configuration files
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load the COCO dataset classes
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define the layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load the video capture device (e.g. a webcam)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size to improve performance

while True:
    # Read a frame from the video capture device
    ret, frame = cap.read()

    if not ret:
        break

    # Get the frame dimensions
    height, width, channels = frame.shape

    # Create a blob from the frame and pass it through the network
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize the detected objects
    class_ids = []
    confidences = []
    boxes = []

    # Iterate over the detections
    for out in outs:
        for detection in out:
            # Get the scores, class_id, and confidence
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter out weak detections
            if confidence > 0.5:
                # Get the bounding box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate the top-left corner coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Append the detection to the lists
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw the bounding boxes
    font = cv2.FONT_HERSHEY_SIMPLEX
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + confidence, (x, y + 20), font, 1, color, 2)

    # Display the frame
    cv2.imshow("Image", frame)

    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and close the window
cap.release()
cv2.destroyAllWindows()
