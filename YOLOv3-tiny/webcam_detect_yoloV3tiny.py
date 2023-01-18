import cv2
import numpy as np
import time

from loadYOLO import net, classes, output_layers
from bounding_box import bounding_box as bb
from enum import Enum

colors = ["green", "red", "blue"]

start_time = time.time()
display_time = 2
fps = 0
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    re,img = video_capture.read()
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    # USing blob function of opencv to preprocess image
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (224, 224),
     swapRB=True, crop=False)
    #Detecting objects
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                left = int(center_x - w / 2)
                top = int(center_y - h / 2)
                right = int(center_x + w / 2)
                bottom = int(center_y + h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    #We use NMS function in opencv to perform Non-maximum Suppression
    #we give it score threshold and nms threshold as arguments.
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    colors = np.random.uniform(0, 255, size=(len(classes), 0))
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            confidence = "%.2f" % confidences[i]
            label = str(classes[class_ids[i]] + " " + confidence)
            color = colors[class_ids[i]]
            color = "green"
            #cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            #cv2.putText(img, label, (x, y), font, 0.75, color, 2)
            bb.add(img, left, top, right, bottom, label, color)

    cv2.imshow("Image",cv2.resize(img, (800,600)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    fps += 1
    TIME = time.time() - start_time

    if TIME > display_time:
        print("FPS: ", fps / TIME)
        seconds = str("%.2f " % TIME)
        print("Inference time: " + seconds)
        fps = 0
        start_time = time.time()

video_capture.release()
cv2.destroyAllWindows()
