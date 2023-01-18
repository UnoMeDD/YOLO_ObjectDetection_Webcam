import cv2
import numpy as np

# Load Yolo
print("LOADING YOLO")
net=cv2.dnn.readNet("yolov3-tiny.weights","yolov3-tiny.cfg")
#save all the names in file o the list classes
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
#get layers of the network
layer_names = net.getLayerNames()
#Determine the output layer names from the YOLO model 
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
print("YOLO LOADED")
