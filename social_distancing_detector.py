# imports
from calendar import c
from configs import config
from configs.mailer import Mailer
from configs.detection import detect_people
from scipy.spatial import distance as dist 
import numpy as np
import argparse
import imutils
import cv2
import os
import pyttsx3
import threading
import time

# Create an empty list of points for the coordinates
list_points = list()

#mouse click callback for top down conversion
def CallBackFunc(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Left button of the mouse is clicked - position (", x, ", ",y, ")")
        list_points.append([x,y])
    elif event == cv2.EVENT_RBUTTONDOWN:
        print("Right button of the mouse is clicked - position (", x, ", ", y, ")")
        list_points.append([x,y])

#text to speech converter
stopFrameCheck = False
def voice_alarm():
    engine = pyttsx3.init()
    # engine.stop()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1)
    engine.say("Please observe social distancing")
    engine.runAndWait()

t = threading.Thread(target=voice_alarm)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="", help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="", help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1, help="whether or not output frame should be displayed")
args = vars(ap.parse_args())

# load the COCO class labels the YOLO model was trained on
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO tiny weights and model configuration
# weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3-tiny.weights"])
# configPath = os.path.sep.join([config.MODEL_PATH, "yolov3-tiny.cfg"])

# derive the paths to the YOLO tiny weights and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

# load the YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# check if GPU is to be used or not
if config.USE_GPU:
    # set CUDA s the preferable backend and target
    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the "output" layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream and pointer to output video file
print("[INFO] accessing video stream...")
# open input video if available else webcam stream
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None
violations = 0
previousFrameViolation = 0
# loop over the frames from the video stream
while True:
    # read the next frame from the input video
    (grabbed, frame) = vs.read()
    # if the frame was not grabbed, then that's the end fo the stream 
    if not grabbed:
        break
    
    if (stopFrameCheck == False):
        if (previousFrameViolation != 0):
            frameCounter += 1
        
        else:
            frameCounter = 0
    
    else:
        frameCounter = 0
 
    # resize the frame and then detect people (only people) in it
    frame = imutils.resize(frame, width=1200)
    results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))

    # initialize the set of indexes that violate the minimum social distance
    violate = set()

    # ensure there are at least two people detections (required in order to compute the
    # the pairwise distance maps)
    if len(results) >= 2:
        # extract all centroids from the results and compute the Euclidean distances
        # between all pairs of the centroids
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")

        # loop over the upper triangular of the distance matrix
        for i in range(0, D.shape[0]):
            for j in range(i+1, D.shape[1]):
                # check to see if the distance between any two centroid pairs is less
                # than the configured number of pixels
                if D[i, j] < config.MIN_DISTANCE:
                    # update the violation set with the indexes of the centroid pairs
                    violate.add(i)
                    violate.add(j)
    
    # loop over the results
    for (i, (prob, bbox, centroid)) in enumerate(results):
        # extract teh bounding box and centroid coordinates, then initialize the color of the annotation
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)

        # if the index pair exists within the violation set, then update the color
        if i in violate:
            color = (0, 0, 255)

        # draw (1) a bounding box around the person and (2) the centroid coordinates of the person
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame, (cX, cY), 5, color, 1)

    
    # cv2.imshow("qweqwe", centroid)
    Threshold = "Threshold limit: {}".format(config.Threshold)
    cv2.putText(frame, Threshold, (350, frame.shape[0] - 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
    # draw the total number of social distancing violations on the output frame
    text = "Social Distancing Violations: {}".format(len(violate))
    cv2.putText(frame, text, (18, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    #total no of violations
    totalViolation = "Total Violation Warning: {}".format(violations)
    cv2.putText(frame, totalViolation, (18, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    
    # bird eye view sample 
    if (config.TOP_DOWN):
        #top left
        cv2.circle (frame, (690, 5), 5, (0,0,255), -1)
        #top right
        cv2.circle (frame, (1160, 5), 5, (0,0,255), -1)
        #bottom left
        cv2.circle (frame, (5, 390), 5, (0,0,255), -1)
        #bottom right
        cv2.circle (frame, (1000, 660), 5, (0,0,255), -1)

        pts1 = np.float32([[690, 5], [1160, 5], [5, 390], [1000, 660]])
        pts2 = np.float32([[0,20], [350,0], [0,650], [350,650]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(frame, matrix, (350, 650))
        cv2.imshow("Bird Eye View", result)
    

    #------------------------------Alert function----------------------------------#
    if len(violate) >= config.Threshold:
        if (t.is_alive() == True):
            stopFrameCheck = True
        else:
            t = threading.Thread(target=voice_alarm)
            if (frameCounter >= config.frameLimit):      
                violations += 1
                t.start()
            stopFrameCheck = False
            

        previousFrameViolation = format(len(violate))
        # cv2.putText(frame, "-ALERT: Violations over limit-", (10, frame.shape[0] - 80),
        #     cv2.FONT_HERSHEY_COMPLEX, 0.60, (0, 0, 255), 2)
        
        if config.ALERT:
            Mailer().send(config.MAIL)

    else:
        previousFrameViolation = 0
        
    text1 = "Frame Counter: {}".format(frameCounter)
    cv2.putText(frame, text1, (350, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    # check to see if the output frame should be displayed to the screen
    if args["display"] > 0:
        # show the output frame
        cv2.imshow("Output", frame)
        key = cv2.waitKey(1) & 0xFF

        # bind the callback function to window
        cv2.setMouseCallback("Output", CallBackFunc)

        # if the 'q' key is pressed, break from the loop
        if key == ord("q"):
            break
        # if p is pressed, pause
        if key == ord('p'):
            cv2.waitKey(-1) #wait until any key is pressed
    
    # if an output video file path has been supplied and the video writer ahs not been 
    # initialized, do so now
    if args["output"] != "" and writer is None:
        # initialize the video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 25, (frame.shape[1], frame.shape[0]), True)

    # if the video writer is not None, write the frame to the output video file
    if writer is not None:
        # print("[INFO] writing stream to output")
        writer.write(frame)

#Clean up, Free memory
cv2.destroyAllWindows

    
    



