# imports
from cv2 import log
from configs import config
from configs.detection import detect_people
from configs.smsnotif import sms_email_notification
from configs.mailer import Mailer
from scipy.spatial import distance as dist 
from analytics.recorded import recorded_plot
import numpy as np
import argparse
import imutils
import cv2
import os
import pyttsx3
import threading
from timeit import default_timer as timer
import json
import csv
import pandas as pd
from datetime import date
from subprocess import Popen
from urllib.request import urlopen
Popen('python analytics/realtime.py')
#analytics
today = date.today()
date = today.strftime("%Y-%m-%d")
x_value = 0
totalViolations = 0
realtimeFields = ["x_value", "config.Human_Data", "detectedViolators", "totalViolations" ]
recordedFields = ["date", "averagePerson", "averageViolator", "averageViolation" ]

esp32url = 'http://192.168.100.61/capture'

with open('realtimeData.csv', 'w') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=realtimeFields)
    csv_writer.writeheader()

# Initial list of points for top down view
f = open('test-config.json','r')
TopdownPointConfig = json.loads(f.read())
list_points = list((
    TopdownPointConfig['TopLeft'],
    TopdownPointConfig['TopRight'],
    TopdownPointConfig['BottomLeft'],
    TopdownPointConfig['BottomRight']
))
f.close()
TopLeft_calibrate = False
TopRight_calibrate = False
BottomLeft_calibrate = False
BottomRight_calibrate = False
Calibrate_checker = False
FrameViewSelector = 1

#mouse click callback for top down conversion
def CallBackFunc(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Left button of the mouse is clicked - position (", x, ", ",y, ")")
        global TopLeft_calibrate, TopRight_calibrate, BottomLeft_calibrate, BottomRight_calibrate, Calibrate_checker, list_points
        if TopLeft_calibrate == True:
            list_points[0] = [x,y]
            TopdownPointConfig["TopLeft"] = [x,y]
            f = open("test-config.json", "w")
            json.dump(TopdownPointConfig, f)
            f.close()
            TopLeft_calibrate = False
            Calibrate_checker = False
        if TopRight_calibrate == True:
            list_points[1] = [x,y]
            TopdownPointConfig["TopRight"] = [x,y]
            f = open("test-config.json", "w")
            json.dump(TopdownPointConfig, f)
            f.close()
            TopRight_calibrate = False
            Calibrate_checker = False
        if BottomLeft_calibrate == True:
            list_points[2] = [x,y]
            TopdownPointConfig["BottomLeft"] = [x,y]
            f = open("test-config.json", "w")
            json.dump(TopdownPointConfig, f)
            f.close()
            BottomLeft_calibrate = False
            Calibrate_checker = False
        if BottomRight_calibrate == True:
            list_points[3] = [x,y]
            TopdownPointConfig["BottomRight"] = [x,y]
            f = open("test-config.json", "w")
            json.dump(TopdownPointConfig, f)
            f.close()
            BottomRight_calibrate = False
            Calibrate_checker = False

#text to speech converter
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
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov4.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov4.cfg"])

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

# old input condition
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)

writer = None
# loop over the frames from the video stream
while True:    
    # num += 1
    # read the next frame from the input video
    if args["input"] == "":    
        imgResp=urlopen("http://192.168.0.190/capture")
        imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
        esp32=cv2.imdecode(imgNp,-1)
    else:
        (grabbed, esp32) = vs.read()
        # if the frame was not grabbed, then that's the end fo the stream
        if not grabbed:
            break
 
    # resize the frame and then detect people (only people) in it
    frame = imutils.resize(esp32, width=1200)
    birdeyeframe= imutils.resize(esp32, width=1200)
    results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))

    # initialize the set of indexes that violate the minimum social distance
    violate = set()
    TopDownViolate = set()

    #initialize variables for bird eye conversion
    array_ground_points = list()
    array_boxes = list()

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
        array_ground_points.append((cX, endY))
        array_boxes.append((startX,startY,endX,endY))
        color = (0, 255, 0)

        # if the index pair exists within the violation set, then update the color
        if i in violate:
            color = (0, 0, 255)

        # draw (1) a bounding box around the person and (2) the centroid coordinates of the person
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame, (cX, cY), 5, color, 1)
        cv2.line(frame, list_points[0], list_points[1], (225,0,0), 1)
        cv2.line(frame, list_points[0], list_points[2], (225,0,0), 1)
        cv2.line(frame, list_points[1], list_points[3], (225,0,0), 1)
        cv2.line(frame, list_points[2], list_points[3], (225,0,0), 1)
    
    #threshold value
    Threshold = "Threshold limit: {}".format(config.Threshold)
    cv2.putText(frame, Threshold, (350, frame.shape[0] - 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)

    # draw the total number of social distancing violations on the output frame
    text = "Social Distancing Violations: {}".format(len(violate))
    detectedViolators = format(len(violate))
    cv2.putText(frame, text, (18, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    #total no of violations
    totalViolation = "Violation Voice Warning Total: {}".format(totalViolations)
    cv2.putText(frame, totalViolation, (18, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    #----------------------------------------------------------Bird eye-------------------------------------------------------------#
    # bird eye view
    if (config.TOP_DOWN):
        TopDownFreezeImage = cv2.imread('TopDown.jpg')
        #top left
        if (TopLeft_calibrate == True):
            cv2.circle (frame, list_points[0], 5, (0,255,0), -1)
        else:
            cv2.circle (frame, list_points[0], 5, (0,0,255), -1)
        #top right
        if (TopRight_calibrate == True):
            cv2.circle (frame, list_points[1], 5, (0,255,0), -1)
        else:
            cv2.circle (frame, list_points[1], 5, (0,0,255), -1)
        #bottom left
        if (BottomLeft_calibrate == True):
            cv2.circle (frame, list_points[2], 5, (0,255,0), -1)
        else:
            cv2.circle (frame, list_points[2], 5, (0,0,255), -1)
        #bottom right
        if (BottomRight_calibrate == True):
            cv2.circle (frame, list_points[3], 5, (0,255,0), -1)
        else:
            cv2.circle (frame, list_points[3], 5, (0,0,255), -1)
        width = 350
        height = 650

        pts1 = np.float32([list_points[0], list_points[1], list_points[2], list_points[3]])
        pts2 = np.float32([[0,0], [width,0], [0,height], [width,height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        blank_image = np.zeros((height,width,3), np.uint8)
        warpedImage = cv2.warpPerspective(birdeyeframe, matrix, (width, height))
        result = cv2.warpPerspective(birdeyeframe, matrix, (width, height))
        list_points_to_detect = np.float32(array_ground_points).reshape(-1, 1, 2)
        transformed_points = cv2.perspectiveTransform(list_points_to_detect, matrix)
        transformed_points_list = list()
        
        if(len(results) >= 2):
            for i in range(0,transformed_points.shape[0]):
                transformed_points_list.append([int(transformed_points[i][0][0]),int(transformed_points[i][0][1])])
        
        #separate distance calculator for top-down view
        if(len(results) >= 2):
            TopDownCentroids = np.array([point for point in transformed_points_list])
            TopDownD = dist.cdist(TopDownCentroids, TopDownCentroids, metric="euclidean")

            # loop over the upper triangular of the distance matrix
            for i in range(0, TopDownD.shape[0]):
                for j in range(i+1, TopDownD.shape[1]):
                    # check to see if the distance between any two centroid pairs is less
                    # than the configured number of pixels
                    if TopDownD[i, j] < config.MIN_DISTANCE:
                        # update the violation set with the indexes of the centroid pairs
                        TopDownViolate.add(i)
                        TopDownViolate.add(j)

        for point in transformed_points_list:
            x,y = point
            BIG_CIRCLE = 20  
            SMALL_CIRCLE = 3
            COLOR = (0, 255, 0)
            if transformed_points_list.index(point) in TopDownViolate:
                COLOR = (0,0,255)
            switcher = {
                1: warpedImage,
                2: TopDownFreezeImage,
                3: blank_image,
            }
            selectedview = switcher.get(FrameViewSelector, warpedImage)
            cv2.circle(selectedview, (int(x),int(y)), BIG_CIRCLE, COLOR, 2)
            cv2.circle(selectedview, (int(x),int(y)), SMALL_CIRCLE, COLOR, -1)

        cv2.imshow("Bird Eye View", selectedview)
    
    #------------------------------Alert function----------------------------------#
    if len(violate) >= config.Threshold:
        if (t.is_alive() == True):
            violationTimer = timer()
        else:
            t = threading.Thread(target=voice_alarm)
            if config.SENDSMS:
                t2 = threading.Thread(target=sms_email_notification)
            if (round((timer() - violationTimer), 2) > config.TIMERTHRESHOLD):
                if config.ALERT:
                    totalViolations += 1
                    t.start()     
                    if config.SENDSMS:
                        t2.start()
    else:
        violationTimer = timer()
        
    # check to see if the output frame should be displayed to the screen
    if args["display"] > 0:
        # show the output frame
        cv2.imshow("Output", frame)
        key = cv2.waitKey(1) & 0xFF

        # bind the callback function to window
        cv2.setMouseCallback("Output", CallBackFunc)
        CallBackFunc

        # if the 'q' key is pressed, break from the loop
        if key == ord("q"):
            break

        if key == ord("1"):
            if TopLeft_calibrate == False and Calibrate_checker == False:
                TopLeft_calibrate = True
                Calibrate_checker = True
            elif TopLeft_calibrate == True and Calibrate_checker == True:
                TopLeft_calibrate = False
                Calibrate_checker = False
                
        if key == ord("2"):
            if TopRight_calibrate == False and Calibrate_checker == False:
                TopRight_calibrate = True
                Calibrate_checker = True
            elif TopRight_calibrate == True and Calibrate_checker == True:
                TopRight_calibrate = False 
                Calibrate_checker = False

        if key == ord("3"):
            if BottomLeft_calibrate == False and Calibrate_checker == False:
                BottomLeft_calibrate = True
                Calibrate_checker = True
            elif BottomLeft_calibrate == True and Calibrate_checker == True:
                BottomLeft_calibrate = False
                Calibrate_checker = False

        if key == ord("4"):
            if BottomRight_calibrate == False and Calibrate_checker == False:
                BottomRight_calibrate = True
                Calibrate_checker = True
            elif BottomRight_calibrate == True and Calibrate_checker == True:
                BottomRight_calibrate = False
                Calibrate_checker = False

        if key == ord("i"):
            cv2.imwrite("TopDown.jpg", warpedImage)
        
        if key == ord("v"):
            FrameViewSelector += 1
            if FrameViewSelector > 3:
                FrameViewSelector = 1

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

    #-----------------------------------Records Realtime Data----------------------------------------------------------
    #-----------------------------------Records Realtime Data----------------------------------------------------------
    #-----------------------------------Records Realtime Data----------------------------------------------------------
    #-----------------------------------Records Realtime Data----------------------------------------------------------
    #-----------------------------------Records Realtime Data----------------------------------------------------------
    #-----------------------------------Records Realtime Data----------------------------------------------------------
    with open('realtimeData.csv', 'a') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=realtimeFields)

        info = {
            "x_value": x_value,
            "config.Human_Data": config.Human_Data,
            "detectedViolators": detectedViolators,
            "totalViolations": totalViolations,
        }
        csv_writer.writerow(info)

        x_value += 1
        config.Human_Data = config.Human_Data
        detectedViolators = detectedViolators
        totalViolations = totalViolations


# Records Average Data after the loop
if config.ATTACH:
    df= pd.read_csv ('realtimeData.csv')
    #get average per column
    averagePerson = round(df['config.Human_Data'].mean(), 0)
    averageViolator = round(df['detectedViolators'].mean(), 0)
    averageViolation = round(df['totalViolations'].mean(), 0)

    with open('recordedData.csv', 'a') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=recordedFields)
        info = {
                "date": date,
                "averagePerson": averagePerson,
                "averageViolator": averageViolator,
                "averageViolation": averageViolation,
            }
        csv_writer.writerow(info)

        date = date
        averagePerson = averagePerson
        averageViolator = averageViolator
        averageViolation = averageViolation
    recorded_plot()
    # Send recorded data through email
    Mailer().sendData(config.MAIL)

#Clean up, Free memory
cv2.destroyAllWindows
quit()

    
    



