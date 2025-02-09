# base path to YOLO directory
from sqlalchemy import false


MODEL_PATH = "yolo-coco"

# initialize minimum probability to filter weak detections along with the
# threshold when applying non-maxim suppression
MIN_CONF = 0.3
NMS_THRESH = 0.3
#Set the threshold value for total violations limit.
Threshold = 15
#time counted before violation warning
TIMERTHRESHOLD = 3
SENDSMS = False
#top down view
TOP_DOWN = True
# email alert
ALERT = True
MAIL = 'lulumopanot@gmail.com'
# should NVIDIA CUDA GPU be used?
USE_GPU = True
# people counter
People_Counter = True
# define the minimum safe distance (in pixels) that two people can be from each other
MIN_DISTANCE = 200
# analytics
Human_Data = 0
ATTACH = False