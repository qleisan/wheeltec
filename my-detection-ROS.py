# wheeltec use ROS 1
# cv2 is only installed in python 2.7

# HOWTO RUN (NB! python 2.7) 
#
#   (shell#1) roscore
#   (shell#2) roslaunch usb_cam usb_cam-test.launch
#   (shell#3) python my-detection-ROS.py 
#

import rospy
from sensor_msgs.msg import Image

from cv_bridge import CvBridge, CvBridgeError
import cv2

import jetson.inference
import jetson.utils
from time import sleep


net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)


# Instantiate CvBridge
bridge = CvBridge()

def image_callback(msg):
    print("Received an image!")
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError, e:
        print(e)
    else:
        # Save your OpenCV2 image as a jpeg 
        cv2.imwrite('camera_image_X.jpeg', cv2_img)
        #img = jetson.utils.loadImage("camera_image_X.jpeg")
        # TODO: for now load another image with interesting content
        img = jetson.utils.loadImage("peds_0.jpg")
        #img = jetson.utils.loadImage("my-recognition-python/polar_bear.jpg")
        detections = net.Detect(img)
        for detection in detections:
            print(net.GetClassDesc(detection.ClassID))

def main():
    rospy.init_node('image_listener')
    # Define your image topic
    #image_topic = "/cameras/left_hand_camera/image"
    image_topic = "/camera/rgb/image_raw"
    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic, Image, image_callback)
    print("start to spin...")
    # Spin until ctrl + c
    rospy.spin()

if __name__ == '__main__':
    main()