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

import jetson.inference
import jetson.utils
from time import sleep

import numpy as np

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

def image_callback(msg):
    print("Received an image!")
    print(msg.encoding, msg.width, msg.height)
    print(type(msg.data))

    im = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    print(im.shape)

    cuda_img = jetson.utils.cudaAllocMapped(width=640, height=480, format='rgb8')
    cepa = jetson.utils.cudaToNumpy(cuda_img)

    for y in range(0,msg.height):
        for x in range(0,msg.width):
            cepa[y,x,0] = im[y,x,0]
            cepa[y,x,1] = im[y,x,1]
            cepa[y,x,2] = im[y,x,2]

    jetson.utils.saveImage("cuda_test.jpg", cuda_img)
    print("--------------")
    # TODO: for now load another image with interesting content
    img = jetson.utils.loadImage("peds_0.jpg")
    #img = jetson.utils.loadImage("my-recognition-python/polar_bear.jpg")
    #detections = net.Detect(img)
    detections = net.Detect(cuda_img)
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