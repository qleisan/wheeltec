#import cv2
import jetson.inference
import jetson.utils
from time import sleep
import requests

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
#camera = jetson.utils.videoSource("csi://0")      # '/dev/video0' for V4L2
display = jetson.utils.videoOutput("display://0") # 'my_video.mp4' for file

#while display.IsStreaming():
while True:
    img_data = requests.get("http://192.168.203.75:8080/snapshot?topic=/camera/rgb/image_rect_color").content
    with open('image_name.jpg', 'wb') as handler:
        handler.write(img_data)

    img = jetson.utils.loadImage("image_name.jpg")
#	img = camera.Capture()
    detections = net.Detect(img)
    display.Render(img)
    display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
    sleep(10)
    #exit(0)
print("hello")
input_img = jetson.utils.loadImage("peds_0.jpg")
