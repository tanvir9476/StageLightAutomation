import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
import math
from pyfirmata import Arduino  #library to communicate with arduino 

options = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolov2.weights',
    'threshold': 0.3,
    'gpu': 0.6
}

#constant declareation
height=2.15 #stage to LIGHT distance
frameWidth=2.2 #stage width
height = height*1280/frameWidth #normalized height

#Arduino Initialization
board = Arduino('COM5')
ypin = board.get_pin('d:9:s')
xpin=board.get_pin('d:10:s')
board.servo_config(9, 544, 2400, 0)
board.servo_config( 10, 544, 2400, 0)
xpin.write(0)
ypin.write(0)

#tensorflow code to detect and track 
tfnet = TFNet(options)
colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

capture = cv2.VideoCapture(0)

capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)
posx=0
posy=0
time.sleep(5)
while True:
    stime = time.time()
    ret, frame = capture.read()
    if ret:
        results = tfnet.return_predict(frame)
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            confidence = result['confidence']
            text = '{}: {:.0f}%'.format(label, confidence * 100)
            frame = cv2.rectangle(frame, tl, br, color, 5)
            frame = cv2.putText(
                frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow('frame', frame)
        
        #Position Transformation & pin out 
        if result['label'] == 'person':
            (posx, posy) = ((result['topleft']['x']+result['bottomright']['x'])/2 , (result['topleft']['y']+result['bottomright']['y'])/2)
            phi =  90 - (math.atan(posy/posx)*180/math.pi)
            theta =  (math.atan(height/math.sqrt(posx*posx+posy*posy))*180/math.pi)
            xpin.write(phi-5)
            ypin.write(theta-25)
            print((phi,theta))
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
capture.release()
cv2.destroyAllWindows()
