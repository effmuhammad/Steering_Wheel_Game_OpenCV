import numpy as np
import cv2
import imutils
import time
from imutils.video import VideoStream
from directkeys import PressKey, A, D, Space, ReleaseKey

fps=0
prev_time = time.time()
h_nit = 40
cam = VideoStream(src=0).start()
currentKey = list()

while True:
    key = False

    img = cam.read()
    img = np.flip(img,axis=1)
    img = imutils.resize(img, width=640)
    img = imutils.resize(img, height=480)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    value = (11, 11)
    blurred = cv2.GaussianBlur(hsv, value,0)
    colourLower = np.array([0, 80, 170])
    colourUpper = np.array([180,255,255])

    height = img.shape[0]
    width = img.shape[1]

    mask = cv2.inRange(blurred, colourLower, colourUpper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))

    upContour = mask[0:height//2,0:width]
    downContour = mask[3*height//4+h_nit:height,2*width//5:3*width//5]

    cnts_up = cv2.findContours(upContour, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts_up = imutils.grab_contours(cnts_up)


    cnts_down = cv2.findContours(downContour, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts_down = imutils.grab_contours(cnts_down)

    left, right, space = False, False, False
    if len(cnts_up) > 0:
        c = max(cnts_up, key=cv2.contourArea)
        M = cv2.moments(c)
        cX = int(M["m10"]/(M["m00"]+0.000001))

        if cX < (width//2 - 35):
            PressKey(A)
            key = True
            currentKey.append(A)
            left = True
        elif cX > (width//2 + 35):
            PressKey(D)
            key = True
            currentKey.append(D)
            right = True

    if len(cnts_down) > 0:
        PressKey(Space)
        key = True
        currentKey.append(Space)
        space = True
    
    # tampilkan FPS
    if time.time()-prev_time >= 1:
        prev_time = time.time()
        fps_str = 'FPS : ' + str(fps)
        fps = 0
    fps += 1                                                                                     

    cv2.putText(img, fps_str, (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
 
    img = cv2.rectangle(img,(0,0),(width//2- 35,height//2 ),(0,255,0),1)
    cv2.putText(img,'LEFT',(110,30),cv2.FONT_HERSHEY_DUPLEX,1,(139,0,0))
    if left:
        overlay1 = img.copy()
        cv2.rectangle(overlay1, (0, 0), (width//2- 35,height//2 ), (0, 200, 0), -1)  # A filled rectangle
        alpha = 0.4  # Transparency factor.
        img = cv2.addWeighted(overlay1, alpha, img, 1 - alpha, 1)

    img = cv2.rectangle(img,(width//2 + 35,0),(width-2,height//2 ),(0,255,0),1)
    cv2.putText(img,'RIGHT',(440,30),cv2.FONT_HERSHEY_DUPLEX,1,(139,0,0))
    if right:
        overlay2 = img.copy()
        cv2.rectangle(overlay2, (width//2 + 35,0),(width-2,height//2 ), (0, 200, 0), -1)  # A filled rectangle
        alpha = 0.4  # Transparency factor.
        img = cv2.addWeighted(overlay2, alpha, img, 1 - alpha, 1)

    img = cv2.rectangle(img,(2*(width//5),3*(height//4)+h_nit),(3*width//5,height),(0,255,0),1)
    cv2.putText(img,'NITRO',(2*(width//5) + 20,height-10),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,0))
    if space:
        overlay3 = img.copy()
        cv2.rectangle(overlay3, (2*(width//5),3*(height//4)+h_nit),(3*width//5,height), (0, 200, 0), -1)  # A filled rectangle
        alpha = 0.4  # Transparency factor.
        img = cv2.addWeighted(overlay3, alpha, img, 1 - alpha, 1)

    cv2.imshow("Steering", img)

    if not key and len(currentKey) != 0:
        for current in currentKey:
            ReleaseKey(current)
        currentKey = list()

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
