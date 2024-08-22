import cv2
import numpy as np
import time
import os
import mediapipe as mp

import HandTrackingModule as htm

####################-PARAMETERS-########################
brushThickness = 20
eraserThickness = 60
########################################################

folderPath = "tools/resized"
myList = os.listdir(folderPath)
# print(myList)

# store images
overlayList = []
for imgPath in myList:
    image = cv2.imread(f'{folderPath}/{imgPath}')
    overlayList.append(image)
# print(overlayList)

header = overlayList[0]
drawColor = (153, 0, 153)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    #1. import image
    success, img = cap.read()
    img = cv2.flip(img, 1)
    
    
    #2. find landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    
    if len(lmList) != 0:
        # print(lmList)
        
        # index finger tip
        x1, y1 = lmList[8][1:]
        # middle finger tip
        x2, y2 = lmList[12][1:]
        
        #3. check fingers
        fingers = detector.fingersUp()
        # print(fingers)
        
        #4. selection mode: 2fingers are up
        if fingers[1] and fingers[2]:
            # whenever we do selection set parameters from 0
            xp, yp = 0, 0
            print('selection mode')
            # cv2.rectangle(img, (x1, y1-25), (x2, y2+25), (255, 0, 0), cv2.FILLED)
            # inside header
            if y1 < 125:
                # checking click
                #black brush
                if 250 < x1 < 350:
                    header = overlayList[0]
                    drawColor = (255, 255, 255)
                # red brush
                elif 400 < x1 < 550:
                    header = overlayList[1]
                    drawColor = (0, 0, 255)
                # yellow brush
                elif 650 < x1 < 800:
                    header = overlayList[2]
                    drawColor = (0, 255, 255)
                # eraser
                elif 950 < x1 < 1000:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            
            
        #5. drawing mode: index finger is up
        if fingers[1] and fingers[2] == False:
            print('drawing mode')
            cv2.circle(img, (x1, y1), 12, drawColor, cv2.FILLED)
            cv2.circle(img, (x1, y1), 18, drawColor, 2)
            
            # if just started
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            
            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1
    
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    # converting to binary image
    _, imgInverse = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    
    imgInverse = cv2.cvtColor(imgInverse, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInverse)
    img = cv2.bitwise_or(img, imgCanvas)
    
    # set header
    img[0:125, 0:1280] = header
    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
    
    