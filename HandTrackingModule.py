import cv2
import mediapipe as mp
import time  # to check the framerates



class handDetector():
    # def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
    #     self.mode = mode 
    #     self.maxHands = maxHands
    #     self.detectionCon = detectionCon
    #     self.trackCon = trackCon
    
        # self.mpHands = mp.solutions.hands
        # self.hands = self.mpHands.Hands(self.mode, self.maxHands, 
        #                                 self.detectionCon, self.trackCon)
        # self.mpDraw = mp.solutions.drawing_utils
        
        
    def __init__(self, detectionCon = 0.5):
        # Hand detection
        self.detectionCon = detectionCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()  # Using default parameters
        self.mpDraw = mp.solutions.drawing_utils
        
        self.tipIds = [4, 8, 12, 16, 20]


    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # show points
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks) # discovering hands
        
        # if hands present then how many 1, 2, ....... ?
        if self.results.multi_hand_landmarks:
            # handling each hand at a time
            for handLms in self.results.multi_hand_landmarks:
                # draw hand connections
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, 
                                               self.mpHands.HAND_CONNECTIONS) 
            
        return img
    
    
    # find location of each landmark
    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        
        # if hands detected then find positions
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lmList.append([id, cx, cy])
                if draw:
                    # pointer
                    cv2.circle(img, (cx, cy), 2, (125, 0, 0), cv2.FILLED)
                    
        return self.lmList
    
    def fingersUp(self):
        fingers = []
        
        # for thumb
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # for remaining fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                # print('index finger opened')
                fingers.append(1)
            else:
                # print('index finger closed')
                fingers.append(0)
        
        return fingers

def main():
    # frame rates
    pTime = 0
    cTime = 0
    
    # capture video
    cap = cv2.VideoCapture(0)
    
    detector = handDetector()
    
    # Running webcam
    while True: 
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[8])
        
        # fps calculation
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        
        # show fps on screen
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            
        cv2.imshow("Image", img)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
    
if __name__ == "__main__":
    main()

