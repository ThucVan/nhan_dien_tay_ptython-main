import cv2
import time
import os
import hand


#thoi gian ban dau
pTime = 0
cam = cv2.VideoCapture(0)
FolderPath = "Fingers"
list = os.listdir(FolderPath)
list_2 = []

#id các đầu ngón tay
fingerid=[4,8,12,16,20]
for i in list:
    image_hand = cv2.imread(f"{FolderPath}/{i}")
    list_2.append(image_hand)


print(list)

detector = hand.handDetector(0.55)

while True:
    ret, frame = cam.read()

    frame = detector.findHands(frame)
    lmlist=detector.findPosition(frame,draw=False) #phát hiện vị trí
    fingers = []
    if len(lmlist)!= 0:
        #nhan dien ngon cai (diem 4 nam ben trai hay ben phai diem 3)
        if lmlist[fingerid[0]][1] < lmlist[fingerid[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # nhan dien 4 ngon dai (so sanh cao hay thap)
        for id in range(1,5):
            if lmlist[fingerid[id]][2] < lmlist[fingerid[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

    songontay = fingers.count(1)
    print(songontay)



    h, w, c = list_2[songontay-1].shape

    frame[0:h, 0:w] = list_2[songontay-1]

    #ve hinh chu nhat show so ngon tay
    cv2.rectangle(frame, (0,100), (100,200), (0,255,0), -1)
    cv2.putText(frame, str(songontay), (10, 200), cv2.FONT_HERSHEY_PLAIN, 7, (255,0,0), 5)

    #show fps, cTime : thoi gian hien tai
    cTime= time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    #show fps
    cv2.putText(frame, f"FPS : {int(fps)}", (150,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0 ,0), 3)

    newframe = cv2.resize(frame, (0,0), fy=1.5, fx=1.5)

    cv2.imshow("nhan dien ngon tay", newframe)

    if cv2.waitKey(1)==ord("x"):
        break

#giai phong cam
cam.release()

cv2.destroyAllWindows()
