import cv2
import numpy as np

cap=cv2.VideoCapture(0)
punkty=[]

while(True):
    ret, frame=cap.read()
    b,g,r=cv2.split(frame)
    maxValue=b.argmax()

    if maxValue>85000:
        i, j = np.unravel_index(b.argmax(), b.shape)
        print(str(j)+" "+str(i))
        cv2.circle(frame,(j,i),15,(0,0,255),2)
        punkty.append((j,i))


    #wrzucNaSiec() -> wchodza punkty wychodziliterkaD
    #pisanie w ramce np w rogu jaka litera

    cv2.putText(img=frame,text="OpenCV",
                org=(10,100),
                fontFace=cv2.FONT_ITALIC,
                fontScale=0.8,
                color=(0,255,0),
                thickness=2,
                bottomLeftOrigin=False)
    for i in range(len(punkty)-1):
        p = punkty[i]
        nastepny = punkty[i+1]
        cv2.line(frame,p,nastepny,(255,0,0),2,8)
    cv2.imshow("RBG", frame)
    if cv2.waitKey(1) & 0xFF == ord('z'):
        break
cap.release()
cv2.destroyAllWindows()

print(punkty)

