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

    cv2.imshow("RBG", frame)
    #wrzucNaSiec() -> wchodza punkty wychodziliterka
    #pisanie w ramce np w rogu jaka litera

    cv2.putText(img=frame,text="OpenCV",
                org=(200,200),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2,
                color=(0,0,0),
                thickness=,
                bottomLeftOrigin=False)
    if cv2.waitKey(1) & 0xFF == ord('z'):
        break
cap.release()
cv2.destroyAllWindows()

print(punkty)

