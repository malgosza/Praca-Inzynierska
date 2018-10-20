import cv2
import numpy as np

cap=cv2.VideoCapture(0)
while(True):
    ret, frame=cap.read()
    b,g,r=cv2.split(frame)
    i, j = np.unravel_index(b.argmax(), b.shape)
    # maxIndeks=np.argmax(b)
    print(str(i)+" "+str(j))
    cv2.imshow("RBG", frame)
    if cv2.waitKey(1) & 0xFF == ord('z'):
        break
cap.release()
cv2.destroyAllWindows()

# cap = cv2.VideoCapture(cv2.CAP_INTELPERC_IMAGE)

# while(True):
#     # Capture frame-by-frame
#     # ret,frame=cap.retrieve(cv2.CAP_OPENNI_DEPTH_MAP)
#     ret, frame = cap.read()
#     b, g, r = cv2.split(cap)
#     print(r)
#
#     # Display the resulting frame
#     cv2.imshow('Finger Detection',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()
