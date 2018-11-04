import numpy as np
import cv2

punkty = [(136,230),(272,230),(187,529)]
img = np.zeros((480,640,3))
img[:] = 255
for i in range(len(punkty)-1):
    start = punkty[i]
    stop = punkty[i + 1]
    img=cv2.line(img, start, stop, (0,0,0), 20, 8)

newImage = cv2.resize(img, (28, 28))

cv2.imshow("Test",newImage)
cv2.waitKey(0)

cv2.imwrite("jedynka.png",newImage,  [cv2.IMWRITE_PNG_COMPRESSION, 9])
print(newImage)
print(newImage)

#---------DZIALA--------
# import cv2
#
# image=cv2.imread("Kamil.jpg")
# cv2.imshow("Kamil",image)
#
# newImage = cv2.resize(image, (28, 28))
# cv2.imshow("New Image", newImage)
# print(newImage.shape)
# cv2.waitKey(0)