import numpy as np
import cv2
import scipy.misc

punkty=[
(153,161),
(170,138),
(187,115),
(204,115),
(221,115),
(238,115),
(255,115),
(255,138),
(272,161),
(272,184),
(289,184),
(272,207),
(289,207),
(272,230),
(289,230),
(272,253),
(289,253),
(272,276),
(272,299),
(255,299),
(238,322),
(221,322),
(204,345),
(187,368),
(170,368),
(153,391),
(136,391),
(119,414),
(136,414),
(153,414),
(170,414),
(187,414),
(204,414),
(221,414),
(238,414),
(255,414),
(272,414),
(289,414),
(306,414),
(323,414),
(119,437),
(136,437),
(153,437),
(170,437),
(187,437),
(204,437),
(221,437),
(238,437),
(255,437),
(272,437),
(289,437),
(306,437),
(323,437)
]

# punkty = [(136,230),(272,230),(187,529)]
img = np.zeros((480,640,3))
img[:] = 255
for i in range(len(punkty)-1):
    start = punkty[i]
    stop = punkty[i + 1]
    img=cv2.line(img, start, stop, (65,65,65), 20, 8)

newImage = cv2.resize(img, (28, 28))
obrazekDoSieci=newImage.ravel()
np.savetxt("test.csv",obrazekDoSieci,'%s', ',')
print(obrazekDoSieci.shape)


# cv2.imshow("Test",newImage)
# cv2.waitKey(0)

# cv2.imwrite("jedynka.png",newImage,  [cv2.IMWRITE_PNG_COMPRESSION, 9])
#
# print("stop")

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