from openni import openni2
import cv2
import numpy as np
from convolutionalNeuralNetwork import loadArtificialNeuralNetwork

def startApp():
    openni2.initialize()     # can also accept the path of the OpenNI redistribution

    dev = openni2.Device.open_any()
    print(dev.get_device_info())

    depth_stream = dev.create_depth_stream()
    depth_stream.start()
    punkty=[]

    def getFrame(readFrame):
        frame_data = readFrame.get_buffer_as_uint16()
        img = np.frombuffer(frame_data, dtype=np.uint16)
        img.shape = (480, 640)
        return img

    while(True):
        img = getFrame(depth_stream.read_frame())
        img = np.ma.masked_equal(img, 0.0, copy=True) #moze nie bedzie potrzebny
        indexClosestToCamera=img.argmin()


        j, i = np.unravel_index(indexClosestToCamera, img.shape)
        point = (i,j)
        dlX = 350
        dlY = 300
        xStart = 120
        yStart=120

        czyWGranicachPionowych = lambda p: xStart <= p[0] < (xStart + dlX)
        czyWGranicachPoziomych = lambda p: yStart <= p[1] < (yStart + dlY)

        if czyWGranicachPionowych(point) and czyWGranicachPoziomych(point):
            pixelValueNearestToCamera = img.min()
            print(str(j) +" " + str(i) +"->" + str(pixelValueNearestToCamera))
            # if 1700 > pixelValueNearestToCamera > 1200:
            cv2.circle(img,(i,j),30,(0,0,0),5)
            punkty.append((i,j))
            if pixelValueNearestToCamera>1400 and len(punkty)>30:
                result = loadArtificialNeuralNetwork(punkty)
                print(result)
                break
        # cv2.line(img,(xStart, yStart), (xStart+dlX, yStart), (0,0,0), 5)
        # cv2.line(img,(xStart+dlX, yStart), (xStart+dlX, yStart+dlY), (0,0,0), 5)
        # cv2.line(img,(xStart+dlX, yStart+dlY), (xStart, yStart+dlY), (0,0,0), 5)
        # cv2.line(img,(xStart, yStart+dlY), (xStart, yStart), (0,0,0), 5)

        cv2.imshow("Malgorzata Niewiadomska Inzynieria Biomedyczna", img)
        if cv2.waitKey(1) & 0xFF == ord('z'):
            break

    depth_stream.stop()
    openni2.unload()
    print(punkty)
