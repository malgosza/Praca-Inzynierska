from openni import openni2
import cv2
import numpy as np
from convolutionalNeuralNetwork import loadArtificialNeuralNetwork

def fuDoZmiany():
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
        pixelValueNearestToCamera=img.min()
        if pixelValueNearestToCamera<1700 and pixelValueNearestToCamera>1200:
            j,i = np.unravel_index(indexClosestToCamera, img.shape)
            print(str(j) +" " + str(i) +"->" + str(pixelValueNearestToCamera))
            cv2.circle(img,(i,j),30,(0,0,0),5)
            punkty.append((i,j))
        elif(len(punkty)>0):
            result = loadArtificialNeuralNetwork(punkty)
            print(result)
            break
        cv2.imshow("Malgorzata Niewiadomska Inzynieria Biomedyczna", img)
        if cv2.waitKey(1) & 0xFF == ord('z'):
            break

    depth_stream.stop()
    openni2.unload()
    print(punkty)