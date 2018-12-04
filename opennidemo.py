from openni import openni2
import cv2
import numpy as np
openni2.initialize()     # can also accept the path of the OpenNI redistribution

dev = openni2.Device.open_any()
print(dev.get_device_info())

depth_stream = dev.create_depth_stream()
depth_stream.start()
colorStream = dev.create_color_stream()
colorStream.start()

def getFrame(readFrame, isColor):
    frame_data = readFrame.get_buffer_as_uint16()
    img = np.frombuffer(frame_data, dtype=np.uint16)
    img.shape = (1, 480, 640)

    img = np.concatenate((img, img, img), axis=0)
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 0, 1)
    return img

while True:
    img = getFrame(depth_stream.read_frame(), False)
    # rgbFrame = getFrame(colorStream.read_frame(), True)

    cv2.imshow("image", img)
    # cv2.imshow("rgb", rgbFrame)
    if cv2.waitKey(1) & 0xFF == ord('z'):
        break

depth_stream.stop()
colorStream.stop()
openni2.unload()
