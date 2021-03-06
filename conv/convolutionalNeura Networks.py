import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import matplotlib.pyplot as plt
from matplotlib.image import  imread
import numpy as np
import cv2

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB mpyvalue.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

# Importing the required Keras modules containing model and layers
# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x=x_train,y=y_train, epochs=1)

model.evaluate(x_test, y_test)


# image_index = 4444
# plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
# pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
# print("predicted: " + str(pred.argmax()))
# print("actual: " + str(y_test[image_index]))

# 3 kanaly
# obrazek  = imread(r"C:\Users\User\PycharmProjects\FingerTracking\dwojeczkatif.tif")
# obrazek = cv2.split(obrazek)[0]
# obrazek = obrazek.reshape(1,28,28,1)
# # obrazek = obrazek.astype('float32')
# # obrazek /= 255

# 4 kanaly
obrazek = cv2.imread(r'C:\Users\User\PycharmProjects\FingerTracking\dwojeczkatif.tif',0)
obrazek = obrazek.reshape(1,28,28,1)
pred = model.predict(obrazek)
print(pred.argmax())