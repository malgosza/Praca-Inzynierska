#Libraries and settings
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import os
from resultLayout import displayResult

# Based on https://www.kaggle.com/kakauandme/tensorflow-deep-nn
def loadArtificialNeuralNetwork(punkty):
    img = np.zeros((480,640,1))

    for i in range(len(punkty) - 1):
        start = punkty[i]
        stop = punkty[i + 1]
        cv2.line(img, start, stop, (255, 255, 255), 20, 8)

    newImage = cv2.resize(img, (28, 28))

    obrazekDoSieci = np.array([np.ravel(newImage)])

    LEARNING_RATE = 1e-4

    TRAINING_ITERATIONS = 2500

    DROPOUT = 0.25
    BATCH_SIZE = 25

    VALIDATION_SIZE =2000

    data = pd.read_csv('train.csv')

    images = data.iloc[:,1:].values
    images = images.astype(np.float)

    images = np.multiply(images, 1.0 / 255.0)
    image_size = images.shape[1]

    image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)

    # display image
    def display(img):
        # (784) => (28,28)
        one_image = img.reshape(image_width, image_height)

        plt.axis('off')
        plt.imshow(one_image, cmap=cm.binary)
        plt.show()

    labels_flat =data.iloc[:, 0].values.ravel()
    labels_count = np.unique(labels_flat).shape[0]

    def dense_to_one_hot(labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    labels = dense_to_one_hot(labels_flat, labels_count)
    labels = labels.astype(np.uint8)

    validation_images = images[:VALIDATION_SIZE]
    validation_labels = labels[:VALIDATION_SIZE]

    train_images = images[VALIDATION_SIZE:]
    train_labels = labels[VALIDATION_SIZE:]

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    x = tf.placeholder('float', shape=[None, image_size])

    y_ = tf.placeholder('float', shape=[None, labels_count])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    image = tf.reshape(x, [-1,image_width , image_height,1])

    h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    layer1 = tf.reshape(h_conv1, (-1, image_height, image_width, 4 ,8))

    layer1 = tf.transpose(layer1, (0, 3, 1, 4,2))

    layer1 = tf.reshape(layer1, (-1, image_height*4, image_width*8))

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    layer2 = tf.reshape(h_conv2, (-1, 14, 14, 4 ,16))

    layer2 = tf.transpose(layer2, (0, 3, 1, 4,2))

    layer2 = tf.reshape(layer2, (-1, 14*4, 14*16))

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    keep_prob = tf.placeholder('float')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, labels_count])
    b_fc2 = bias_variable([labels_count])

    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = -tf.reduce_sum(y_*tf.log(y))

    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    predict = tf.argmax(y,1)

    epochs_completed = 0
    index_in_epoch = 0
    num_examples = train_images.shape[0]

    def next_batch(batch_size):
        global train_images
        global train_labels
        global index_in_epoch
        global epochs_completed

        start = index_in_epoch
        index_in_epoch += batch_size

        if index_in_epoch > num_examples:
            epochs_completed += 1
            perm = np.arange(num_examples)
            np.random.shuffle(perm)
            train_images = train_images[perm]
            train_labels = train_labels[perm]
            start = 0
            index_in_epoch = batch_size
            assert batch_size <= num_examples
        end = index_in_epoch
        return train_images[start:end], train_labels[start:end]

    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()

    sess.run(init)

    train_accuracies = []
    validation_accuracies = []
    x_range = []

    display_step = 1

    saver = tf.train.Saver()
    if len(os.listdir('modele'))==0:
        for i in range(TRAINING_ITERATIONS):

            batch_xs, batch_ys = next_batch(BATCH_SIZE)

            if i % display_step == 0 or (i + 1) == TRAINING_ITERATIONS:

                train_accuracy = accuracy.eval(feed_dict={x: batch_xs,
                                                          y_: batch_ys,
                                                          keep_prob: 1.0})
                if (VALIDATION_SIZE):
                    validation_accuracy = accuracy.eval(feed_dict={x: validation_images[0:BATCH_SIZE],
                                                                   y_: validation_labels[0:BATCH_SIZE],
                                                                   keep_prob: 1.0})
                    print('training_accuracy / validation_accuracy => %.2f / %.2f for step %d' % (
                    train_accuracy, validation_accuracy, i))

                    validation_accuracies.append(validation_accuracy)

                else:
                    print('training_accuracy => %.4f for step %d' % (train_accuracy, i))
                train_accuracies.append(train_accuracy)
                x_range.append(i)

                if i % (display_step * 10) == 0 and i:
                    display_step *= 10
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: DROPOUT})

        saver.save(sess,'modele/myModel')

    else:
        saver.restore(sess, 'modele/myModel')
    if(VALIDATION_SIZE and len(os.listdir('modele'))==0):
        validation_accuracy = accuracy.eval(feed_dict={x: validation_images,
                                                       y_: validation_labels,
                                                       keep_prob: 1.0})
        print('validation_accuracy => %.4f'%validation_accuracy)
        plt.plot(x_range, train_accuracies,'-b', label='Training')
        plt.plot(x_range, validation_accuracies,'-g', label='Validation')
        plt.legend(loc='lower right', frameon=False)
        plt.ylabel('accuracy')
        plt.xlabel('step')
        plt.grid()
        plt.show()

    test_images = np.multiply(obrazekDoSieci, 1.0 / 255.0)
    test_images=test_images.astype(np.float)
    predicted_lables = predict.eval(feed_dict={x: test_images, keep_prob: 1.0})

    np.savetxt('submission_softmax.csv',
               np.c_[range(1,len(test_images)+1),predicted_lables],
               delimiter=',',
               header = 'ImageId,Label',
               comments = '',
               fmt='%d')

    sess.close()

    displayResult(str(predicted_lables[0]))
