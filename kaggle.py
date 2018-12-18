#Libraries and settings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import cv2
import os

# punkty=[
#     (153,161),
#     (170,138),
#     (187,115),
#     (204,115),
#     (221,115),
#     (238,115),
#     (255,115),
#     (255,138),
#     (272,161),
#     (272,184),
#     (289,184),
#     (272,207),
#     (289,207),
#     (272,230),
#     (289,230),
#     (272,253),
#     (289,253),
#     (272,276),
#     (272,299),
#     (255,299),
#     (238,322),
#     (221,322),
#     (204,345),
#     (187,368),
#     (170,368),
#     (153,391),
#     (136,391),
#     (119,414),
#     (136,414),
#     (153,414),
#     (170,414),
#     (187,414),
#     (204,414),
#     (221,414),
#     (238,414),
#     (255,414),
#     (272,414),
#     (289,414),
#     (306,414),
#     (323,414),
#     (119,437),
#     (136,437),
#     (153,437),
#     (170,437),
#     (187,437),
#     (204,437),
#     (221,437),
#     (238,437),
#     (255,437),
#     (272,437),
#     (289,437),
#     (306,437),
#     (323,437)
#     ]

def zapuscSiec(punkty):
    # punkty = [(136,230),(272,230),(187,529)]
    img = np.zeros((480,640,1))

    for i in range(len(punkty)-1):
        start = punkty[i]
        stop = punkty[i + 1]
        cv2.line(img, start, stop, (255,255,255), 20, 8)

    newImage = cv2.resize(img, (28, 28))

    obrazekDoSieci=np.array([np.ravel(newImage)])

    LEARNING_RATE = 1e-4#jak to zmniejsze i doloze iteracje to uzyskam moj wykres
    # set to 20000 on local environment to get 0.99 accuracy
    TRAINING_ITERATIONS = 10

    DROPOUT = 0.5
    BATCH_SIZE = 50#ile probek jest wrzucane do sieci na raz -HIPERARAMETRY

    # set to 0 to train on all available data
    VALIDATION_SIZE =2000

    # image number to output
    IMAGE_TO_DISPLAY = 10

    # read training data from CSV file
    data = pd.read_csv('train.csv')

    images = data.iloc[:,1:].values
    images = images.astype(np.float)

    # convert from [0:255] => [0.0:1.0]
    images = np.multiply(images, 1.0 / 255.0)
    image_size = images.shape[1]

    # in this case all images are square
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

    # split data into training & validation
    validation_images = images[:VALIDATION_SIZE]
    validation_labels = labels[:VALIDATION_SIZE]

    train_images = images[VALIDATION_SIZE:]
    train_labels = labels[VALIDATION_SIZE:]

    # weight initialization
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # convolution
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # input & output of NN

    # images
    x = tf.placeholder('float', shape=[None, image_size])
    # labels
    y_ = tf.placeholder('float', shape=[None, labels_count])

    # first convolutional layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    # (40000,784) => (40000,28,28,1)
    image = tf.reshape(x, [-1,image_width , image_height,1])

    h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)
    #print (h_conv1.get_shape()) # => (40000, 28, 28, 32)
    h_pool1 = max_pool_2x2(h_conv1)
    #print (h_pool1.get_shape()) # => (40000, 14, 14, 32)


    # Prepare for visualization
    # display 32 fetures in 4 by 8 grid
    layer1 = tf.reshape(h_conv1, (-1, image_height, image_width, 4 ,8))

    # reorder so the channels are in the first dimension, x and y follow.
    layer1 = tf.transpose(layer1, (0, 3, 1, 4,2))

    layer1 = tf.reshape(layer1, (-1, image_height*4, image_width*8))

    # second convolutional layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    # print (h_conv2.get_shape()) # => (40000, 14,14, 64)
    h_pool2 = max_pool_2x2(h_conv2)
    # print (h_pool2.get_shape()) # => (40000, 7, 7, 64)

    # Prepare for visualization
    # display 64 fetures in 4 by 16 grid
    layer2 = tf.reshape(h_conv2, (-1, 14, 14, 4 ,16))

    # reorder so the channels are in the first dimension, x and y follow.
    layer2 = tf.transpose(layer2, (0, 3, 1, 4,2))

    layer2 = tf.reshape(layer2, (-1, 14*4, 14*16))

    # densely connected layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    # (40000, 7, 7, 64) => (40000, 3136)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # print (h_fc1.get_shape()) # => (40000, 1024)

    # dropout
    keep_prob = tf.placeholder('float')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # readout layer for deep net
    W_fc2 = weight_variable([1024, labels_count])
    b_fc2 = bias_variable([labels_count])

    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # cost function
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))

    # optimisation function
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

    # evaluation
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    # prediction function
    #[0.1, 0.9, 0.2, 0.1, 0.1 0.3, 0.5, 0.1, 0.2, 0.3] => 1
    predict = tf.argmax(y,1)

    #--------------------------------------------------------------------------------------------------------------------

    # Train, validate and predict
    epochs_completed = 0
    index_in_epoch = 0
    num_examples = train_images.shape[0]

    # serve data by batches
    def next_batch(batch_size):
        global train_images
        global train_labels
        global index_in_epoch
        global epochs_completed

        start = index_in_epoch
        index_in_epoch += batch_size

        # when all trainig data have been already used, it is reorder randomly
        if index_in_epoch > num_examples:
            # finished epoch
            epochs_completed += 1
            # shuffle the data
            perm = np.arange(num_examples)
            np.random.shuffle(perm)
            train_images = train_images[perm]
            train_labels = train_labels[perm]
            # start next epoch
            start = 0
            index_in_epoch = batch_size
            assert batch_size <= num_examples
        end = index_in_epoch
        return train_images[start:end], train_labels[start:end]

    # start TensorFlow session
    # init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()

    sess.run(init)

    # visualisation variables
    train_accuracies = []
    validation_accuracies = []
    x_range = []

    display_step = 1

    saver = tf.train.Saver()
    # jak bede probowac to mam wstawic True
    if len(os.listdir('modele'))==0:
        for i in range(TRAINING_ITERATIONS):

            # get new batch
            batch_xs, batch_ys = next_batch(BATCH_SIZE)

            # check progress on every 1st,2nd,...,10th,20th,...,100th... step
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

                # increase display_step
                if i % (display_step * 10) == 0 and i:
                    display_step *= 10
            # train on batch
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: DROPOUT})

        saver.save(sess,'modele/myModel')

    else:
        saver.restore(sess, 'modele/myModel')
    # check final accuracy on validation set
    if(VALIDATION_SIZE):
        validation_accuracy = accuracy.eval(feed_dict={x: validation_images,
                                                       y_: validation_labels,
                                                       keep_prob: 1.0})
        print('validation_accuracy => %.4f'%validation_accuracy)
        # plt.figure()
        plt.plot(x_range, train_accuracies,'-b', label='Training')
        plt.plot(x_range, validation_accuracies,'-g', label='Validation')
        plt.legend(loc='lower right', frameon=False)
        # plt.ylim(bottom = 1.1, top = 0.7)
        plt.ylabel('accuracy')
        plt.xlabel('step')
        plt.grid()
        plt.show()

    # convert from [0:255] => [0.0:1.0]
    test_images = np.multiply(obrazekDoSieci, 1.0 / 255.0)

    # predict test set
    predicted_lables = predict.eval(feed_dict={x: test_images, keep_prob: 1.0})
    print("wynik:" + str(predicted_lables))

    # odkomentowac dla innych danych (z pliku test.csv)
    # using batches is more resource efficient
    # predicted_lables = np.zeros(test_images.shape[0])
    # for i in range(0,test_images.shape[0]//BATCH_SIZE):
    #     predicted_lables[i*BATCH_SIZE : (i+1)*BATCH_SIZE] = predict.eval(feed_dict={x: test_images[i*BATCH_SIZE : (i+1)*BATCH_SIZE],
    #                                                                                 keep_prob: 1.0})


    # print('predicted_lables({0})'.format(len(predicted_lables)))

    # output test image and prediction
    # display(test_images[IMAGE_TO_DISPLAY])
    # print ('predicted_lables[{0}] => {1}'.format(IMAGE_TO_DISPLAY,predicted_lables[IMAGE_TO_DISPLAY]))

    # save results
    np.savetxt('submission_softmax.csv',
               np.c_[range(1,len(test_images)+1),predicted_lables],
               delimiter=',',
               header = 'ImageId,Label',
               comments = '',
               fmt='%d')

    sess.close()