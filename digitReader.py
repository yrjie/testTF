import tensorflow as tf
import input_data
from PIL import Image
import os, sys
from numpy import array
import numpy as np

if len(sys.argv)<2:
    print "Usage: bmpFile"
    exit(1)

filename = sys.argv[1]
img = Image.open(filename)
(col, row) = img.size

x0 = []
for i in range(row):
    for j in range(col):
        if img.getpixel((j,i))[0]:
            x0.append(1.0)
        else:
            x0.append(0.0)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# ind = 4
# eps = 1e-8

# dataTr = mnist.train.images[ind].reshape(28,28)
# print mnist.train.labels[ind]

# (colTr, rowTr) = dataTr.shape
# imgTr = Image.new("RGB", (colTr,rowTr), "black") # create a new black image
# pixels = imgTr.load() # create the pixel map

# for i in range(rowTr):    # for every pixel:
#     for j in range(colTr):
#         val = int(dataTr[j][i]*255.0)
#         pixels[i,j] = (val, val, val) # set the colour accordingly
# imgTr.save("data/test1.bmp")

x = tf.placeholder("float", [None, 784])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)

y_ = tf.placeholder("float", [None,10])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

NUM_CORES = 4

with tf.Session() as sess:
    sess.run(init)

    for i in range(1000):
      batch_xs, batch_ys = mnist.train.next_batch(100)
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

    x1 = array(x0).reshape(1, len(x0))
    print np.argmax(sess.run(y, feed_dict={x: x1}))
