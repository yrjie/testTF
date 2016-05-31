import tensorflow as tf
from numpy import array

x = tf.placeholder("float", shape=[1, 3, 4, 1])
W = tf.placeholder("float", shape=[3, 3, 1, 2])
# x = tf.placeholder("float", shape=[1, 3, 4, 2])
# W = tf.placeholder("float", shape=[3, 3, 2, 1])

# W is flattened to [filter_height * filter_width * in_channels, output_channels]
# output[b, i, j, k] =
#     sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *
#                     filter[di, dj, q, k]
cv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

x1 = array([1, 5, 2, 3, 8, 7, 3, 6, 3, 3, 9, 1]).reshape([1, 3, 4, 1])
x2 = array([1, 1, 5, 5, 2, 2, 3, 3, 8, 8, 7, 7, 3, 3, 6, 6, 3, 3, 3, 3, 9, 9, 1, 1]).reshape([1, 3, 4, 2])
print x2
# W1 = array([1, 2, 3, 0, 0, 0, 6, 5, 4]).reshape([3, 3, 1, 1])
W1 = array([4, 5, 6, 0, 0, 0, 3, 2, 1]).reshape([3, 3, 1, 1])
W2 = array([4, 4, 5, 5, 6, 6, 0, 0, 0, 0, 0, 0, 3, 3, 2, 2, 1, 1]).reshape([3, 3, 2, 1])
W3 = array([4, 4, 5, 5, 6, 6, 0, 0, 0, 0, 0, 0, 3, 3, 2, 2, 1, 1]).reshape([3, 3, 1, 2])
# print W2

pool = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME")

with tf.Session() as sess:
    # result = cv.eval(feed_dict={x:x2, W:W2})
    result = cv.eval(feed_dict={x:x1, W:W3})
    print result
    # print pool.eval(feed_dict={x:x2})
    # W1r = W1[::-1,::-1,:,:]
    # print W1r
    # resultr = cv.eval(feed_dict={x:result, W:W1r})
    # print resultr
