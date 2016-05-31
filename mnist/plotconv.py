import pylab
from PIL import Image
import heapq
import tensorflow as tf
import numpy as np
from numpy import array

def get_top(mat, labels, class_set, num_top):
    ret = []
    sum_class = [[] for _ in range(len(class_set))]
    for i in range(mat.shape[0]):
        mat_sum = tf.reduce_sum(mat[i]).eval()
        sum_class[np.argmax(labels[i])].append((mat_sum, i))
    for c in class_set:
        largestVal = heapq.nlargest(num_top, sum_class[c])
        ret.append([x[1] for x in largestVal])
    return ret

def plot(imgs, imgs_f, labels, class_set, fname):
    """
    data are 3-d matrix
    """
    n = len(class_set)
    m = 3
    top_idx = get_top(imgs_f, labels, class_set, m)
    pylab.gray();
    for i in range(n):
        for j in range(m):
            if j >= len(top_idx[i]):
                continue
            # print top_idx[i][j]
            sig_imgs = tf.sigmoid(imgs[top_idx[i][j]][:, :, 0]).eval()
            sig_imgs_f = tf.sigmoid(imgs_f[top_idx[i][j]][:, :, 0]).eval()
            pylab.subplot2grid((n, m*2), (i, 2*j)); pylab.axis('off'); pylab.imshow(sig_imgs)
            pylab.subplot2grid((n, m*2), (i, 2*j+1)); pylab.axis('off'); pylab.imshow(sig_imgs_f)
    pylab.savefig(fname)

def get_decon(conv, w):
    w_t = tf.transpose(w, [0, 1, 3, 2])
    w_r = tf.reverse(w_t, [True, True, False, False])
    decon = tf.nn.conv2d(conv, w_r, strides=[1, 1, 1, 1], padding='SAME').eval()
    return decon