import tensorflow as tf
import math

sess = tf.Session()
sess.run(tf.initialize_all_variables())

hidden1_units = 128
hidden2_units = 32
norm = tf.truncated_normal([hidden1_units, hidden2_units], stddev=1.0 / math.sqrt(float(hidden1_units)))

print(sess.run(norm))
