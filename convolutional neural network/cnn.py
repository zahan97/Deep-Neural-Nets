import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
batch_size = 100
learning_rate = 0.003
num_iters = 5000
# no. of feature maps
K = 4
L = 8
M = 12
N = 200
pkeep = tf.placeholder(tf.float32)  # droupout

# model creation
x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
Y_ = tf.placeholder(tf.float32, shape=[None, 10])

w1 = tf.Variable(tf.truncated_normal([5, 5, 1, K], stddev=0.1))
b1 = tf.Variable(tf.ones([K]) / 10)

w2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
b2 = tf.Variable(tf.ones([L]) / 10)

w3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
b3 = tf.Variable(tf.ones([M]) / 10)

w4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
b4 = tf.Variable(tf.ones([N]) / 10)

w5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
b5 = tf.Variable(tf.ones([10]) / 10)

y1 = tf.nn.relu(tf.nn.conv2d(x, w1, [1, 1, 1, 1], padding='SAME') + b1)

y2 = tf.nn.relu(tf.nn.conv2d(y1, w2, [1, 2, 2, 1], padding='SAME') + b2)

y3 = tf.nn.relu(tf.nn.conv2d(y2, w3, [1, 2, 2, 1], padding='SAME') + b3)

yy = tf.reshape(y3, shape=[-1, 7 * 7 * M])

y4 = tf.nn.relu(tf.matmul(yy, w4) + b4)

do = tf.nn.dropout(y4, pkeep)

y = tf.nn.softmax(tf.matmul(do, w5) + b5)

# loss function
cross_entropy = -tf.reduce_sum(Y_ * tf.log(y))

is_correct = tf.equal(tf.argmax(Y_, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# training algo.
optimizer = tf.train.AdamOptimizer(learning_rate, 0.9, 0.999, 1e-8)
train_step = optimizer.minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# training
for i in range(num_iters):
    batch_X, batch_Y = mnist.train.next_batch(batch_size)
    img = np.array(batch_X).reshape(-1, 28, 28, 1)
    train_data = {x: img, Y_: batch_Y, pkeep: 0.75}

    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 2000.0
    sess.run(train_step, feed_dict=train_data)

    atr, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)

    #print(i)

print('Accuracy on training set is ', atr)

# testing
np.array(mnist.test.images).reshape(-1, 28, 28, 1)
test_data = {x: np.array(mnist.test.images).reshape(-1, 28, 28, 1), Y_: mnist.test.labels, pkeep: 1.0}
at, ac = sess.run([accuracy, cross_entropy], feed_dict=test_data)

print('Accuracy on test set is', at)

# Accuracy on training set is  0.99
# Accuracy on test set is 0.9892
