import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
batch_size = 100
learning_rate = 0.003
num_iters = 3000
K = 200
L = 100
M = 60
N = 30
pkeep = tf.placeholder(tf.float32)

# model creation
x = tf.placeholder(tf.float32, shape=[None, 784])

w1 = tf.Variable(tf.truncated_normal([784, K], stddev=0.1))
b1 = tf.Variable(tf.zeros([K]))

w2 = tf.Variable(tf.truncated_normal([K, L], stddev=0.1))
b2 = tf.Variable(tf.zeros([L]))

w3 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
b3 = tf.Variable(tf.zeros([M]))

w4 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
b4 = tf.Variable(tf.zeros([N]))

w5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
b5 = tf.Variable(tf.zeros([10]))

init = tf.global_variables_initializer()

Y_ = tf.placeholder(tf.float32, shape=[None, 10])

y1 = tf.nn.relu(tf.matmul(x, w1) + b1)
do1 = tf.nn.dropout(y1, pkeep)

y2 = tf.nn.relu(tf.matmul(do1, w2) + b2)
do2 = tf.nn.dropout(y2, pkeep)

y3 = tf.nn.relu(tf.matmul(do2, w3) + b3)
do3 = tf.nn.dropout(y3, pkeep)

y4 = tf.nn.relu(tf.matmul(do3, w4) + b4)

y = tf.nn.softmax(tf.matmul(y4, w5) + b5)

# loss function
cross_entropy = -tf.reduce_sum(Y_ * tf.log(y))

is_correct = tf.equal(tf.argmax(Y_, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# training algo.
optimizer = tf.train.AdamOptimizer(learning_rate)
train_step = optimizer.minimize(cross_entropy)

sess = tf.Session()
sess.run(init)

# training
for i in range(num_iters):
    batch_X, batch_Y = mnist.train.next_batch(batch_size)
    train_data = {x: batch_X, Y_: batch_Y, pkeep: 0.75}

    sess.run(train_step, feed_dict=train_data)

    atr, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)

print('Accuracy on training set is ', atr)

# testing
test_data = {x: mnist.test.images, Y_: mnist.test.labels, pkeep: 1.0}
at, ac = sess.run([accuracy, cross_entropy], feed_dict=test_data)

print('Accuracy on test set is', at)

# Accuracy on training set is  0.97
# Accuracy on test set is 0.981
