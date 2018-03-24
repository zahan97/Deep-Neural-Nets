import tensorflow as tf
from tensorflow.contrib import rnn, layers
import numpy as np
import my_txtutils as tt
import os
import time
import math

tf.set_random_seed(0)

# hyperparameters
seq_len = 30
batch_size = 200
alphabetsize = tt.ALPHASIZE
cell_size = 512
no_layers = 3
learning_rate = 0.001

shakedir = "shakespeare/*.txt"
codetext, valitext, bookranges = tt.read_data_files(shakedir, validation=False)

# display some stats on the data
epoch_size = len(codetext) // (batch_size * seq_len)
tt.print_data_stats(len(codetext), len(valitext), epoch_size)

lr = tf.placeholder(tf.float32)  # learning rate
batchsize = tf.placeholder(tf.int32)

# inputs
Xd = tf.placeholder(tf.uint8, [None, None])  # [ batch_size, seq_len ]
X = tf.one_hot(Xd, alphabetsize, 1.0, 0.0)  # [ batch_size, seq_len, alphabetsize ]
Yd = tf.placeholder(tf.uint8, [None, None])  # [ batch_size, seq_len ]
Y_ = tf.one_hot(Yd, alphabetsize, 1.0, 0.0)  # [ batch_size, seq_len, alphabetsize ]
Hin = tf.placeholder(tf.float32, [None, cell_size * no_layers])  # [ batch_size, cell_size * no_layers]
# expected outputs = same sequence shifted by 1 since we are trying to predict the next character

# model
cell = [tf.nn.rnn_cell.GRUCell(cell_size) for _ in range(no_layers)]
mcell = tf.nn.rnn_cell.MultiRNNCell(cell, state_is_tuple=False)

Hr, H = tf.nn.dynamic_rnn(mcell, X, initial_state=Hin)
# Hr: [ batch_size, seq_len, cell_size ]
# H:  [ batch_size, cell_size*no_layers ]

# softmax output
Yflat = tf.reshape(Hr, [-1, cell_size])  # [ batch_size * seq_len, cell_size ]
Ylogits = layers.linear(Yflat, alphabetsize)  # [ batch_size * seq_len, alphabetsize ]
Y = tf.nn.softmax(Ylogits)

predictions = tf.argmax(Y, 1)  # [ batch_size * seq_len]
predictions = tf.reshape(predictions, [batch_size, -1])  # [ batch_size, seq_len ]

# training
loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

DISPLAY_FREQ = 50
_50_BATCHES = DISPLAY_FREQ * batch_size * seq_len

# initializing
initial_H = np.zeros([batch_size, cell_size * no_layers])
step = 0

for x, y_, epoch in tt.rnn_minibatch_sequencer(codetext, batch_size, seq_len, nb_epochs=10):

    # train on one mini_batch
    feed_dict = {Xd: x, Yd: y_, Hin: initial_H, lr: learning_rate, batchsize: batch_size}
    _, y, ostate = sess.run([train_step, predictions, H], feed_dict=feed_dict)

    if step // 3 % _50_BATCHES == 0:
        tt.print_text_generation_header()

        ry = np.array([[tt.convert_from_alphabet(ord("K"))]])
        rh = np.zeros([1, cell_size * no_layers])

        for k in range(1000):
            ryo, rh = sess.run([Y, H], feed_dict={Xd: ry, Hin: rh, batchsize: 1})
            rc = tt.sample_from_probabilities(ryo, topn=10 if epoch <= 1 else 2)
            print(chr(tt.convert_to_alphabet(rc)), end="")
            ry = np.array([[rc]])

        tt.print_text_generation_footer()

    initial_H = ostate
    step += batch_size * seq_len
