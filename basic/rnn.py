
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np
import random

def make_mini_batch(train_data, size_of_mini_batch, length_of_sequences):
    inputs  = []
    outputs = []
    for i in range(size_of_mini_batch):
        index  = random.randint(0, len(train_data) - length_of_sequences)
        part   = np.matrix(train_data[index:index + length_of_sequences])
        input  = np.array(part[:, 0])
        output = [part[-1, 1]]
        inputs.append(input)
        outputs.append(output)
    return (np.array(inputs), np.array(outputs))

def make_prediction_initial(train_data, length_of_sequences):
    part = np.matrix(train_data[0:length_of_sequences])
    return np.array(part[:, 0])

num_of_input_nodes  = 1
num_of_hidden_nodes = 2
num_of_output_nodes = 1
length_of_sequences = 50
# num_of_steps        = 5000
num_of_steps        = 1000
size_of_mini_batch  = 100
learning_rate       = 0.5
print("num_of_input_nodes  = %d" % num_of_input_nodes)
print("num_of_hidden_nodes = %d" % num_of_hidden_nodes)
print("num_of_output_nodes = %d" % num_of_output_nodes)
print("length_of_sequences = %d" % length_of_sequences)
print("num_of_steps        = %d" % num_of_steps)
print("size_of_mini_batch  = %d" % size_of_mini_batch)
print("learning_rate       = %f" % learning_rate)

train_data = np.load("train_data.npy")
print(train_data)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

with tf.Graph().as_default():
    input_ph      = tf.placeholder(tf.float32, [None, length_of_sequences, num_of_input_nodes], name="input")
    supervisor_ph = tf.placeholder(tf.float32, [None, num_of_output_nodes], name="supervisor")
    istate_ph     = tf.placeholder(tf.float32, [None, num_of_hidden_nodes * 2], name="istate") # 1セルあたり2つの値を必要とする。

    with tf.name_scope("inference") as scope:
        weight1_var = tf.Variable(tf.truncated_normal([num_of_input_nodes, num_of_hidden_nodes], stddev=0.1), name="weight1")
        weight2_var = tf.Variable(tf.truncated_normal([num_of_hidden_nodes, num_of_output_nodes], stddev=0.1), name="weight2")
        bias1_var   = tf.Variable(tf.truncated_normal([num_of_hidden_nodes], stddev=0.1), name="bias1")
        bias2_var   = tf.Variable(tf.truncated_normal([num_of_output_nodes], stddev=0.1), name="bias2")

        in1  = tf.transpose(input_ph, [1, 0, 2])         # (batch, sequence, data) -> (sequence, batch, data)
        in2  = tf.reshape(in1, [-1, num_of_input_nodes]) # (sequence, batch, data) -> (sequence * batch, data)
        in3  = tf.matmul(in2, weight1_var) + bias1_var
        in4  = tf.split(0, length_of_sequences, in3)     # sequence * (batch, data)

        cell = rnn_cell.BasicLSTMCell(num_of_hidden_nodes, forget_bias=1.0)
        rnn_output, states_op = rnn.rnn(cell, in4, initial_state=istate_ph)
        output_op = tf.matmul(rnn_output[-1], weight2_var) + bias2_var

    with tf.name_scope("loss") as scope:
        square_error = tf.reduce_mean(tf.square(output_op - supervisor_ph))
        loss_op      = square_error
        tf.scalar_summary("loss", loss_op)

    with tf.name_scope("training") as scope:
        training_op = optimizer.minimize(loss_op)

    summary_op = tf.merge_all_summaries()
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        saver = tf.train.Saver()
        summary_writer = tf.train.SummaryWriter("data", graph_def=sess.graph_def)
        sess.run(init)

        for i in range(num_of_steps):
            inputs, outputs = make_mini_batch(train_data, size_of_mini_batch, length_of_sequences)

            train_dict = {
                input_ph:      inputs,
                supervisor_ph: outputs,
                istate_ph:     np.zeros((size_of_mini_batch, num_of_hidden_nodes * 2)),
            }
            sess.run(training_op, feed_dict=train_dict)

            if (i + 1) % 10 == 0:
                summary_str, train_loss = sess.run([summary_op, loss_op], feed_dict=train_dict)
                summary_writer.add_summary(summary_str, i)
                print("step#%d, train loss: %e" % (i + 1, train_loss))

        inputs  = make_prediction_initial(train_data, length_of_sequences)
        outputs = np.empty((0, 1))
        states  = np.zeros((num_of_hidden_nodes * 2)),

        for i in range(100):
            pred_dict = {
                input_ph:  np.array([inputs]),
                istate_ph: states,
            }

            output, states = sess.run([output_op, states_op], feed_dict=pred_dict)
            print("pred#%d, output: %f" % (i + 1, output))

            inputs  = np.delete(inputs, 0)       # 先頭の要素を削除
            inputs  = np.append(inputs, output).reshape((-1, 1))  # 末尾に要素を追加
            outputs = np.append(outputs, output) # 末尾に要素を追加

        print("outputs:", outputs)
        np.save("output.npy", np.array(outputs))

        saver.save(sess, "data/out")
