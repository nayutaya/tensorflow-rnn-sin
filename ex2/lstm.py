
import sys
import yaml # sudo pip3 install pyyaml
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np
import random


def make_mini_batch(train_data, size_of_mini_batch, length_of_sequences):
    inputs  = np.empty(0)
    outputs = np.empty(0)
    for _ in range(size_of_mini_batch):
        index   = random.randint(0, len(train_data) - length_of_sequences)
        part    = train_data[index:index + length_of_sequences]
        inputs  = np.append(inputs, part[:, 0])
        outputs = np.append(outputs, part[-1, 1])
    inputs  = inputs.reshape(-1, length_of_sequences, 1)
    outputs = outputs.reshape(-1, 1)
    return (inputs, outputs)


if len(sys.argv) <= 1:
    print("Usage: " + sys.argv[0] + " param", file=sys.stderr)
    exit(1)

param_path = sys.argv[1]
print("param_path:", param_path)

with open(param_path, "r") as file:
    param = yaml.load(file.read())

for key in param.keys():
    print("%s:" % key, param[key])

seed                     = param["seed"]
train_data_path          = param["train_data_path"]
num_of_input_nodes       = param["num_of_input_nodes"]
num_of_hidden_nodes      = param["num_of_hidden_nodes"]
num_of_output_nodes      = param["num_of_output_nodes"]
length_of_sequences      = param["length_of_sequences"]
num_of_training_epochs   = param["num_of_training_epochs"]
num_of_prediction_epochs = param["num_of_prediction_epochs"]
size_of_mini_batch       = param["size_of_mini_batch"]
learning_rate            = param["learning_rate"]
forget_bias              = param["forget_bias"]

if param["optimizer"] == "GradientDescentOptimizer":
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
elif param["optimizer"] == "AdamOptimizer":
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
else:
    raise Exception("unknown optimizer -- %s" % param["optimizer"])

random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

train_data = np.load(train_data_path)
print("train_data:", train_data)

with tf.Graph().as_default():
    input_ph      = tf.placeholder(tf.float32, [None, length_of_sequences, num_of_input_nodes], name="input")
    supervisor_ph = tf.placeholder(tf.float32, [None, num_of_output_nodes], name="supervisor")
    istate_ph     = tf.placeholder(tf.float32, [None, num_of_hidden_nodes * 2], name="istate") # 1セルあたり2つの値を必要とする。

    with tf.name_scope("inference") as scope:
        weight1_var = tf.Variable(tf.truncated_normal([num_of_input_nodes, num_of_hidden_nodes], stddev=0.1), name="weight1")
        weight2_var = tf.Variable(tf.truncated_normal([num_of_hidden_nodes, num_of_output_nodes], stddev=0.1), name="weight2")
        bias1_var   = tf.Variable(tf.truncated_normal([num_of_hidden_nodes], stddev=0.1), name="bias1")
        bias2_var   = tf.Variable(tf.truncated_normal([num_of_output_nodes], stddev=0.1), name="bias2")

        in1 = tf.transpose(input_ph, [1, 0, 2])         # (batch, sequence, data) -> (sequence, batch, data)
        in2 = tf.reshape(in1, [-1, num_of_input_nodes]) # (sequence, batch, data) -> (sequence * batch, data)
        in3 = tf.matmul(in2, weight1_var) + bias1_var
        in4 = tf.split(0, length_of_sequences, in3)     # sequence * (batch, data)

        cell = rnn_cell.BasicLSTMCell(num_of_hidden_nodes, forget_bias=forget_bias)
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
        summary_writer = tf.train.SummaryWriter("data", graph=sess.graph)
        sess.run(init)

        losses = np.empty((0, 2))

        for epoch in range(num_of_training_epochs):
            inputs, supervisors = make_mini_batch(train_data, size_of_mini_batch, length_of_sequences)

            train_dict = {
                input_ph:      inputs,
                supervisor_ph: supervisors,
                istate_ph:     np.zeros((size_of_mini_batch, num_of_hidden_nodes * 2)),
            }
            sess.run(training_op, feed_dict=train_dict)

            if (epoch + 1) % 10 == 0:
                summary_str, train_loss = sess.run([summary_op, loss_op], feed_dict=train_dict)
                summary_writer.add_summary(summary_str, epoch)
                losses = np.append(losses, [[epoch + 1, train_loss]], axis=0)
                print("train#%d, train loss: %e" % (epoch + 1, train_loss))

        print("losses:", losses)
        np.save("losses.npy", losses)

        inputs  = train_data[0:length_of_sequences, 0]
        outputs = np.empty(0)
        states  = np.zeros((num_of_hidden_nodes * 2)),

        print("initial:", inputs)
        np.save("initial.npy", inputs)

        for epoch in range(num_of_prediction_epochs):
            pred_dict = {
                input_ph:  inputs.reshape((1, length_of_sequences, 1)),
                istate_ph: states,
            }
            output, states = sess.run([output_op, states_op], feed_dict=pred_dict)
            print("prediction#%d, output: %f" % (epoch + 1, output))

            inputs  = np.delete(inputs, 0)
            inputs  = np.append(inputs, output)
            outputs = np.append(outputs, output)

        print("outputs:", outputs)
        np.save("output.npy", outputs)

        saver.save(sess, "data/model")
