---
layout: post
comments: true
mathjax: true
title: “TensorFlow advance topics”
excerpt: “Cover TensorFlow advance topics including Supervisor.”
date: 2017-03-07 14:00:00
---
### Supervisor

Supervisor allows a long running training to be recovered after a crash.  Checkpoint is constantly made.  When the process die, it can reload trained parameters from the checkpoint appoint startup. The code related to supervisor is shown as below:
```python
# For monitor purpose, we need to setup the global_step variable.
global_step = tf.Variable(0, name='global_step', trainable=False)

...
# Define at least one summary to collect.
tf.summary.scalar('loss', loss)

...
train = optimizer.minimize(loss, global_step=global_step)

...

summary_op = tf.summary.merge_all()
...

# Set up a supervisor and use that to start a session.
sv = tf.train.Supervisor(logdir="/tmp/tensorflow/supervisor", summary_op=None)
with sv.managed_session() as sess:
    sess.run(init)
    for step in range(10000):
        # Listen to the supervisor if it want us to stop.
        if sv.should_stop():
            break
        sess.run(train, {x:x_train, y:y_train})

	# For large model, we need to run the summary section manually once a while.
        if step % 1000 == 0:
            _, summ = sess.run([train, summary_op], {x:x_train, y:y_train})
            sv.summary_computed(sess, summ)
        else:
            sess.run(train, {x: x_train, y: y_train})
```

The full source code to train a linear model:
```python
import tensorflow as tf

W = tf.Variable([0.1], tf.float32)
b = tf.Variable([0.0], tf.float32)
global_step = tf.Variable(0, name='global_step', trainable=False)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# Define a linear model y = Wx + b
model = W * x + b

loss = tf.reduce_sum(tf.square(model - y))
tf.summary.scalar('loss', loss)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss, global_step=global_step)

x_train = [1.0, 2.0, 3.0, 4.0]
y_train = [1.5, 3.5, 5.5, 7.5]

summary_op = tf.summary.merge_all()

init = tf.global_variables_initializer()

sv = tf.train.Supervisor(logdir="/tmp/tensorflow/supervisor", summary_op=None)
with sv.managed_session() as sess:
    sess.run(init)
    for step in range(1000):
        if sv.should_stop():
            break
        sess.run(train, {x:x_train, y:y_train})

        if step % 100 == 0:
            _, summ = sess.run([train, summary_op], {x:x_train, y:y_train})
            sv.summary_computed(sess, summ)
        else:
            sess.run(train, {x: x_train, y: y_train})

    l_W, l_b, l_cost  = sess.run([W, b, loss], {x:x_train, y:y_train})
    print(f"W: {l_W} b: {l_b} cost: {l_cost}")
    # W: [ 1.99999797] b: [-0.49999401] cost: 2.2751578399038408e-11
```

### Pre-train data
To load pre-train data into a model. The following show the changes added.
```python
logdir = "/tmp/tensorflow/supervisor"

def load_pretrain(sess):
    pre_train_saver.restore(sess, "logdir")

...
# Load the trainable parameters W and b
pre_train_saver = tf.train.Saver([W, b])

...
# Call load_retain when session starts
sv = tf.train.Supervisor(logdir=logdir, summary_op=None, init_fn=load_retrain)
...
```
Here is the full code for the linear model.
```python
import tensorflow as tf

logdir = "/tmp/tensorflow/supervisor"

def load_pretrain(sess):
    pre_train_saver.restore(sess, "logdir")

W = tf.Variable([0.1], tf.float32)
b = tf.Variable([0.0], tf.float32)
global_step = tf.Variable(0, name='global_step', trainable=False)

pre_train_saver = tf.train.Saver([W, b])

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# Define a linear model y = Wx + b
model = W * x + b

loss = tf.reduce_sum(tf.square(model - y))
tf.summary.scalar('loss', loss)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss, global_step=global_step)

x_train = [1.0, 2.0, 3.0, 4.0]
y_train = [1.5, 3.5, 5.5, 7.5]

summary_op = tf.summary.merge_all()

init = tf.global_variables_initializer()

sv = tf.train.Supervisor(logdir=logdir, summary_op=None, init_fn=load_pretrain)
with sv.managed_session() as sess:
    sess.run(init)
    for step in range(1000):
        if sv.should_stop():
            break
        sess.run(train, {x:x_train, y:y_train})

        if step % 100 == 0:
            _, summ = sess.run([train, summary_op], {x:x_train, y:y_train})
            sv.summary_computed(sess, summ)
        else:
            sess.run(train, {x: x_train, y: y_train})

    l_W, l_b, l_cost  = sess.run([W, b, loss], {x:x_train, y:y_train})
    print(f"W: {l_W} b: {l_b} cost: {l_cost}")
    # W: [ 1.99999797] b: [-0.49999401] cost: 2.2751578399038408e-11
```









