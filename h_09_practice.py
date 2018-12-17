import tensorflow as tf

x = tf.Variable(3, name='x')
y = tf.Variable(3, name='y')
f = x*x*y + y + 2

sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)

result = sess.run(f)
sess.close()
print(result)



with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()

print(result)


import tensorflow as tf
x = tf.Variable(3, name = 'x')
y = tf.Variable(3, name = 'y')
f = x*x*y + y + 2

sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
sess.close()
print(result)

with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()
print(result)


# 7 ===========================

init = tf.global_variables_initializer()  # prepare an init node

with tf.Session() as sess:
    init.run()
    result = f.eval()
print('7 = ', result)

# 8 ===========================
sess = tf.InteractiveSession()
init.run()
result = f.eval()
print(result)
sess.close()
print('8 = ', result)
# ===========================

x1 = tf.Variable(1)
print(x1.graph is tf.get_default_graph())

graph = tf.Graph()
with graph.as_default():
    x2 = tf.Variable(2)

print(x2.graph is graph)
print(x2.graph is tf.get_default_graph)

tf.reset_default_graph()

# ===========================

w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

with tf.Session() as sess:
    print(y.eval())
    print(z.eval())

with tf.Session() as sess:
    y_eval, z_eval = sess.run([y, z])
    print(y.eval())
    print(z.eval())

import numpy as np
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
m,n = housing.data.shape
print(m, n)
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

with tf.Session() as sess:
    theta_value = theta.eval()
print(theta_value)

# ===========================

X = housing_data_plus_bias
y = housing.target.reshape(-1, 1)
#########################    # Compare with pure NumPy

# Compare with Scikit-Learn

# In[16]:
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing.data, housing.target.reshape(-1, 1))

print(np.r_[lin_reg.intercept_.reshape(-1, 1), lin_reg.coef_.T])

print(lin_reg.coef_)
# ===========================

# In[17]:
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

# In[18]:
print(scaled_housing_data_plus_bias.mean(axis=0))
print(scaled_housing_data_plus_bias.mean(axis=1))
print(scaled_housing_data_plus_bias.mean())
print(scaled_housing_data_plus_bias.shape)

# ========= manually computing the gradients==================
tf.reset_default_graph()
n_epochs = 10
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")


gradients = 2/m * tf.matmul(tf.transpose(X), error)
training_op = tf.assign(theta, theta - learning_rate * gradients)


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE = ", mse.eval() )
        sess.run(training_op)
    best_theta = theta.eval()

print("Best theta:", best_theta)


# ========== using autodiff =================
tf.reset_default_graph()
n_epochs = 10
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="X")
theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name = "predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")


# gradients = 2/m * tf.matmul(tf.transpose(X), error)
gradients = tf.gradients(mse, [theta])[0]
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE = ", mse.eval() )
        sess.run(training_op)
    best_theta = theta.eval()

print("Best theta:", best_theta)

# ### Using a `GradientDescentOptimizer`
tf.reset_default_graph()
n_epochs = 10
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n+1, 1], -1, 1, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE = ", mse.eval())
        sess.run(training_op)

    best_theta = theta.eval()
print("Best theta:", best_theta)

# ===========================

tf.reset_default_graph()
A = tf.placeholder(tf.float32, shape=(None, 3))
B = A + 5
with tf.Session() as sess:
    B_val_1 = B.eval(feed_dict={A:[[1,2,3]]})
    B_val_2 = B.eval(feed_dict={A:[[1,2,3], [4,5,6]]})
    # tf.get_default_session().run(B)

print(B_val_1)
print(B_val_2)

# ===========================
import numpy.random as rnd

def fetch_batch(epoch, batch_index, batch_size):
    rnd.seed(epoch*n_batches + batch_index)
    indices = rnd.randint(m, size=batch_size)
    X_batch = scaled_housing_data_plus_bias[indices]
    y_batch = housing.target.reshape(-1, 1)[indices]
    return X_batch, y_batch

# In[25]:

tf.reset_default_graph()

n_epochs = 10
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=(None, n+1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n+1, 1], -1, 1, seed = 42), name="theta")
y_pred=tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op=optimizer.minimize(mse)

init=tf.global_variables_initializer()


# In[26]:
n_epochs = 10
batch_size=100
n_batches=int(np.ceil(m/batch_size))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            feed_dict = {X: X_batch, y:y_batch}
            sess.run(training_op, feed_dict={X: X_batch, y:y_batch})

    best_theta = theta.eval()
print("Best theta: ", best_theta)


# ========= save and restore model==================
tf.reset_default_graph()
X = tf.constant(scaled_housing_data_plus_bias, dtype = tf.float32, name = "X")
y = tf.constant(housing.target.reshape(-1, 1), dtype = tf.float32, name = "y")
theta = tf.Variable(tf.random_uniform([n+1, 1], -1, 1, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op=optimizer.minimize(mse)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE = ", mse)
            save_path = saver.save(sess, "./tmp/my_model.ckpt")
        sess.run(training_op)
    print("Best theta: ", best_theta)
    save_path = saver.save(sess, "/tmp/my_model_final.ckpt")

with tf.Session() as sess:
    saver.restore(sess, "/tmp/my_model_final.ckpt")
    saver = tf.train.Saver({"weights": theta})

### Using TensorBoard

# In[31]: p243

tf.reset_default_graph()

from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}".format(root_logdir, now)

n_epochs = 10
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=(None, n+1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n+1, 1], -1, 1, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
with tf.name_scope('loss') as scope:
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

mse_summary = tf.summary.scalar("MSE", mse)
summary_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m/batch_size))

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y:y_batch})
                step = epoch * n_batches + batch_index
                summary_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
    best_theta = theta.eval()

summary_writer.flush()
summary_writer.close()

print("Best theta: ", best_theta)


### An ugly flat code

tf.reset_default_graph()
n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name = "X")
w1 = tf.Variable(tf.random_normal((n_features, 1)), name="weights1")
w2 = tf.Variable(tf.random_normal((n_features, 1)), name="weights2")

b1 = tf.Variable(0.0, name="bias1")
b2 = tf.Variable(0.0, name="bias2")

linear1 = tf.add(tf.matmul(X, w1), b1, name="linear1")
linear2 = tf.add(tf.matmul(X, w2), b2, name="linear2")

relu1 = tf.maximum(linear1, 0, name='relu1')
relu2 = tf.maximum(linear2, 0, name='relu2')

output = tf.add_n([relu1, relu2], name="output")

# Much better, using function to build the ReLUs:


# In[39]
tf.reset_default_graph()

def relu(X):
    with tf.name_scope("relu"):
        w_shape = int(X.get_shape()[1]), 1  # (3, 1)
        w = tf.Variable(tf.random_normal(w_shape), name="weights")
        b = tf.Variable(0.0, name="bias")
        linear = tf.add(tf.matmul(X, w), b, name="linear")
        return tf.maximum(linear, 0, name="relu")

n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = [relu(X) for i in range(5)]
output=tf.add_n(relus, name="output")
summary_writer = tf.summary.FileWriter("logs/relu1", tf.get_default_graph())
summary_writer = tf.summary.FileWriter("logs/relu2", tf.get_default_graph())

# Even better using name scopes:

# In [40]:

summary_writer.close()



# In [41]: as parameter

tf.reset_default_graph()

def relu(X, threshold):
    with tf.name_scope("relu"):
        w_shape = int(X.get_shape()[1]), 1
        w = tf.Variable(tf.random_normal(w_shape), name="weights")
        b = tf.Variable(0.0, name="bias")
        linear = tf.add(tf.matmul(X, w), b, name="linear")
        return tf.maximum(linear, threshold, name="max")

threshold = tf.Variable(0.0, name="threshold")
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = [relu(X, threshold) for i in range(5)]
output = tf.add_n(relus, name="output")


# In [43]: as attribute

tf.reset_default_graph()

def relu(X):
    with tf.name_scope("relu"):
        if not hasattr(relu, "threshold"):
            relu.threshold = tf.Variable(0.0, name="threshold")
        w_shape = int(X.get_shape()[1]), 1
        w = tf.Variable(tf.random_normal(w_shape))
        b = tf.Variable(0.0, name="bias")
        linear = tf.add(tf.matmul(X, w), b, name="linear")
        return tf.maximum(linear, relu.threshold, name="max")

X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name="output")


# In [44] as get variable

tf.reset_default_graph()

def relu(X):
    with tf.variable_scope("relu", reuse=True):
        threshold = tf.get_variable("threshold", shape=(), initializer=tf.constant_initializer(0.0))
        w_shape = int(X.get_shape()[1]), 1
        w = tf.Variable(tf.random_normal(w_shape), name="weights")
        b = tf.Variable(0.0, name="bias")
        linear = tf.add(tf.matmul(X, w), b, name="linear")
        return tf.maximum(linear, threshold, name="max")

X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
with tf.variable_scope("relu"):
    threshold = tf.get_variable("threshold", shape=(), initializer=tf.constant_initializer(0.0))
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name="output")

summary_writer = tf.summary.FileWriter("logs/relu6", tf.get_default_graph())
summary_writer.close()



# In[46]:
tf.reset_default_graph()

with tf.variable_scope("param"):
    x = tf.get_variable("x", shape=(), initializer=tf.constant_initializer(0.))
    # x = tf.Variable(0., name="x")
with tf.variable_scope("param", reuse=True):
    y = tf.get_variable("x")

with tf.variable_scope("", default_name="", reuse=True):
    z = tf.get_variable("param/x", shape=(), initializer=tf.constant_initializer(0.))

print(x is y)
print(x.op.name)
print(y.op.name)
print(z.op.name)


# ===========================
# ===========================
# ===========================
# ===========================
# ===========================
# ===========================
# ===========================
# ===========================
# ===========================
# ===========================
# ===========================
# ===========================
# ===========================
# ===========================
# ===========================
# ===========================
# ===========================
# ===========================
# ===========================
# ===========================
# ===========================
# ===========================
# ===========================

print(">>>>>>>>>>>>>>>>>>> Done! <<<<<<<<<<<<<<")