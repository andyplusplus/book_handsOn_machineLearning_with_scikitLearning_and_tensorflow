
n_epochs_4_test = 1

######################################### Setup ######################

# Common imports
import numpy as np
import numpy.random as rnd
import os

# to make this notebook's output stable across runs
rnd.seed(42)

# To plot pretty figures

import matplotlib
import matplotlib.pyplot as plt
from my_utility import show_plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "tensorflow"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


######################################### Creating and running a graph ######################

import tensorflow as tf

tf.reset_default_graph()

x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2
sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
print(sess.run(f))
sess.close()



with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()
print(result)



#6
init = tf.global_variables_initializer()
with tf.Session():
    init.run()
    result = f.eval()
print(result)



#7
init = tf.global_variables_initializer()

sess = tf.InteractiveSession()
init.run()
result = f.eval()
sess.close()
print(result)


######################################### Managing graphs ######################

#8  Manage Graphs

# In[8]:
tf.reset_default_graph()

x1 = tf.Variable(1)
print(x1.graph is tf.get_default_graph()) # true

# In[9]:
graph = tf.Graph()
with graph.as_default():
    x2 = tf.Variable(2)

print(x2.graph is tf.get_default_graph()) # false

# x2.graph is graph #true

# In[11]:
w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

with tf.Session() as sess:
    print(y.eval())  # 10
    print(z.eval())  # 15

# In[12]:
with tf.Session() as sess:
    y_val, z_val = sess.run([y, z])
    print(y)  # 10
    print(z)  # 15




##################### # Linear Regression

# ## Using the Normal Equation

# In[13]:
if True:
    from sklearn.datasets import fetch_california_housing

    housing = fetch_california_housing()
    m, n = housing.data.shape
    housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

    # In[14]:
    tf.reset_default_graph()

    X = tf.constant(housing_data_plus_bias, dtype=tf.float64, name="X")
    y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float64, name="y")
    XT = tf.transpose(X)
    theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

    with tf.Session() as sess:
        result = theta.eval()

    print(result)

#########################    # Compare with pure NumPy

    # In[15]:
    X = housing_data_plus_bias
    y = housing.target.reshape(-1, 1)
    theta_numpy = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    print(theta_numpy)

# Compare with Scikit-Learn

# In[16]:
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing.data, housing.target.reshape(-1, 1))

print(np.r_[lin_reg.intercept_.reshape(-1, 1), lin_reg.coef_.T])

################################## ## Using Batch Gradient Descent

# Gradient Descent requires scaling the feature vectors first. We could do this using TF, but let's just use Scikit-Learn for now.

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

# ### compute the gradients

# In[19]:
if True:
    tf.reset_default_graph()

    n_epochs = 1000
    learning_rate = 0.01

    X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
    y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")


    is_manually, is_audodiff, is_gradient_descent_optimizer, is_momentum_optimizer = False, False, True, False
    if is_manually:
        gradients = 2 / m * tf.matmul(tf.transpose(X), error)
        training_op = tf.assign(theta, theta - learning_rate * gradients)
    if is_audodiff:
        gradients = tf.gradients(mse, [theta])[0]
        training_op = tf.assign(theta, theta - learning_rate * gradients)
    elif is_gradient_descent_optimizer:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        training_op = optimizer.minimize(mse)
    elif is_momentum_optimizer:
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.25)
        training_op = optimizer.minimize(mse)


    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(n_epochs):
            if epoch % 100 == 0:
                print("Epoch", epoch, "MSE =", mse.eval())
            sess.run(training_op)

        best_theta = theta.eval()

    print("Best theta:")
    print(best_theta)

    # ### Using autodiff
    # Same as above except for the `gradients = ...` line.

# # Feeding data to the training algorithm

# ## Placeholder nodes

# In[24]:
def fetch_batch(epoch, batch_index, batch_size):
    rnd.seed(epoch * n_batches + batch_index)
    indices = rnd.randint(m, size=batch_size)
    X_batch = scaled_housing_data_plus_bias[indices]
    y_batch = housing.target.reshape(-1, 1)[indices]
    return X_batch, y_batch



if True:
    tf.reset_default_graph()

    A = tf.placeholder(tf.float32, shape=(None, 3))
    B = A + 5
    with tf.Session() as sess:
        B_val_1 = B.eval(feed_dict={A: [[1, 2, 3]]})
        B_val_2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8, 9]]})

    print(B_val_1)
    print(B_val_2)

    # ## Mini-batch Gradient Descent

    # In[25]:
    tf.reset_default_graph()

    n_epochs = 1000
    learning_rate = 0.01

    X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
    y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(mse)

    init = tf.global_variables_initializer()


    # In[26]:

    n_epochs = 10
    batch_size = 100
    n_batches = int(np.ceil(m / batch_size))

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        best_theta = theta.eval()

    print("Best theta:")
    print(best_theta)

# # Saving and restoring a model

# In[27]:
if True:
    tf.reset_default_graph()

    n_epochs = 1000
    learning_rate = 0.01

    X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
    y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(mse)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # In[28]:
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(n_epochs):
            if epoch % 100 == 0:
                print("Epoch", epoch, "MSE =", mse.eval())
                save_path = saver.save(sess, "./tmp/my_model.ckpt")
            sess.run(training_op)

        best_theta = theta.eval()
        save_path = saver.save(sess, "./tmp/my_model_final.ckpt")

    print("Best theta:")
    print(best_theta)

# # Visualizing the graph
# ## inside Jupyter

# In[29]:
if False:
    from IPython.display import clear_output, Image, display, HTML


    def strip_consts(graph_def, max_const_size=32):
        """Strip large constant values from graph_def."""
        strip_def = tf.GraphDef()
        for n0 in graph_def.node:
            n = strip_def.node.add()
            n.MergeFrom(n0)
            if n.op == 'Const':
                tensor = n.attr['value'].tensor
                size = len(tensor.tensor_content)
                if size > max_const_size:
                    tensor.tensor_content = b"<stripped %d bytes>" % size
        return strip_def


    def show_graph(graph_def, max_const_size=32):
        """Visualize TensorFlow graph."""
        if hasattr(graph_def, 'as_graph_def'):
            graph_def = graph_def.as_graph_def()
        strip_def = strip_consts(graph_def, max_const_size=max_const_size)
        code = """
            <script>
              function load() {{
                document.getElementById("{id}").pbtxt = {data};
              }}
            </script>
            <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
            <div style="height:600px">
              <tf-graph-basic id="{id}"></tf-graph-basic>
            </div>
        """.format(data=repr(str(strip_def)), id='graph' + str(np.random.rand()))

        iframe = """
            <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
        """.format(code.replace('"', '&quot;'))
        display(HTML(iframe))


    # In[30]:
    show_graph(tf.get_default_graph())

# ## Using TensorBoard

# In[31]:   P243

if True:
    tf.reset_default_graph()

    from datetime import datetime

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    logdir = "{}/run-{}/".format(root_logdir, now)

    n_epochs = 1000
    learning_rate = 0.01

    X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
    y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(mse)

    init = tf.global_variables_initializer()

    mse_summary = tf.summary.scalar('MSE', mse)
    summary_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    # In[32]:
    n_epochs = min(10, n_epochs_4_test)
    batch_size = 100
    n_batches = int(np.ceil(m / batch_size))

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
                if batch_index % 10 == 0:
                    summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                    step = epoch * n_batches + batch_index
                    summary_writer.add_summary(summary_str, step)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        best_theta = theta.eval()

    summary_writer.flush()
    summary_writer.close()
    print("Best theta:")
    print(best_theta)

    # # Name scopes


    # In[33]:
    tf.reset_default_graph()

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    logdir = "{}/run-{}/".format(root_logdir, now)

    n_epochs = min(1000, n_epochs_4_test)
    learning_rate = 0.01

    X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
    y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    with tf.name_scope('loss') as scope:
        error = y_pred - y
        mse = tf.reduce_mean(tf.square(error), name="mse")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(mse)

    init = tf.global_variables_initializer()

    mse_summary = tf.summary.scalar('MSE', mse)
    summary_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    # In[34]:
    n_epochs = min(10, n_epochs_4_test)
    batch_size = 100
    n_batches = int(np.ceil(m / batch_size))

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
                if batch_index % 10 == 0:
                    summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                    step = epoch * n_batches + batch_index
                    summary_writer.add_summary(summary_str, step)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        best_theta = theta.eval()

    summary_writer.flush()
    summary_writer.close()
    print("Best theta:")
    print(best_theta)





    # In[35]:
    print(error.op.name)

    # In[36]:
    print(mse.op.name)

# In[37]:
tf.reset_default_graph()

a1 = tf.Variable(0, name="a")  # name == "a"
a2 = tf.Variable(0, name="a")  # name == "a_1"

with tf.name_scope("param"):  # name == "param"
    a3 = tf.Variable(0, name="a")  # name == "param/a"

with tf.name_scope("param"):  # name == "param_1"
    a4 = tf.Variable(0, name="a")  # name == "param_1/a"

for node in (a1, a2, a3, a4):
    print(node.op.name)

print("TOBEHERE")

# # Modularity

# An ugly flat code:

# In[38]:
tf.reset_default_graph()

n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")

w1 = tf.Variable(tf.random_normal((n_features, 1)), name="weights1")
w2 = tf.Variable(tf.random_normal((n_features, 1)), name="weights2")
b1 = tf.Variable(0.0, name="bias1")
b2 = tf.Variable(0.0, name="bias2")

linear1 = tf.add(tf.matmul(X, w1), b1, name="linear1")
linear2 = tf.add(tf.matmul(X, w2), b2, name="linear2")

relu1 = tf.maximum(linear1, 0, name="relu1")
relu2 = tf.maximum(linear2, 0, name="relu2")  # Oops, cut&paste error! Did you spot it?

output = tf.add_n([relu1, relu2], name="output")

# Much better, using a function to build the ReLUs:

# In[39]:
tf.reset_default_graph()


def relu(X):
    w_shape = int(X.get_shape()[1]), 1
    w = tf.Variable(tf.random_normal(w_shape), name="weights")
    b = tf.Variable(0.0, name="bias")
    linear = tf.add(tf.matmul(X, w), b, name="linear")
    return tf.maximum(linear, 0, name="relu")


n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name="output")
summary_writer = tf.summary.FileWriter("logs/relu1", tf.get_default_graph())

# Even better using name scopes:

# In[40]:
tf.reset_default_graph()


def relu(X):
    with tf.name_scope("relu"):
        w_shape = int(X.get_shape()[1]), 1
        w = tf.Variable(tf.random_normal(w_shape), name="weights")
        b = tf.Variable(0.0, name="bias")
        linear = tf.add(tf.matmul(X, w), b, name="linear")
        return tf.maximum(linear, 0, name="max")


n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name="output")

summary_writer = tf.summary.FileWriter("logs/relu2", tf.get_default_graph())

# In[41]:
summary_writer.close()

# Sharing a `threshold` variable the classic way, by defining it outside of the `relu()` function then passing it as a parameter:

# In[42]: as parameter

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




# In[43]: as attribute

tf.reset_default_graph()

def relu(X):
    with tf.name_scope("relu"):
        if not hasattr(relu, "threshold"):
            relu.threshold = tf.Variable(0.0, name="threshold")
        w_shape = int(X.get_shape()[1]), 1
        w = tf.Variable(tf.random_normal(w_shape), name="weights")
        b = tf.Variable(0.0, name="bias")
        linear = tf.add(tf.matmul(X, w), b, name="linear")
        return tf.maximum(linear, relu.threshold, name="max")


X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name="output")

# In[44]: as get_variable


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

# In[45]:
tf.reset_default_graph()


def relu(X):
    with tf.variable_scope("relu"):
        threshold = tf.get_variable("threshold", shape=(), initializer=tf.constant_initializer(0.0))
        w_shape = int(X.get_shape()[1]), 1
        w = tf.Variable(tf.random_normal(w_shape), name="weights")
        b = tf.Variable(0.0, name="bias")
        linear = tf.add(tf.matmul(X, w), b, name="linear")
        return tf.maximum(linear, threshold, name="max")


X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
with tf.variable_scope("", default_name="") as scope:
    first_relu = relu(X)  # create the shared variable
    scope.reuse_variables()  # then reuse it
    relus = [first_relu] + [relu(X) for i in range(4)]
output = tf.add_n(relus, name="output")

summary_writer = tf.summary.FileWriter("logs/relu8", tf.get_default_graph())
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

# # Extra material   # TOBEHERE

# ## Strings

# In[47]:
tf.reset_default_graph()

text = np.array("Do you want some café?".split())
text_tensor = tf.constant(text)

with tf.Session() as sess:
    print(text_tensor.eval())

# ## Distributed TensorFlow

# In[48]:
server = tf.train.Server.create_local_server()

# In[49]:
x = tf.constant(2) + tf.constant(3)
with tf.Session(server.target) as sess:
    print(sess.run(x))

# In[50]:
server.target

# In[51]:
class Const(object):
    def __init__(self, value):
        self.value = value

    def evaluate(self, **variables):
        return self.value

    def __str__(self):
        return str(self.value)


class Var(object):
    def __init__(self, name):
        self.name = name

    def evaluate(self, **variables):
        return variables[self.name]

    def __str__(self):
        return self.name


class BinaryOperator(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b


class Add(BinaryOperator):
    def evaluate(self, **variables):
        return self.a.evaluate(**variables) + self.b.evaluate(**variables)

    def __str__(self):
        return "{} + {}".format(self.a, self.b)


class Mul(BinaryOperator):
    def evaluate(self, **variables):
        return self.a.evaluate(**variables) * self.b.evaluate(**variables)

    def __str__(self):
        return "({}) * ({})".format(self.a, self.b)


x = Var("x")
y = Var("y")
f = Add(Mul(Mul(x, x), y), Add(y, Const(2)))  # f(x,y) = x²y + y + 2
print("f(x,y) =", f)
print("f(3,4) =", f.evaluate(x=3, y=4))

# ## Computing gradients
# ### Mathematical differentiation

# In[52]:
df_dx = Mul(Const(2), Mul(Var("x"), Var("y")))  # df/dx = 2xy
df_dy = Add(Mul(Var("x"), Var("x")), Const(1))  # df/dy = x² + 1
print("df/dx(3,4) =", df_dx.evaluate(x=3, y=4))
print("df/dy(3,4) =", df_dy.evaluate(x=3, y=4))


# ### Numerical differentiation

# In[53]:
def derivative(f, x, y, x_eps, y_eps):
    return (f.evaluate(x=x + x_eps, y=y + y_eps) - f.evaluate(x=x, y=y)) / (x_eps + y_eps)


df_dx_34 = derivative(f, x=3, y=4, x_eps=0.0001, y_eps=0)
df_dy_34 = derivative(f, x=3, y=4, x_eps=0, y_eps=0.0001)
print("df/dx(3,4) =", df_dx_34)
print("df/dy(3,4) =", df_dy_34)

# In[54]:
def f(x, y):
    return x ** 2 * y + y + 2


def derivative(f, x, y, x_eps, y_eps):
    return (f(x + x_eps, y + y_eps) - f(x, y)) / (x_eps + y_eps)


df_dx = derivative(f, 3, 4, 0.00001, 0)
df_dy = derivative(f, 3, 4, 0, 0.00001)

# In[55]:
print(df_dx)
print(df_dy)

# ### Symbolic differentiation

# In[56]:
Const.derive = lambda self, var: Const(0)
Var.derive = lambda self, var: Const(1) if self.name == var else Const(0)
Add.derive = lambda self, var: Add(self.a.derive(var), self.b.derive(var))
Mul.derive = lambda self, var: Add(Mul(self.a, self.b.derive(var)), Mul(self.a.derive(var), self.b))

x = Var("x")
y = Var("y")
f = Add(Mul(Mul(x, x), y), Add(y, Const(2)))  # f(x,y) = x²y + y + 2

df_dx = f.derive("x")  # 2xy
df_dy = f.derive("y")  # x² + 1
print("df/dx(3,4) =", df_dx.evaluate(x=3, y=4))
print("df/dy(3,4) =", df_dy.evaluate(x=3, y=4))


# ### Automatic differentiation (autodiff) – forward mode

# In[57]:
class Const(object):
    def __init__(self, value):
        self.value = value

    def evaluate(self, derive, **variables):
        return self.value, 0

    def __str__(self):
        return str(self.value)


class Var(object):
    def __init__(self, name):
        self.name = name

    def evaluate(self, derive, **variables):
        return variables[self.name], (1 if derive == self.name else 0)

    def __str__(self):
        return self.name


class BinaryOperator(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b


class Add(BinaryOperator):
    def evaluate(self, derive, **variables):
        a, da = self.a.evaluate(derive, **variables)
        b, db = self.b.evaluate(derive, **variables)
        return a + b, da + db

    def __str__(self):
        return "{} + {}".format(self.a, self.b)


class Mul(BinaryOperator):
    def evaluate(self, derive, **variables):
        a, da = self.a.evaluate(derive, **variables)
        b, db = self.b.evaluate(derive, **variables)
        return a * b, a * db + da * b

    def __str__(self):
        return "({}) * ({})".format(self.a, self.b)


x = Var("x")
y = Var("y")
f = Add(Mul(Mul(x, x), y), Add(y, Const(2)))  # f(x,y) = x²y + y + 2
f34, df_dx_34 = f.evaluate(x=3, y=4, derive="x")
f34, df_dy_34 = f.evaluate(x=3, y=4, derive="y")
print("f(3,4) =", f34)
print("df/dx(3,4) =", df_dx_34)
print("df/dy(3,4) =", df_dy_34)


# ### Autodiff – Reverse mode

# In[58]:
class Const(object):
    def __init__(self, value):
        self.derivative = 0
        self.value = value

    def evaluate(self, **variables):
        return self.value

    def backpropagate(self, derivative):
        pass

    def __str__(self):
        return str(self.value)


class Var(object):
    def __init__(self, name):
        self.name = name

    def evaluate(self, **variables):
        self.derivative = 0
        self.value = variables[self.name]
        return self.value

    def backpropagate(self, derivative):
        self.derivative += derivative

    def __str__(self):
        return self.name


class BinaryOperator(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b


class Add(BinaryOperator):
    def evaluate(self, **variables):
        self.derivative = 0
        self.value = self.a.evaluate(**variables) + self.b.evaluate(**variables)
        return self.value

    def backpropagate(self, derivative):
        self.derivative += derivative
        self.a.backpropagate(derivative)
        self.b.backpropagate(derivative)

    def __str__(self):
        return "{} + {}".format(self.a, self.b)


class Mul(BinaryOperator):
    def evaluate(self, **variables):
        self.derivative = 0
        self.value = self.a.evaluate(**variables) * self.b.evaluate(**variables)
        return self.value

    def backpropagate(self, derivative):
        self.derivative += derivative
        self.a.backpropagate(derivative * self.b.value)
        self.b.backpropagate(derivative * self.a.value)

    def __str__(self):
        return "({}) * ({})".format(self.a, self.b)

# In[59]:
x = Var("x")
y = Var("y")
f = Add(Mul(Mul(x, x), y), Add(y, Const(2)))  # f(x,y) = x²y + y + 2
f34 = f.evaluate(x=3, y=4)
f.backpropagate(1)
print("f(3,4) =", f34)
print("df/dx(3,4) =", x.derivative)
print("df/dy(3,4) =", y.derivative)

# ### Autodiff – reverse mode (using TensorFlow)

# In[60]:
tf.reset_default_graph()

x = tf.Variable(3., name="x")
y = tf.Variable(4., name="x")
f = x * x * y + y + 2

gradients = tf.gradients(f, [x, y])

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    f_val, gradients_val = sess.run([f, gradients])

f_val, gradients_val

# # Exercise solutions

# **Coming soon**







print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> DONE! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
