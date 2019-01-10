# tobeplot


# Common imports
import numpy as np
import numpy.random as rnd
import os
import time

# to make this notebook's output stable across runs
rnd.seed(42)

# To plot pretty figures and animations
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from my_utility import show_plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "rl"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

import gym
env = gym.make('MsPacman-v0')
obs = env.reset()
# print('obs.shape', obs.shape())


# tobeplot ： Static image
if False:
    img = env.render(mode="rgb_array")
    plt.figure(figsize=(5,4))
    plt.imshow(img)
    plt.axis("off")
    save_fig("MsPacman")
    show_plt(plt, is_plt_show=False)



#  (img == obs).all()

def plot_environment(env, figsize=(5,4)):
    plt.close()  # or else nbagg sometimes plots in the previous cell
    plt.figure(figsize=figsize)
    img = env.render(mode="rgb_array")
    plt.imshow(img)
    plt.axis("off")
    show_plt(plt, is_plt_show=False)

# env.action_space  #Discrete(9)

env.reset()
for step in range(110):
    env.step(3) #left
    # time.sleep(1)
    # plot_environment(env)
for step in range(40):
    env.step(8) #lower-left


# plot_environment(env)
obs, reward, done, info = env.step(0)

# obs.shape  #(210, 160, 3)
# info: internal state of env

frames = []

n_max_steps = 1000
n_change_steps = 10

obs = env.reset()
for step in range(n_max_steps):
    img = env.render(mode="rgb_array")
    frames.append(img)
    if step % n_change_steps == 0:
        action = env.action_space.sample() # play randomly
    obs, reward, done, info = env.step(action)
    if done:
        break



def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

def plot_animation(frames, repeat=False, interval=40):
    plt.close()  # or else nbagg sometimes plots in the previous cell
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    return animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)

# tobeplot ： animation
if False:
    video = plot_animation(frames)
    show_plt(plt, is_plt_show=False)

env.close()



# ########################################################
# CartPole-v0
# ########################################################
# xvfb-run -s "-screen 0 1400x900x24" jupyter notebook


env = gym.make("CartPole-v0")
obs = env.reset()

from PIL import Image, ImageDraw

################# Single Step
if True:
    try:
        from pyglet.gl import gl_info
        openai_cart_pole_rendering = True   # no problem, let's use OpenAI gym's rendering function
    except Exception:
        openai_cart_pole_rendering = False  # probably no X server available, let's use our own rendering function

    def render_cart_pole(env, obs):
        if openai_cart_pole_rendering:
            # use OpenAI gym's rendering function
            return env.render(mode="rgb_array")
        else:
            # rendering for the cart pole environment (in case OpenAI gym can't do it)
            img_w = 600
            img_h = 400
            cart_w = img_w // 12
            cart_h = img_h // 15
            pole_len = img_h // 3.5
            pole_w = img_w // 80 + 1
            x_width = 2
            max_ang = 0.2
            bg_col = (255, 255, 255)
            cart_col = 0x000000 # Blue Green Red
            pole_col = 0x669acc # Blue Green Red

            pos, vel, ang, ang_vel = obs
            img = Image.new('RGB', (img_w, img_h), bg_col)
            draw = ImageDraw.Draw(img)
            cart_x = pos * img_w // x_width + img_w // x_width
            cart_y = img_h * 95 // 100
            top_pole_x = cart_x + pole_len * np.sin(ang)
            top_pole_y = cart_y - cart_h // 2 - pole_len * np.cos(ang)
            draw.line((0, cart_y, img_w, cart_y), fill=0)
            draw.rectangle((cart_x - cart_w // 2, cart_y - cart_h // 2, cart_x + cart_w // 2, cart_y + cart_h // 2), fill=cart_col) # draw cart
            draw.line((cart_x, cart_y - cart_h // 2, top_pole_x, top_pole_y), fill=pole_col, width=pole_w) # draw pole
            return np.array(img)

    def plot_cart_pole(env, obs):
        plt.close()  # or else nbagg sometimes plots in the previous cell
        img = render_cart_pole(env, obs)
        plt.imshow(img)
        plt.axis("off")
        show_plt(plt, is_plt_show=False)

    plot_cart_pole(env, obs)

    # env.action_space



    ################# Start actions

    def plot_cart_pole(plt, env, obs):
        plt.close()  # or else nbagg sometimes plots in the previous cell
        img = render_cart_pole(env, obs)
        plt.imshow(img)
        plt.axis("off")
        save_fig("cart_pole_plot")


    ################# Step left
    obs = env.reset()
    while True:
        obs, reward, done, info = env.step(0)
        if done:
            break

    plot_cart_pole(plt, env, obs)

    ################# Step right

    obs = env.reset()
    while True:
        obs, reward, done, info = env.step(1)
        if done:
            break

    plot_cart_pole(plt, env, obs)

################# Simple hard-coded policy
# tobeplot ：
if False:
    frames = []

    n_max_steps = 1000
    n_change_steps = 10

    obs = env.reset()
    for step in range(n_max_steps):
        img = render_cart_pole(env, obs)
        frames.append(img)

        # hard-coded policy
        position, velocity, angle, angular_velocity = obs
        if angle < 0:
            action = 0
        else:
            action = 1

        obs, reward, done, info = env.step(action)
        if done:
            break


    video = plot_animation(frames)
    show_plt(plt, is_plt_show=False)






################# Neural Network

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

# 1. Specify the network architecture
n_inputs = 4  # == env.observation_space.shape[0]
n_hidden = 4  # it's a simple task, we don't need more than this
n_outputs = 1 # only outputs the probability of accelerating left
initializer = tf.contrib.layers.variance_scaling_initializer()

# 2. Build the neural network
X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden = fully_connected(X, n_hidden, activation_fn=tf.nn.elu,
                         weights_initializer=initializer)
outputs = fully_connected(hidden, n_outputs, activation_fn=tf.nn.sigmoid,
                          weights_initializer=initializer)

# 3. Select a random action based on the estimated probabilities
p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)

#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

n_max_steps = 1000
frames = []

with tf.Session() as sess:
    init.run()
    obs = env.reset()
    for step in range(n_max_steps):
        img = render_cart_pole(env, obs)
        frames.append(img)
        action_val = action.eval(feed_dict={X: obs.reshape(1, n_inputs)})
        obs, reward, done, info = env.step(action_val[0][0])
        if done:
            break

env.close()


video = plot_animation(frames)
show_plt(plt, is_plt_show=False)





# tobeplot ：

# tobeplot ：

#
# tobeplot ：



print('done -------------------------- ')