import tensorflow as tf

import warnings
import numpy as np
import gym

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


env = gym.make('CartPole-v0')

num_actions = env.action_space.n
state_shape = env.observation_space.shape[0]

gamma = 0.95

def discount_and_normalize_rewards(episode_rewards):
    discounted_rewards = np.zeros_like(episode_rewards)
    
    reward_to_go = 0.0
    for i in reversed(len(episode_rewards)):
        reward_to_go = reward_to_go * gamma * episode_rewards[i]
        discounted_rewards[i] = reward_to_go
    
    # normalize and return the reward
    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards)
    
    return discounted_rewards


tf.reset_default_graph()

state_ph = tf.placeholder(tf.float32, [None, state_shape], name="state_ph")
action_ph = tf.placeholder(tf.float32, [None, num_actions], name="action_ph")
discounted_rewards_ph = tf.placeholder(tf.float32, [None,], name="discounted_rewards")


layer1 = tf.layers.dense(state_ph, units=32, activation=tf.nn.relu)
layer2 = tf.layers.dense(layer1, units=num_actions)
prob_dist = tf.nn.softmax(layer2)


neg_log_policy = tf.nn.softmax_cross_entropy_with_logits_v2(logits = layer2, labels = action_ph)
loss = tf.reduce_mean(neg_log_policy * discounted_rewards_ph)
train = tf.train.AdamOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())




