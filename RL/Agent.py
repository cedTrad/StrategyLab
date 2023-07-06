import datetime
import functools
import os

from collections import deque
from functools import reduce

#import imageio

import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

tf.keras.backend.set_floatx("float64")


def actor(state_shape, action_shape, units = (512, 256, 64)):
    state_shape_flattened = functools.reduce(lambda x, y : x*y, start_shape)
    
