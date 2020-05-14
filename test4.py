import konlpy
import matplotlib
from konlpy.tag import Okt
import json
import os
from pprint import pprint
import nltk
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from keras.models import load_model
import tensorflow as tf

#model = load_model('myModel.h5')
#model.summary()

model = tf.keras.models.load_model('myModel.h5')
model.summary()