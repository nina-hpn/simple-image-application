import numpy as np
import os
import skimage.io as io
from keras.models import Conv2D, MaxPooling2D
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

class UNet:
    def __init__(self, pretrained_weights=None, input_size=(225, 256,1)):
        self.pretrained_weights = pretrained_weights
        self.input_size = input_size

    def model(self):
        self.conv = self.conv()
        self.deconv = self.deconv()

    def conv(self):
        self.inputs = Input(self.input_size)
        self.conv1 =  Conv2D(64, 3, activation='relu', padding='same', )
