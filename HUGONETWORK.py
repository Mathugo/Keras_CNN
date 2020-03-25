import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras import backend as K
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Activation

class HUGONETWORK:
    @staticmethod
    def build(n_classes=2, width=224, height=224, depth=3):
        inputShape = (height, width, depth)
        model = Sequential()
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        #first set
        model.add(Conv2D(input_shape=inputShape, filters=16, kernel_size=(5,5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        # First set of CONV -> RELU -> POOL
        #CONV layer wil learn 16 convolution filters, each of which are 5x5

        #second set
        model.add(Conv2D(filters=32, kernel_size=(5, 5), padding="same"))
        model.add(Conv2D(filters=32, kernel_size=(5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        # Learning 32 convolutionnal filters (we are going deeper in the network architecture)
        #third set
        model.add(Conv2D(filters=64, kernel_size=(5,5), padding="same"))
        model.add(Conv2D(filters=64, kernel_size=(5,5), padding="same"))
        #model.add(Conv2D(filters=64, kernel_size=(5,5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        #fourth set
        model.add(Conv2D(filters=128, kernel_size=(5,5), padding="same"))
        model.add(Conv2D(filters=128, kernel_size=(5,5), padding="same"))
        model.add(Conv2D(filters=128, kernel_size=(5,5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        #fifth (and only) set of FC => RELU layers
        model.add(Flatten()) 
        #we flatten the output of MaPooling2D layer into a single vector -> apply dense/fully connected layers
        # transform format of the images from two dimensional array (ex : 28x28 pixels) to a one : 28*28=784
        model.add(Dense(512))
        #fully-connected layer contains 512 nodes -> we pass it though nonlinear ReLu activation
        model.add(Activation("relu"))
        #softmax classifier
        model.add(Dense(n_classes)) 
        #Number of nodes is equal to the number of classes which will yield the proba for each class
        model.add(Activation("softmax"))
        #return the constructed network architecture
        return model
