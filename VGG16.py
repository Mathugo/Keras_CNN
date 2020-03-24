import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras import backend as K

class VGG16Net:
    @staticmethod
    def build(self, classes=2, width=224, height=224, depth=3):
        inputShape = (height, width, depth)
        model = Sequential()
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        model.add(Conv2D(input_shape=inputShape,filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

        """
                → 2 x convolution layer of 64 channel of 3x3 kernal and same padding
        → 1 x maxpool layer of 2x2 pool size and stride 2x2
        → 2 x convolution layer of 128 channel of 3x3 kernal and same padding
        → 1 x maxpool layer of 2x2 pool size and stride 2x2
        → 3 x convolution layer of 256 channel of 3x3 kernal and same padding
        → 1 x maxpool layer of 2x2 pool size and stride 2x2
        → 3 x convolution layer of 512 channel of 3x3 kernal and same padding
        → 1 x maxpool layer of 2x2 pool size and stride 2x2
        → 3 x convolution layer of 512 channel of 3x3 kernal and same padding
        → 1 x maxpool layer of 2x2 pool size and stride 2x2
        """
        model.add(Flatten())
        model.add(Dense(units=4096,activation="relu"))
        model.add(Dense(units=4096,activation="relu"))
        model.add(Dense(units=classes, activation="softmax"))   
        """→ 1 x Dense layer of 4096 units
        → 1 x Dense layer of 4096 units
        → 1 x Dense Softmax layer of n classes units
        """
        return model