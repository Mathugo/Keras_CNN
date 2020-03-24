from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K



# ---------------

class Network:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential() #class since we will be sequentially adding layers to the model
        inputShape = (height, width, depth)
        #depth -> The number of channels in our input images (1  for grayscale single channel images, 3  for standard RGB images
        #classes -> number of classes we want to recognize 
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        #first set
        model.add(Conv2D(20, (5,5), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        # First set of CONV -> RELU -> POOL
        #CONV layer wil learn 20 convolution filters, each of which are 5x5

        #second set
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        # Learning 50 convolutionnal filters (we are going deeper in the network architecture)

        #third set
        #first (and only) set of FC => RELU layers
        model.add(Flatten()) 
        #we flatten the output of MaPooling2D layer into a single vector -> apply dense/fully connected layers
        model.add(Dense(500))
        #fully-connected layer contains 500 nodes -> we pass it though nonlinear ReLu activation
        model.add(Activation("relu"))
        #softmax classifier
        model.add(Dense(classes)) 
        #Number of nodes is equal to the number of classes which will yield the proba for each class
        model.add(Activation("softmax"))
        #return the constructed network architecture
        return model