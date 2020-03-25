import matplotlib
matplotlib.use("Agg") # We can save the plot to disk in background
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from network import Network as LeNet
from VGG16 import VGG16Net
from imutils import paths
import matplotlib.pyplot as plt
import argparse
import random
import cv2
import keras

from RESNET50 import RESNET50
from INCEPTIONV3 import INCEPTIONV3

# keras imports
from keras.applications.vgg16 import VGG16, preprocess_input
#from keras.applications.vgg19 import VGG19, preprocess_input
#from keras.applications.xception import Xception, preprocess_input
#from keras.applications.resnet50 import ResNet50, preprocess_input
#from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
#from keras.applications.mobilenet import MobileNet, preprocess_input
#from keras.applications.inception_v3 import InceptionV3, preprocess_input
#from keras.preprocessing import image
#from keras.models import Model
#from keras.models import model_from_json
from keras.layers import Input
from keras.callbacks import ModelCheckpoint, EarlyStopping

# other imports
from sklearn.preprocessing import LabelEncoder
import numpy as np
import glob
import cv2
import h5py
import os
import json
import datetime
import time

#Load our image dataset
#pre-process images
#instantiate our CNN
#train our image classifier

IMAGE_SIZE = (224, 224)
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
CLASSES = 2

class Model:
    
    def __init__(self, args, EPOCHS=25, INIT_LR = 1e-3, BS=64):
        self.args = args
    # initialize the number of epochs to train for, initial learning rate,
        # and batch size
        self.EPOCHS = EPOCHS #25
        self.INIT_LR = INIT_LR #-3
        self.BS = BS # 32
        self.number = self.args["number_images"]

        print("[STATUS] start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
        start = time.time()
        print("[*] Starting model with EPOCHS {} LR {} Batch Size {}".format(self.EPOCHS, self.INIT_LR, self.BS))
  
    def load_dataset(self):
        # initialize the data and labels
        print("[INFO] loading images...")
        self.data = []
        self.labels = []
        # grab the image paths and randomly shuffle them
        self.imagePaths = sorted(list(paths.list_images(self.args["dataset"])))
        random.seed(42)
        random.shuffle(self.imagePaths)
        print("[*] Loading {} images".format(self.number))
        self.count_label = {"hugo":0, "not_hugo":0, "alex":0}
    
        for imagePath in self.imagePaths:
            image = cv2.imread(imagePath)
            image = cv2.resize(image, IMAGE_SIZE) #Spatial dimension requiered for LeNet
            image = img_to_array(image)
            
            # extract class label from image path and update the labels list
            label = imagePath.split(os.path.sep)[-2]

            if self.count_label["hugo"] == -1 and self.count_label["not_hugo"] == -1:
                print("[!] {} images loaded".format(self.number))
                break 

            elif self.count_label[label] >= self.number:
                self.count_label[label] = -1

            elif self.count_label[label] != -1:

                self.count_label[label]+=1 # Update the number of image loaded
                label = 1 if label == "hugo" else 0
                self.data.append(image)
                self.labels.append(label)
            
        print("[*] Done loading dataset")

    def transform_data(self):
        print("[*] Processing data ..")
        # scale the raw pixel intensities to te range [0, 1]
        #self.data = np.array(self.data, dtype="float") / 255.0
        self.data = np.array(self.data, dtype="float32")
        self.labels = np.array(self.labels)
        #partition the data into trainning and testing splits using 75% of 
        # the data for training and other for testing
        (self.trainX, self.testX, self.trainY, self.testY) = train_test_split(self.data, 
        self.labels, test_size=0.25, random_state=42)

        #convert the labels from integers to vectors
        self.trainY = to_categorical(self.trainY, num_classes=2)
        self.testY = to_categorical(self.testY, num_classes=2)
        #construct the image generator for data augmentation -> generate additionnal training data
        self.aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode="nearest")
        print("[*] Done")

    def build(self):
        print("[*] Compiling model ..")
        #self.model = LeNet.build(width=IMAGE_WIDTH, height=IMAGE_HEIGHT, depth=3, classes=CLASSES)
        #self.model = VGG16Net.build(classes=CLASSES, width=IMAGE_WIDTH, height=IMAGE_HEIGHT, depth=3)
        self.model = RESNET50.build(n_classes=CLASSES)
        self.opt = Adam(lr=self.INIT_LR, decay=self.INIT_LR/self.EPOCHS)
        self.model.compile(loss=keras.losses.categorical_crossentropy, metrics=["accuracy"], optimizer=self.opt)
        print("[*] Done")
        self.model.summary()
       #model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
            #loss is binary because two classes, otherwise --> categorical_crossentropy
      
    def train(self):
        print("[*] Creating check point ..")
        checkPoint = ModelCheckpoint(self.args["model"], monitor="val_accuracy",
        verbose=1, save_best_only=True, save_weights_only=False, mode="auto", period=1)
        early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')
        # saved to disk if val_accuracy better than before
        # earlystopping stop trainning of the model if there is no increase in the parameter 
        # patience set to 20, it 0 increase in 20 epochs, it will stop
        print("[*] Done")
        print("[!] Training network ...")
        self.H = self.model.fit_generator(self.aug.flow(self.trainX, self.trainY, batch_size=self.BS),
        validation_data=(self.testX, self.testY), steps_per_epoch=len(self.trainX) // self.BS,
        epochs=self.EPOCHS, verbose=1, callbacks=[checkPoint, early])
        print("[*] Done")

    def run(self):
        self.load_dataset()
        self.transform_data()
        self.build()
        self.train()        

        #save the model
        print("[*] Saving model ..")
        self.model.save(self.args["model"])
        print("[*] Done")

    def save_plot(self):
        #plot the loss and accuracy
        plt.style.use("ggplot")
        plt.figure()
        N = self.EPOCHS
        plt.plot(np.arange(0, N), self.H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, N), self.H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, N), self.H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, N), self.H.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy on Hugo/Not Hugo")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig(self.args["plot"])

"""
Stochastic gradient descent is an optimization algorithm that estimates the error gradient for the current 
state of the model using examples from the training dataset, then updates the weights 
of the model using the back-propagation of errors algorithm, referred to as simply backpropagation.
The amount that the weights are updated during training is referred to as the step size or the “learning rate.”
Specifically, the learning rate is a configurable hyperparameter used in the training of neural networks that has
a small positive value, often in the range between 0.0 and 1.0.
The learning rate controls how quickly the model is adapted to the problem. 
Smaller learning rates require more training epochs given the smaller changes made to the weights 
each update, whereas larger learning rates result in rapid changes and require fewer training epochs.
A learning rate that is too large can cause the model to converge too quickly to a suboptimal solution,
whereas a learning rate that is too small can cause the process to get stuck.
"""


"""
Assume you have a dataset with 200 samples (rows of data) and you choose a batch size of 5 and 1,000 epochs.
This means that the dataset will be divided into 40 batches, each with five samples. The model weights will be updated after each batch of five samples.
This also means that one epoch will involve 40 batches or 40 updates to the model.
With 1,000 epochs, the model will be exposed to or pass through the whole dataset 1,000 times. That is a total of 40,000 batches during the entire training process.
"""