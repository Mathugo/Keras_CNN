import matplotlib
matplotlib.use("Agg") # We can save the plot to disk in background
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from network import Network as LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
#Load our image dataset
#pre-process images
#instantiate our CNN
#train our image classifier

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64

class Model:
    def __init__(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("-d", "--dataset", required=True,
            help="path to input dataset")
        ap.add_argument("-m", "--model", required=True,
            help="path to output model")
        ap.add_argument("-p", "--plot", type=str, default="plot.png",
            help="path to output accuracy/loss plot")
        self.args = vars(ap.parse_args())
    # initialize the number of epochs to train for, initial learning rate,
        # and batch size
        self.EPOCHS = 25
        self.INIT_LR = 1e-3
        self.BS = 32

    def load_dataset(self):
        # initialize the data and labels
        print("[INFO] loading images...")
        self.data = []
        self.labels = []
        # grab the image paths and randomly shuffle them
        self.imagePaths = sorted(list(paths.list_images(self.args["dataset"])))
        random.seed(42)
        random.shuffle(self.imagePaths)

        for imagePath in self.imagePaths:
            image = cv2.imread(imagePath)
            image = cv2.resize(image, (28, 28)) #Spatial dimension requiered for LeNet
            image = img_to_array(image)
            self.data.append(image)

            # extract class label from image path and update the labels list
            label = imagePath.split(os.path.sep)[-2]
            label = 1 if label == "hugo" else 0
            self.labels.append(label)

    def transform_data(self):
        # scale the raw pixel intensities to te range [0, 1]
        self.data = np.array(self.data, dtype="float") / 255.0
        self.labels = np.array(self.labels)
        #partition the data into trainning and testing splits using 75% of 
        # the data for training and other for testing
        (self.trainX, self.testX, self.trainY, self.testY) = train_test_split(self.data, self.labels, test_size=0.25, random_state=42)

        #convert the labels from integers to vectors
        self.trainY = to_categorical(self.trainY, num_classes=2)
        self.testY = to_categorical(self.testY, num_classes=2)

        #construct the image generator for data augmentation -> generate additionnal training data
        self.aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode="nearest")

    def run(self):
        self.load_dataset()
        self.transform_data()
       
        print("[*] Compiling model ..")
        model = LeNet.build(width=IMAGE_WIDTH, height=IMAGE_HEIGHT, depth=3, classes=2)
        opt = Adam(lr=self.INIT_LR, decay=self.INIT_LR/self.EPOCHS)
        model.compile(loss="binary_crossentropy", optimizer=opt,
            metrics=["accuracy"])
            #loss is binary because two classes, otherwise --> categorical_crossentropy
        print("[*] Done")
        #train the network
        print("[!] Training network ...")
        self.H = model.fit_generator(self.aug.flow(self.trainX, self.trainY, batch_size=self.BS),
        validation_data=(self.testX, self.testY), steps_per_epoch=len(self.trainX) // self.BS,
        epochs=self.EPOCHS, verbose=1)

        print("[*] Done")

        #save the model
        print("[*] Saving model ..")
        model.save(self.args["model"])
        print("[*] Done")

    def show(self):
        #plot the loss and accuracy
        plt.style.use("ggplot")
        plt.figure()
        N = self.EPOCHS
        plt.plot(np.arange(0, N), self.H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, N), self.H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, N), self.H.history["acc"], label="train_acc")
        plt.plot(np.arange(0, N), self.H.history["val_acc"], label="val_acc")
        plt.title("Training Loss and Accuracy on Santa/Not Santa")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig(self.args["plot"])