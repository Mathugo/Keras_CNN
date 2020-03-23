from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

# To avoid cnn errors
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

print("[*] Settins config ..")
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
print("[*] Done")
# ---------------

#tried (64*64)
IMAGE_SIZE = (64, 64)


class process_network:
    def __init__(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("-m", "--model", required=True,
        help="path to trained model mode")
        ap.add_argument("-i", "--image", required=True,
        help="path to input image")
        self.args= vars(ap.parse_args())
        print("[*] Loading network ..")
        self.model = load_model(self.args["model"])
        print("[*] Done")
        self.path_frame = self.args["image"]
        self.frame = cv2.imread(self.path_frame)
        self.load_img(self.frame)
        self.getProba()

    def load_img(self, frame):
        #image = cv2.imread(args["image"])
        self.image = frame
        self.orig = self.image.copy()
        self.image = cv2.resize(self.image, IMAGE_SIZE)
        self.image = self.image.astype("float")/ 255.0
        self.image = img_to_array(self.image)
        self.image = np.expand_dims(self.image, axis=0) # to have dims (1, width, height, 3)

    def getProba(self):
        nothugo, hugo = self.model.predict(self.image)[0]
        label = "Hugo" if hugo > nothugo else "Unknown"
        proba = hugo if hugo > nothugo else nothugo
        print("Hugo {} Not Hugo {}".format(str(hugo), str(nothugo)))
        label = "{}: {:.2f}%".format(label, proba * 100)
        output = imutils.resize(self.orig, width=400)
        cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	    0.7, (0, 255, 0), 2)
        cv2.imshow("Output", output)
        cv2.waitKey(0)