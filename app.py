from model import Model
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf
import os
import argparse

class Application:
    def __init__(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("-d", "--dataset", required=True, type=str,
            help="path to input dataset")
        ap.add_argument("-p", "--plot", type=str, default="plot.png",
            help="path to output accuracy/loss plot")
        ap.add_argument("-n", "--number_images", type=int, default=5000,
            help="number of images to load")
        ap.add_argument("-e", "--epochs", type=int, default=25,
            help="number of epochs")
        ap.add_argument("-b", "--batch_size", type=int, default=16,
            help="batch size")
        ap.add_argument("-lr","--learning_rate",type=float, default=1e-3,
            help="learning rate")
        ap.add_argument("-m","--model",type=str, default="RESNET50",
            help="model to choose `LENET` or `INCEPTIONV3` or `RESNET50` or `VGG16`")
        ap.add_argument("-g","--gpu", type=str, default="yes",
            help="Use the gpu `yes` or `no`")
        self.args = vars(ap.parse_args())   
        self.model = Model(self.args)
    
    def run(self):
        self.model.run()
        self.model.save_plot()

    def create_session(self):
        print("[*] Settings config ..")
        if self.args["gpu"] == "no":
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        if tf.test.gpu_device_name():
            print('GPU found')
        else:
            print("No GPU found")
        config = ConfigProto()
        self.session = InteractiveSession(config=config)
        print("[*] Done")
        #config.gpu_options.allow_growth = True
        #config.gpu_options.allocator_type = "BFC"
        #config.gpu_options.per_process_gpu_memory_fraction = 0.90

    def close_session(self):
        print("[!] Closing session ..")
        self.session.close()
        del self.session
        print("[*] Done")
    