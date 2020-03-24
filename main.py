# To avoid cnn errors
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

print("[*] Settings config ..")
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
print("[*] Done")

from model import *
from load_network import *

EPOCHS_ = 25
LR_ = 1e-3
BS_ = 8 #batch size of 128 may increase gpu utilisation

def main():
    #l = process_network()
    model = Model(EPOCHS=EPOCHS_, INIT_LR=LR_, BS=BS_)
    model.run()
    model.save_plot()
    
main()
print("[!] Closing session ..")
session.close()
del session
print("[*] Done")
