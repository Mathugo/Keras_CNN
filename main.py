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
BS_ = 32

def main():
   # l = process_network()
    mo = Model(EPOCHS=EPOCHS_, INIT_LR=LR_, BS=BS_)
    mo.run()
    mo.show()
    
main()
session.close()
del session
