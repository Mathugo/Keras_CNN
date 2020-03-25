from model import *
from load_network import *
from app import Application

EPOCHS_ = 25
LR_ = 1e-3
BS_ = 16 #batch size of 128 may increase gpu utilisation

def main():
    #l = process_network()
    ap = Application()
    ap.create_session()
    ap.run()
    app.close_session()
    
main()


