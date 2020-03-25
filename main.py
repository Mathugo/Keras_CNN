from model import *
from app import Application

EPOCHS_ = 25
LR_ = 1e-3
BS_ = 16 #batch size of 128 may increase gpu utilisation

def main():
    def start_app():
        ap = Application()
        ap.create_session()
        ap.run()
        ap.close_session()
    def start_process():
        from load_network import process_network
        l = process_network()
    
    start_process()
main()


