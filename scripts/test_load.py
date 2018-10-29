from __future__ import print_function

try:
    import time

    import os
    import sys
    import argparse

    import torch

    from cyphercat.load_data import get_lfw, get_tiny_imagenet


except ImportError as e:
    print(e)
    raise ImportError




def main():
    global args
    parser = argparse.ArgumentParser(description="Convolutional NN Testing Script")
    parser.add_argument("-c", "--config", dest="configfile", default='config.yml', help="Path to yaml")
    args = parser.parse_args()

    print("Testing")

    print("Python: %s" % sys.version)
    print("Pytorch: %s" % torch.__version__)
    
    # determine device to run network on (runs on gpu if available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    out_dir = '/Users/zhampela/lab41/projects/cyphercat/my-cyphercat-repo/Datasets/'

    get_tiny_imagenet(out_dir)

    get_lfw(out_dir)

if __name__ == "__main__":
    main()
