from __future__ import print_function

try:
    import os
    import sys 
    import argparse
    import numpy as np 
    import matplotlib.pyplot as plt
    
    import torch
    import torchvision 
    import torch.nn as nn
    import torch.optim as optim
    
    from cyphercat.models import get_predef_model, weights_init
    from cyphercat.train import *
    from cyphercat.metrics import *  
    from cyphercat.datadefs import CCATDataset
    from cyphercat.datadefs.cifar10_dataset import Cifar10_preload_and_split
    
    from cyphercat.load_data import prep_data
    from cyphercat.utils import Configurator, ModelConfig, DataStruct

except ImportError as e:
    print(e)
    raise ImportError
    
    
def main():
    global args
    parser = argparse.ArgumentParser(description="Convolutional NN Testing Script")
    parser.add_argument("-c", "--config", dest="configfile", default='scripts/configs/lfw.yml', help="Path to yaml")

    args = parser.parse_args()

    print("Python: %s" % sys.version)
    print("Pytorch: %s" % torch.__version__)
    
    # determine device to run network on (runs on gpu if available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    # Get configuration file
    configr = Configurator(args.configfile)

    # Get dataset configuration 
    dataset_config       = configr.dataset
    train_model_config   = configr.train_model

    # Training model params
    train_config = ModelConfig(train_model_config)
    model_name = train_config.name

    # Datastruct for prepping data 
    data_struct = DataStruct(dataset_config)

    # Simple download / unpacker function
    prep_data(data_struct)

    # Hyperparameters
    n_epochs = train_config.epochs
    batch_size = train_config.batchsize
    learnrate = train_config.learnrate
    loss = nn.CrossEntropyLoss()
    
    # Data augmentation 
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
        
    splits = [0.4, 0.1, 0.25, 0.25]

    data_name = data_struct.name
    ccatset = CCATDataset(path=data_struct.save_path, name=data_name, splits=splits, transforms=[train_transform])
    trainset = ccatset.get_split_n(0)
    testset = ccatset.get_split_n(1)
    n_classes = data_struct.n_classes
    img_size = data_struct.height
    n_in = data_struct.channels
    
    ## Define pyTorch ingestors for training and testing
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)
    #
    ## helper function to unnormalize and plot image 
    #def imshow(img):
    #    img = np.array(img)
    #    img = img / 2 + 0.5
    #    img = np.moveaxis(img, 0, -1)
    #    plt.imshow(img)
    #    
    ## display sample from dataset 
    #imgs,labels = iter(trainloader).next()
    #imshow(torchvision.utils.make_grid(imgs))  
    
    # Prepare the model for training
    model = get_predef_model(model_name)(n_in=n_in, n_classes=n_classes, n_filters=64, size=img_size)
    model.to(device)
    model.apply(weights_init)
    model_optim = optim.Adam(model.parameters(), lr=learnrate/10)

    # Train the model
    train(model=model, data_loader=trainloader, test_loader=testloader, 
          optimizer=model_optim, criterion=loss, n_epochs=n_epochs, classes=None, verbose=False)

    # Evaluate analytics on triaining and testing sets
    print("\nPerformance on training set: ")
    train_accuracy = eval_target_model(model, trainloader, classes=None)
    print("\nPerformance on test set: ")
    test_accuracy = eval_target_model(model, testloader, classes=None)


if __name__ == "__main__":
    main()
