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
    
    from cyphercat.load_data import prep_data
    from cyphercat.utils import Configurator, ModelConfig, DataStruct

except ImportError as e:
    print(e)
    raise ImportError

    
    
def main():
    global args
    parser = argparse.ArgumentParser(description="Convolutional NN Testing Script")
    parser.add_argument("-c", "--config", dest="configfile", default='scripts/configs/librispeech.yml', help="Path to yaml")

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
    
    # Datastruct for prepping data 
    data_struct = DataStruct(dataset_config)
    # Simple download / unpacker function
    prep_data(data_struct)


    ## Training model params
    #train_config = ModelConfig(train_model_config)
    #model_name = train_config.name

    ## Hyperparameters
    #n_epochs = train_config.epochs
    #batch_size = train_config.batchsize
    #learnrate = train_config.learnrate
    #loss = nn.CrossEntropyLoss()
    #
    ## Data augmentation 
    #train_transform = torchvision.transforms.Compose([
    #    #torchvision.transforms.RandomRotation(10),
    #    #torchvision.transforms.RandomHorizontalFlip(),
    #    #torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    # 
    #    torchvision.transforms.ToTensor(),
    #    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #])
    #
    #test_transform = torchvision.transforms.Compose([
    #    #torchvision.transforms.Pad(2),
    #    torchvision.transforms.ToTensor(),
    #    #torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    #])
    #    
    ## Defined training and testing set splits 
    #trainset, testset = get_split_dataset(dataset_config=dataset_config, transforms=[train_transform, test_transform])
    #n_classes = trainset.n_classes
    #
    ## Define pyTorch ingestors for training and testing
    #trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)
    #
    ## Prepare the model for training
    #model = get_predef_model(model_name)(n_in=3, n_classes=n_classes, n_filters=64, size=250)
    #model.to(device)
    #model.apply(weights_init)
    #model_optim = optim.Adam(model.parameters(), lr=learnrate/10)

    ## Train the model
    #train(model=model, data_loader=trainloader, test_loader=testloader, 
    #      optimizer=model_optim, criterion=loss, n_epochs=n_epochs, classes=None, verbose=False)

    ## Evaluate analytics on triaining and testing sets
    #print("\nPerformance on training set: ")
    #train_accuracy = eval_target_model(model, trainloader, classes=None)
    #print("\nPerformance on test set: ")
    #test_accuracy = eval_target_model(model, testloader, classes=None)


if __name__ == "__main__":
    main()
