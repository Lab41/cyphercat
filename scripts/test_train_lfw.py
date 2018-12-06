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
    
    from cyphercat.load_data import get_split_dataset
    from cyphercat.utils import Configurator, ModelConfig

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

    # Hyperparameters
    n_epochs = train_config.epochs
    batch_size = train_config.batchsize
    learnrate = train_config.learnrate
    loss = nn.CrossEntropyLoss()
    
    # Data augmentation 
    train_transform = torchvision.transforms.Compose([
        #torchvision.transforms.RandomRotation(10),
        #torchvision.transforms.RandomHorizontalFlip(),
        #torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
     
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_transform = torchvision.transforms.Compose([
        #torchvision.transforms.Pad(2),
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
        
    # Defined training and testing set splits 
    trainset, testset = get_split_dataset(dataset_config=dataset_config, transforms=[train_transform, test_transform])
    n_classes = trainset.n_classes
    
    # Define pyTorch ingestors for training and testing
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)
    
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
    
    #print("Made it to training and testing section")
    #sys.exit(0)
    
    ## Train and test CNN
    #conv_model = tiny_cnn(n_in=3, n_classes=n_classes, n_filters=32, size=250).to(device)
    #
    #conv_model.apply(weights_init)
    #
    #conv_optim = optim.Adam(conv_model.parameters(), lr=learnrate)
    #
    #train(conv_model, trainloader, testloader, conv_optim, loss, n_epochs, verbose=False)
    #
    #print("\nPerformance on training set: ")
    #train_accuracy = eval_target_model(conv_model, trainloader, classes=None)
    #
    #print("\nPerformance on test set: ")
    #test_accuracy = eval_target_model(conv_model, testloader, classes=None)
    #
    #
    #
    ## Train and test ResNet18
    ## load the torchvision resnet18 implementation 
    #resnet18 = torchvision.models.resnet18(num_classes=n_classes).to(device)
    #resnet18.fc = nn.Linear(2048, n_classes)
    #resnet18.to(device)
    #
    #resnet18.apply(weights_init)
    #
    #resnet18_optim = optim.Adam(resnet18.parameters(), lr=learnrate)
    #
    #train(resnet18, trainloader, testloader, resnet18_optim, loss, n_epochs, verbose=False)
    #
    #print("\nPerformance on training set: ")
    #train_accuracy = eval_target_model(resnet18, trainloader, classes=None)
    #
    #print("\nPerformance on test set: ")
    #test_accuracy = eval_target_model(resnet18, testloader, classes=None)
    #
    #
    ## Train and test VGG16
    #vgg16 = torchvision.models.vgg16(num_classes=n_classes).to(device)
    #
    #vgg16.apply(weights_init)
    #
    #
    #vgg16_optim = optim.SGD(vgg16.parameters(), lr=learnrate)
    ##vgg16_optim = optim.Adam(vgg16.parameters(), lr=learnrate)
    #
    #train(vgg16, trainloader, testloader, vgg16_optim, loss, n_epochs, verbose=False)
    #
    #print("\nPerformance on training set: ")
    #train_accuracy = eval_target_model(vgg16, trainloader, classes=None)
    #
    #print("\nPerformance on test set: ")
    #test_accuracy = eval_target_model(vgg16, testloader, classes=None)
    #
    ## Train and test AlexNet
    #alexnet = AlexNet(n_in=3, n_classes=n_classes, n_filters=64, size=250).to(device)
    #
    #alexnet.apply(weights_init)
    #
    #alexnet_optim = optim.Adam(alexnet.parameters(), lr=learnrate/10)
    #
    #train(alexnet, trainloader, testloader, alexnet_optim, loss, n_epochs, verbose=False)
    #
    #print("\nPerformance on training set: ")
    #train_accuracy = eval_target_model(alexnet, trainloader, classes=None)
    #
    #print("\nPerformance on test set: ")
    #test_accuracy = eval_target_model(alexnet, testloader, classes=None)

    
    # Prepare the model for training
    model = get_predef_model(model_name)(n_in=3, n_classes=n_classes, n_filters=64, size=250)
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
