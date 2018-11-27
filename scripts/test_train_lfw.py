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
    
    from cyphercat.load_data import prep_data, LFWDataset
    from cyphercat.utils import Configurator, DataStruct, ModelConfig

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

    # Directory structures for data and model saving
    data_struct = DataStruct(dataset_config)

    # Load LFW
    prep_data(dataset_config)

    # Data set directory
    data_dir = data_struct.save_path
   
    # Training model params
    train_config = ModelConfig(train_model_config)
    model_name = train_config.name

    # Hyperparameters
    n_epochs = train_config.epochs
    batch_size = train_config.batchsize
    learnrate = train_config.learnrate
    loss = nn.CrossEntropyLoss()
    
    img_paths = []
    for p in os.listdir(data_dir): 
        for i in os.listdir(os.path.join(data_dir,p)): 
            img_paths.append(os.path.join(data_dir,p,i))
            
    class_list = []
    class_to_idx = {}
    k = 0 
    for i in img_paths: 
        name = i.split('/')[-2]
        if name not in class_to_idx: 
            class_list.append(name)
            class_to_idx[name] = k
            k += 1
    
    
    n_classes = len(class_list)
    
    img_paths = np.random.permutation(img_paths)
    
    dataset_size = len(img_paths)
    
    trainset_size = int(0.8 * dataset_size)
    
    trainset_list = img_paths[:trainset_size]
    testset_list = img_paths[trainset_size:]
    
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
        
    
    trainset = LFWDataset(trainset_list, class_to_idx, transform=train_transform)
    testset = LFWDataset(testset_list, class_to_idx, transform=test_transform)
    
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
    #conv_net = tiny_cnn(n_in=3, n_classes=n_classes, n_filters=32, size=250).to(device)
    #
    #conv_net.apply(weights_init)
    #
    #conv_optim = optim.Adam(conv_net.parameters(), lr=learnrate)
    #
    #train(conv_net, trainloader, testloader, conv_optim, loss, n_epochs, verbose=False)
    #
    #print("\nPerformance on training set: ")
    #train_accuracy = eval_target_net(conv_net, trainloader, classes=None)
    #
    #print("\nPerformance on test set: ")
    #test_accuracy = eval_target_net(conv_net, testloader, classes=None)
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
    #train_accuracy = eval_target_net(resnet18, trainloader, classes=None)
    #
    #print("\nPerformance on test set: ")
    #test_accuracy = eval_target_net(resnet18, testloader, classes=None)
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
    #train_accuracy = eval_target_net(vgg16, trainloader, classes=None)
    #
    #print("\nPerformance on test set: ")
    #test_accuracy = eval_target_net(vgg16, testloader, classes=None)
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
    #train_accuracy = eval_target_net(alexnet, trainloader, classes=None)
    #
    #print("\nPerformance on test set: ")
    #test_accuracy = eval_target_net(alexnet, testloader, classes=None)

    
    # Train and test model
    model = get_predef_model(model_name)(n_in=3, n_classes=n_classes, n_filters=64, size=250)
    model.to(device)
    model.apply(weights_init)
    model_optim = optim.Adam(model.parameters(), lr=learnrate/10)
    train(model, trainloader, testloader, model_optim, loss, n_epochs, verbose=False)
    print("\nPerformance on training set: ")
    train_accuracy = eval_target_net(model, trainloader, classes=None)
    
    print("\nPerformance on test set: ")
    test_accuracy = eval_target_net(model, testloader, classes=None)



if __name__ == "__main__":
    main()
