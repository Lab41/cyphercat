import sys
sys.path.insert(0, '../../')
import cyphercat as cc

import torch
import torch.nn as nn
import numpy as np


class test_cnn(nn.Module): 
    def __init__(self, n_in=3, n_classes=10, n_filters=64, size=64): 
        super(test_cnn, self).__init__()
              
        self.size = size 
        self.n_filters = n_filters

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(n_in, n_filters, kernel_size=5, stride=1, padding=2),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(n_filters, 2*n_filters, kernel_size=5, stride=1,
                      padding=2),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ) 
        self.fc = nn.Linear(2*n_filters * (self.size//4) * (self.size//4),
                            2*n_filters)
        self.output = nn.Linear(2*n_filters, n_classes)
        
    def forward(self, x): 
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = x.view(x.size(0), -1)
        # x = x.view(-1, 2*self.n_filters * (self.size//4) * (self.size//4))
        x = self.fc(x)
        out = self.output(x)
        
        return out

    
def comparison(model, wrap, wrap2, image):
    print('Batch size = ', image.shape[0])
    print(' - Original model: returns full vector ')
    out = model(image)
    print('Batch labels = ', out.argmax(dim=1))
    print('Full label vectors\n', out)
    print(' - Wrapped 1 : returns top 3 ')
    out = wrap(image)
    print('Batch labels = ', out.argmax(dim=1))
    print('Full label vectors\n', out)
    print(' - Wrapped breaking probabilities : returns top 3 ')
    out = wrap2(image)
    print('Batch labels = ', out.argmax(dim=1))
    print('Full label vectors\n', out)

    
conv_net = test_cnn(size=32)
wrapped = cc.dimensionality_reduction(model=conv_net, n_top=3,
                                      break_posterior=False)
wrapped2 = cc.dimensionality_reduction(model=conv_net, n_top=3,
                                       break_posterior=True)

img = torch.randn((2, 3, 32, 32))
print(' ------- Training -------\n')
comparison(conv_net, wrapped, wrapped2, img)
print(' ------- Eval     -------\n')
conv_net.eval()
wrapped.eval()
wrapped2.eval()
comparison(conv_net, wrapped, wrapped2, img)
