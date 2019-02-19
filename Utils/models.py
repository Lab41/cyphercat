import torch
from torch import nn 
import torch.nn.functional as F
import numpy as np 
import os.path

def new_size_conv(size, kernel, stride=1, padding=0): 
    return np.floor((size + 2*padding - (kernel -1)-1)/stride +1)
    
    
def new_size_max_pool(size, kernel, stride=None, padding=0): 
    if stride == None: 
        stride = kernel
    return np.floor((size + 2*padding - (kernel -1)-1)/stride +1)

def calc_alexnet_size(size): 
    x = new_size_conv(size, 6,3,2)
    x = new_size_max_pool(x,3,2)
    x = new_size_conv(x,5,1,2)
    x = new_size_max_pool(x,3,2)
    x = new_size_conv(x,3,1,1)
    x = new_size_conv(x,3,1,1)
    x = new_size_conv(x,3,1,1)
    out = new_size_max_pool(x,2,2)
    
    return out

class AlexNet(nn.Module):
    def __init__(self, n_classes, size=32):
        super(AlexNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=6, stride=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        out_feat_size = calc_alexnet_size(size)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * out_feat_size * out_feat_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, n_classes),
        )
        
    def forward(self, x):

        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
class tiny_cnn(nn.Module): 
    def __init__(self, n_in=3, n_out=10, n_hidden=64, size=64): 
        super(tiny_cnn, self).__init__()
       
        
        self.size = size 
        self.n_hidden = n_hidden

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(n_in, n_hidden, kernel_size=5, stride=1, padding=2), 
            nn.BatchNorm2d(n_hidden), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(n_hidden, 2*n_hidden, kernel_size=5, stride=1, padding=2), 
            nn.BatchNorm2d(2*n_hidden), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ) 
        self.fc = nn.Linear(2*n_hidden * (self.size//4) * (self.size//4), 2*n_hidden)
        self.output = nn.Linear(2*n_hidden, n_out)
        
    def forward(self, x): 
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = x.view(x.size(0), -1)
        #x = x.view(-1, 2*self.n_hidden * (self.size//4) * (self.size//4))
        x = self.fc(x)
        out = self.output(x)
        
        return out
    
def calc_mlleaks_cnn_size(size): 
    x = new_size_conv(size, 5,1,2)
    x = new_size_max_pool(x,2,2)
    x = new_size_conv(x,5,1,2)
    out = new_size_max_pool(x,2,2)
    
    return out

class mlleaks_cnn(nn.Module): 
    def __init__(self, n_in=3, n_out=10, n_hidden=64, size=32): 
        super(mlleaks_cnn, self).__init__()
        
        self.n_hidden = n_hidden 
        
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(n_in, n_hidden, kernel_size=5, stride=1, padding=2), 
            nn.BatchNorm2d(n_hidden), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(n_hidden, 2*n_hidden, kernel_size=5, stride=1, padding=2), 
            nn.BatchNorm2d(2*n_hidden), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ) 

        fc_feature_size = calc_mlleaks_cnn_size(size)
        self.fc = nn.Linear(int(2*n_hidden * fc_feature_size * fc_feature_size), 128)
        self.output = nn.Linear(2*n_hidden, n_out)
        
    def forward(self, x): 
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        out = self.output(x)
        
        return out

class ConvBlock(nn.Module):
    #for audio_CNN_classifier
    def __init__(self, n_input, n_out, kernel_size):
        super(ConvBlock, self).__init__()
        self.cnn_block = nn.Sequential(
            nn.Conv1d(n_input, n_out, kernel_size, padding=1),
            nn.BatchNorm1d(n_out),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )
    
    def forward(self, x):
        return self.cnn_block(x)


class audio_CNN_classifier(nn.Module):
    def __init__(self, in_size, n_hidden, n_classes):
        super(audio_CNN_classifier, self).__init__()
        self.down_path = nn.ModuleList()
        self.down_path.append(ConvBlock(in_size, 2*in_size, 3))
        self.down_path.append(ConvBlock(2*in_size, 4*in_size, 3))
        self.down_path.append(ConvBlock(4*in_size, 8*in_size, 3))
        self.fc = nn.Sequential(
            nn.Linear(8*in_size, n_hidden),
            nn.ReLU()
        )
        self.out = nn.Linear(n_hidden, n_classes)
    def forward(self, x):
        for down in self.down_path:
            x = down(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.out(x)
    
class STFT_CNN_classifier(nn.Module):
    def __init__(self, in_size, n_hidden, n_classes):
        super(STFT_CNN_classifier, self).__init__()
        self.down_path = nn.ModuleList()
        self.down_path.append(ConvBlock(in_size, in_size, 7))
        self.down_path.append(ConvBlock(in_size, in_size*2, 7))
        self.down_path.append(ConvBlock(in_size*2, in_size*4, 7))
        self.fc = nn.Sequential(
            nn.Linear(5264, n_hidden),
            nn.ReLU()
        )
        self.out = nn.Linear(n_hidden, n_classes)
    def forward(self, x):
        for down in self.down_path:
            x = down(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.out(x)
        
    
class mlleaks_mlp(nn.Module): 
    def __init__(self, n_in=3, n_out=1, n_hidden=64): 
        super(mlleaks_mlp, self).__init__()
        
        self.hidden = nn.Linear(n_in, n_hidden)
        #self.bn = nn.BatchNorm1d(n_hidden)
        self.output = nn.Linear(n_hidden, n_out)
        
    def forward(self, x): 
        x = F.sigmoid(self.hidden(x))
        #x = self.bn(x)
        out = self.output(x)
        #out = F.sigmoid(self.output(x))
        
        return out
    

class cnn(nn.Module): 
    def __init__(self, in_channels, out_channels, n_filters): 
        super(cnn, self).__init__()
        
        self.n_filters = n_filters 
        
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels, n_filters, kernel_size=3, padding=1), 
            nn.BatchNorm2d(n_filters), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2)
        ) 
        # shape = [Batch_size, n_filters, height/2, width/2]
            
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(n_filters, n_filters*2, kernel_size=3, padding=1), 
            nn.BatchNorm2d(n_filters*2), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2)
        ) 
        # shape = [Batch_size, n_filters*2, height/4, width/4] 
        
        self.dense_block_1 = nn.Sequential(
            ##nn.Linear(n_filters * 2 * 8 * 8, 64), 
            nn.Linear(n_filters*2 * 8 * 8, 128), 
            ##nn.BatchNorm1d(64), 
            ##nn.ReLU(inplace=True)
        ) 
        # shape = [Batch_size, 64]
        
        self.dense_block_2 = nn.Sequential(
            nn.Linear(64, 32), 
            nn.BatchNorm1d(32), 
            nn.ReLU(inplace=True)
        ) 
        # shape = [Batch_size, 32]
        
        self.dense_block_3 = nn.Sequential( 
            nn.Linear(32, out_channels), 
            nn.BatchNorm1d(out_channels)
        ) 
        # shape = [Batch_size, 10]
        
        
    def forward(self, x): 
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)

        x = x.view(-1, self.n_filters*2 * 8 * 8)
        x = self.dense_block_1(x)
        x = self.dense_block_2(x)
        out = self.dense_block_3(x)

        return out
        
        
class mlp(nn.Module): 
    def __init__(self, in_channels, out_channels, n_filters): 
        super(mlp, self).__init__()
        
        self.n_filters = n_filters 
        
        # shape = [Batch_size, k (top k posteriors)] 
        
        self.dense_block_1 = nn.Sequential(
            nn.Linear(in_channels, n_filters*2), 
            #nn.BatchNorm1d(n_filters*2), 
            nn.ReLU(inplace=True)
        ) 
        # shape = [Batch_size, n_filters*2]
        
        self.dense_block_2 = nn.Sequential(
            nn.Linear(n_filters*2, n_filters*2), 
            #nn.BatchNorm1d(n_filters*2), 
            nn.ReLU(inplace=True)
        ) 
        # shape = [Batch_size, 32]
        
        self.dense_block_3 = nn.Sequential( 
            nn.Linear(n_filters*2, out_channels), 
            #nn.BatchNorm1d(out_channels), 
            nn.Sigmoid()
        ) 
        # shape = [Batch_size, 10]
        
        
    def forward(self, x): 

        x = self.dense_block_1(x)
        x = self.dense_block_2(x)
        out = self.dense_block_3(x)

        return out


class audio_cnn_block(nn.Module):
    '''
    1D convolution block used to build audio cnn classifiers
    Args:
    input: input channels
    output: output channels
    kernel_size: convolution kernel size
    '''
    def __init__(self, n_input, n_out, kernel_size):
        super(audio_cnn_block, self).__init__()
        self.cnn_block = nn.Sequential(
            nn.Conv1d(n_input, n_out, kernel_size, padding=1),
            nn.BatchNorm1d(n_out),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )
    
    def forward(self, x):
        return self.cnn_block(x)


class audio_tiny_cnn(nn.Module):
    '''
    Template for convolutional audio classifiers.
    '''
    def __init__(self, cnn_sizes, n_hidden, kernel_size, n_classes):
        '''
        Init
        Args: 
        cnn_sizes: List of sizes for the convolution blocks
        n_hidden: number of hidden units in the first fully connected layer
        kernel_size: convolution kernel size
        n_classes: number of speakers to classify
        '''
        super(audio_tiny_cnn, self).__init__()
        self.down_path = nn.ModuleList()
        self.down_path.append(audio_cnn_block(cnn_sizes[0], cnn_sizes[1],
                                              kernel_size,))
        self.down_path.append(audio_cnn_block(cnn_sizes[1], cnn_sizes[2],
                                              kernel_size,))
        self.down_path.append(audio_cnn_block(cnn_sizes[2], cnn_sizes[3],
                                              kernel_size,))
        self.fc = nn.Sequential(
            nn.Linear(cnn_sizes[4], n_hidden),
            nn.ReLU()
        )
        self.out = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        for down in self.down_path:
            x = down(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.out(x)


def MFCC_cnn_classifier(n_classes):
    '''
    Builds speaker classifier that ingests MFCC's
    '''
    in_size = 20
    n_hidden = 512
    sizes_list = [in_size, 2*in_size, 4*in_size, 8*in_size, 8*in_size]
    return audio_tiny_cnn(cnn_sizes=sizes_list, n_hidden=n_hidden,
                          kernel_size=3, n_classes=125)


def ft_cnn_classifer(n_classes):
    '''
    Builds speaker classifier that ingests the abs value of fourier transforms
    '''
    in_size = 94
    n_hidden = 512
    sizes_list = [in_size, in_size, 2*in_size, 4*in_size, 14*4*in_size]
    return audio_tiny_cnn(cnn_sizes=sizes_list, n_hidden=n_hidden,
                          kernel_size=7, n_classes=125)


class RNN(torch.nn.Module):
    '''
    Bidirectional LSTM for sentiment analysis
    '''
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, n_layers=2, bidirectional=True, dropout=0.5):
        super(RNN, self).__init__()
        
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size)
        self.rnn = torch.nn.LSTM(embedding_size, hidden_size, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
        self.fc = torch.nn.Linear(hidden_size*2, output_size)
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x):
        
        embedded = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.rnn(embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))

        return self.fc(hidden.squeeze(0))


def weights_init(m): 
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear): 
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias, 0)

def save_checkpoint(model=None, optimizer=None, epoch=None,
                    data_descriptor=None, loss=None, accuracy=None, path='./',
                    filename='checkpoint', ext='.pth.tar'):
    state = {
        'epoch': epoch,
        'arch': str(model.type),
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'dataset': data_descriptor
        }
    torch.save(state, path+filename+ext)


def load_checkpoint(model=None, optimizer=None,  checkpoint=None):
    assert os.path.isfile(checkpoint), 'Checkpoint not found, aborting load'
    chpt = torch.load(checkpoint)
    assert str(model.type) == chpt['arch'], 'Model arquitecture mismatch,\
  aborting load'
    model.load_state_dict(chpt['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict['optimizer']
    print('Succesfully loaded checkpoint \nDataset: %s \nEpoch: %s \nLoss: %s\
\nAccuracy: %s' % (chpt['dataset'], chpt['epoch'], chpt['loss'],
                   chpt['accuracy']))
