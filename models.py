from torch import nn 
import torch.nn.functional as F


class mlleaks_cnn(nn.Module): 
    def __init__(self, n_in=3, n_out=10, n_hidden=128): 
        super(mlleaks_cnn, self).__init__()
        
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(n_in, 64, kernel_size=5, stride=1, padding=2), 
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2), 
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ) 
        self.fc = nn.Linear(128 * 8 * 8, 128)
        self.output = nn.Linear(128, n_out)
        
    def forward(self, x): 
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = x.view(-1, 128 * 8 * 8)
        x = self.fc(x)
        out = self.output(x)
        
        return out
    
class mlleaks_mlp(nn.Module): 
    def __init__(self, n_in=3, n_out=1, n_hidden=64): 
        super(mlleaks_mlp, self).__init__()
        
        self.hidden = nn.Linear(n_in, n_hidden)
        self.output = nn.Linear(n_hidden, n_out)
        
    def forward(self, x): 
        x = F.sigmoid(self.hidden(x))
        out = F.sigmoid(self.output(x))
        
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
        out = self.dense_block_1(x)
        ##x = self.dense_block_2(x)
        ##out = self.dense_block_3(x)

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

            
def weights_init(m): 
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear): 
        nn.init.xavier_normal_(m.weight.data)