import torch as K
from torch import nn 
from torch.nn import functional as F

from lalnets2.gar import acolPool

class Conv3LwACOL(nn.Module):
    def __init__(self, n_p=2, k=5, n_channels=3, n_filter=64):
        super(Conv3LwACOL, self).__init__()
        
        self.n_filter = n_filter
        
        self.BN1 = nn.BatchNorm2d(n_channels)
        self.CV11 = nn.Conv2d(n_channels, n_filter, 3, padding=1)
        self.CV12 = nn.Conv2d(n_filter, n_filter, 3, padding=1)
        self.MP1 = nn.MaxPool2d(2, 2)
        self.DR1 = nn.Dropout2d(0.2)
        
        self.BN2 = nn.BatchNorm2d(n_filter)
        self.CV21 = nn.Conv2d(n_filter, n_filter*2, 3, padding=1)
        self.CV22 = nn.Conv2d(n_filter*2, n_filter*2, 3, padding=1)
        self.MP2 = nn.MaxPool2d(2, 2)
        self.DR2 = nn.Dropout2d(0.3)
        
        self.BN3 = nn.BatchNorm2d(n_filter*2)
        self.CV31 = nn.Conv2d(n_filter*2, n_filter*4, 3, padding=1)
        self.CV32 = nn.Conv2d(n_filter*4, n_filter*4, 3, padding=1)
        self.CV33 = nn.Conv2d(n_filter*4, n_filter*4, 3, padding=1)
        self.MP3 = nn.MaxPool2d(2, 2)
        self.DR3 = nn.Dropout2d(0.4)
        
        self.BN4 = nn.BatchNorm1d(n_filter*4 * 4 * 4)
        self.FC4 = nn.Linear(n_filter*4 * 4 * 4, 2048)
        self.DR4 = nn.Dropout(0.5)
        
        self.BN5 = nn.BatchNorm1d(2048)
        self.FC5 = nn.Linear(2048, k * n_p)
        self.AP5 = acolPool(k, n_p)
        
    def forward(self, x, return_latent=False):
        x = self.BN1(x)
        x = F.relu(self.CV11(x))
        x = F.relu(self.CV12(x))
        x = self.DR1(self.MP1(x))
        
        x = self.BN2(x)
        x = F.relu(self.CV21(x))
        x = F.relu(self.CV22(x))
        x = self.DR2(self.MP2(x))
        
        x = self.BN3(x)
        x = F.relu(self.CV31(x))
        x = F.relu(self.CV32(x))
        x = F.relu(self.CV33(x))
        x = self.DR3(self.MP3(x))
        
        x = x.view(-1, self.n_filter * 4 * 4 * 4)
        
        x = self.BN4(x)
        x = F.relu(self.FC4(x)) # F
        f = x if return_latent else None
        x = self.DR4(x)        
            
        x = self.BN5(x)
        x = self.FC5(x) #Z
        z = x
        x = F.softmax(x, dim=1)
        s = x if return_latent else None
        x = K.matmul(x, self.AP5) # Y
        
        return x, f, z, s