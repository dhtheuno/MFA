import torch 
import torch.nn as nn

class FreqChannelAtten(nn.Module):
    def __init__(self, input_channel, input_dim):
        super(FreqChannelAtten, self).__init__()
        
        self.width = input_channel * input_dim
        
        #Global Average Pooling along the Frame Lengths
        #self.gap = nn.AdaptiveAvgPool3d((None, None,1))
        #Replace the GAP with torch.mean in Frames (dim=3)


        #Flatten the GAP (Channel x Dim(logmelfilterbank))
        self.flatten = nn.Flatten()
        self.layer = nn.Linear(self.width, self.width)
    
        self.flatten_tdnn = nn.Flatten(1,2)
        
        #TDNN with Conv1d
        self.cnn = nn.Conv1d(self.width, input_channel, kernel_size=1)
        self.relu = nn.ReLU()       
        self.bn = nn.BatchNorm1d(input_channel)
        
    
    def forward(self, x):
        """
        input: Batch x Channel x Dim(mel filter bank) x frames
        output: Batch x Channel X Dim x frames 
        """
        x_ = x
        #x = self.gap(x)
        x = torch.mean(x, dim=3)

        x = x.squeeze(-1)
        b,c,d = x.size()
        
        x = self.flatten(x)
        x = self.layer(x)
        
        x = x.reshape(-1, c, d)
        x = x.unsqueeze(-1)*x_
        
        x = self.flatten_tdnn(x)
        
        x = self.cnn(x)
        
        x = self.relu(x)
        x = self.bn(x)
        x = x.unsqueeze(2)
        
        return x