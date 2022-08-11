import torch 
import torch.nn as nn
from model.ECAPA_TDNN.MFA.FreqChannelAtten import FreqChannelAtten as MFA

class DpMsModule(nn.Module):
    def __init__(self, scale, channel_output, input_dim):
        super(DpMsModule, self).__init__()
        self.cnn1 = nn.Conv2d(1, channel_output, kernel_size=3, padding=(1,1))
        self.cnn2 = nn.Conv2d(channel_output, channel_output, kernel_size=3, padding=(1,1))
        self.width = channel_output//scale
        self.scale = scale

        self.mfa1 = MFA(self.width, input_dim)
        self.mfa2 = MFA(self.width, input_dim)
        self.mfa3 = MFA(self.width, input_dim)
        self.mfa4 = MFA(self.width, input_dim)

        self.cnn3_1 = nn.Conv2d(self.width, self.width, kernel_size=3, padding=(1,1))
        self.cnn3_2 = nn.Conv2d(self.width, self.width, kernel_size=3, padding=(1,1))
        self.cnn3_3 = nn.Conv2d(self.width, self.width, kernel_size=3, padding=(1,1))

        self.flatten = nn.Flatten(1,2)
        
        self.cnn4 = nn.Conv1d(channel_output, channel_output, kernel_size=1)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x1, x2, x3, x4 = torch.split(x, self.width, dim=1)

        x1 = self.mfa1(x1)
        x2 = self.cnn3_1(x2)
        x2_ = x2
        x2 = self.mfa2(x1*x2)
        
        x3 = self.cnn3_2(x2_+x3)
        x3_ = x3
        x3 = self.mfa3(x2*x3)
        
        x4 = self.cnn3_3(x3_+x4)
        x4 = self.mfa4(x3*x4)


        y = torch.cat((x1,x2,x3,x4), 1)
        y = self.flatten(y)
        y_ = y
        y= self.cnn4(y)
        y= y+y_

        #batch, c, d, l = x1.size()
        
        return y


if __name__ == "__main__":
    import torch
    test_input  = torch.rand(1, 1, 80, 200) #B X C X D X L
    
    model = DpMsModule(4,32,80)
    model.eval()
    test_output = model(test_input)
    print(test_output.shape)