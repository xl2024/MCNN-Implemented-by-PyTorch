from torch import nn
import torch

class MCNN(nn.Module):

    def __init__(self):
        super(MCNN, self).__init__()

        self.column1=nn.Sequential(
            nn.Conv2d(3,16,9,padding=4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,7,padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,16,7,padding=3),
            nn.ReLU(),
            nn.Conv2d(16,8,7,padding=3),
            nn.ReLU()
        )

        self.column2=nn.Sequential(
            nn.Conv2d(3,20,7,padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(20,40,5,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(40,20,5,padding=2),
            nn.ReLU(),
            nn.Conv2d(20,10,5,padding=2),
            nn.ReLU()
        )

        self.column3=nn.Sequential(
            nn.Conv2d(3,24,5,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24,48,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48,24,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(24,12,3,padding=1),
            nn.ReLU()
        )
        
        self.conv=nn.Conv2d(30,1,1)


    def forward(self, input):
        
        col1=self.column1(input)
        col2=self.column2(input)
        col3=self.column3(input)
        cols=torch.cat((col1,col2,col3),1)
        den_map=self.conv(cols)

        return den_map


if __name__=='__main__':
    mcnn=MCNN()
    print(mcnn)
