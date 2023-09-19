import torch.nn as nn
import torch 
import torch.nn.functional as F

class SineLayer(nn.Module):
    def __init__(self,in_features,out_features,omega_0=30):
        super().__init__()
        self.omega_0=omega_0
        self.in_features=in_features
        self.out_features=out_features
        self.linear=nn.Linear(self.in_features,self.out_features)

        with torch.no_grad():
            self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)

    def forward(self,x):
        return torch.sin(self.omega_0*self.linear(x))

class NeuralSignedDistanceModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden=32
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=self.hidden,kernel_size=5,stride=2,padding=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.hidden,out_channels=self.hidden*2,kernel_size=5,stride=2,padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.hidden*2,out_channels=self.hidden*3,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.hidden*3,out_channels=self.hidden*4,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.hidden*4,out_channels=self.hidden*4,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.hidden*4,out_channels=self.hidden*4,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3200,64)
        )

        self.siren=nn.Sequential(
            SineLayer(66,32),
            SineLayer(32,10),
            nn.Linear(10,1)
        )

    def forward(self,x,pixel_coord):
        output=self.layers(x)
        output=torch.cat((pixel_coord,output),dim=1)
        output=self.siren(output)
        #print(f"Out shape {output.shape}")
        return output
