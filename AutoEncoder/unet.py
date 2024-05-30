import torch.nn as nn
import torch 

class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv_block= nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self,x):
        x=self.conv_block(x)
        return x
    
class DownSample(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels,out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self,x):
        x= self.conv(x)
        p= self.pool(x)

        return x,p
    
class UpSample(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2, stride=2)
        self.conv= ConvBlock(out_channels+out_channels,out_channels)

    def forward(self,x,skip):
        x=self.up(x)
        x=torch.cat([x,skip],axis=1)
        x=self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.skip=dict()

        self.enc1 = DownSample(1,64)
        self.enc2 = DownSample(64,128)
        self.enc3 = DownSample(128,256)
        self.enc4 = DownSample(256,512)

        self.bottleneck= ConvBlock(512,1024)

        self.dec1 = UpSample(1024,512)
        self.dec2 = UpSample(512,256)
        self.dec3 = UpSample(256,128)
        self.dec4 = UpSample(128,64)

        self.output= nn.Conv2d(64,1,kernel_size=1)

    def encoder(self,x):
        skip1,p1= self.enc1(x)
        self.skip['skip1']= skip1
        skip2,p2= self.enc2(p1)
        self.skip['skip2']= skip2
        skip3,p3= self.enc3(p2)
        self.skip['skip3']= skip3
        skip4,p4= self.enc4(p3)
        self.skip['skip4']= skip4

        return p4
    
    def decoder(self,x):
        d1= self.dec1(x,self.skip['skip4'])
        d2= self.dec2(d1,self.skip['skip3'])
        d3= self.dec3(d2,self.skip['skip2'])
        d4= self.dec4(d3,self.skip['skip1'])

        return d4

    def forward(self,x):
        
        e= self.encoder(x)
        b= self.bottleneck(e)
        d=self.decoder(b)
        output= self.output(d)

        return output






