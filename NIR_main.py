
from GlyphDataset import HistoricalGlyphDataset
from NIR_model import NeuralSignedDistanceModel

from torch.utils.data import DataLoader,random_split
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
from skimage import io
import matplotlib.pyplot as plt

#env name : ptorch

path="/Users/poojashetty/Documents/AI_SS23/Project/NeuralImplicitRepr/NIR/synthetic_curated_image_dump"
model_save_path="/Users/poojashetty/Documents/AI_SS23/Project/NeuralImplicitRepr/NIR/Model/model.pt"

def generateSkeleton(img_path,model,device,transform):
    model.eval()
    image=io.imread(img_path)
    skel_img=np.zeros_like(image)
    model.to(device)
    image=transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                pixel_coord=torch.tensor([i,j]).unsqueeze(0).to(device)
                skel_img[i][j] = model(image,pixel_coord)

    plt.imshow(skel_img)
    plt.show()


def test(dataloader,model,loss_fn,optimizer,device):
    size = len(dataloader.dataset)
    num_batches=len(dataloader)
    model.eval()
    test_loss=0
    with torch.no_grad():
        for sample in dataloader:
            image,pixel_coord=sample['image'].to(device),sample['pixel_coord'].to(device)
            pred = model(image,pixel_coord)
            test_loss += loss_fn(pred,sample['sdv']).item()
    test_loss /= num_batches
    print(f"Test loss : {test_loss}")


def train(dataloader,model,loss_fn,optimizer,device):
    size=len(dataloader.dataset)
    model.train()
    for batch,sample in enumerate(dataloader):
        #print(f"batch {batch}, sample {sample}\n")
        image,pixel_coord=sample['image'].to(device),sample['pixel_coord'].to(device)

        optimizer.zero_grad()
        #print(sample['sdv'])
        pred=model(image,pixel_coord)
        loss=loss_fn(pred,sample['sdv'])

        loss.backward()
        optimizer.step()

        if batch %100 == 0:
            print(f"Batch {batch} -- loss: {loss.item()}")


def run(device,transform):
    n=10
    historicalGlyphDataset = HistoricalGlyphDataset(data_dir=path,n=n,transform=transform)
    train_size = int(0.7*len(historicalGlyphDataset))
    test_size = len(historicalGlyphDataset)-train_size

    train_subset,test_subset=random_split(historicalGlyphDataset,[train_size,test_size])

    batch_size=64

    trainGlyphDataloader=DataLoader(train_subset,batch_size=batch_size,shuffle=True)
    testGlyphDataloader=DataLoader(test_subset,batch_size=batch_size,shuffle=True)

    model = NeuralSignedDistanceModel()
    model.to(device)

    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(),lr=1e-3)

    epochs=5

    for i in range(epochs):
        print(f"Epoch {i}------\n")
        train(trainGlyphDataloader,model,loss_fn,optimizer,device)
        test(testGlyphDataloader,model,loss_fn,optimizer,device)

    torch.save(model.state_dict(),model_save_path)
    #print(historicalGlyphDataset[0]['image'])
    #plt.imshow(historicalGlyphDataset[3]['image'][0])
    #plt.show()

if __name__=='__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    transform=transforms.Compose([
            transforms.ToTensor()
        ])
    run(device,transform)
    #inference
    model=NeuralSignedDistanceModel()
    model.load_state_dict(torch.load(model_save_path))
    img_path="/Users/poojashetty/Documents/AI_SS23/Project/NeuralImplicitRepr/synthetic_curated_image_dump/img/Fust & Schoeffer Durandus Gotico-Antiqua 118G_C.ss01_12.png"
    generateSkeleton(img_path,model,device,transform)
    