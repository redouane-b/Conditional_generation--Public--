import os
import numpy as np
import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

k=5
taille=800

data=np.load("Code_python/data/data_latents_for_5.npy",allow_pickle=True)

X=[]
Y=[]
Z=[]

for i,elem in enumerate(data):
    x,label=elem
    if label["gender"]=="female":
        X.append(x)
        Y.append(0)
    elif label["gender"]=="male":
        X.append(x)
        Y.append(1)

    if label["age_group"]=='0-2' or label["age_group"]=='3-6':
        Z.append([1,0,0,0])

    if label["age_group"]=='7-9' or label["age_group"]=='10-14':
        Z.append([0,1,0,0])
        
    if label["age_group"]=='15-19' or label["age_group"]=='20-29' or label["age_group"]=='30-39' or label["age_group"]=='40-49':
        Z.append([0,0,1,0])

    if label["age_group"]=='50-69' or label["age_group"]=='70-120':
        Z.append([0,0,0,1])


coord=[]
for x1 in X:
  ze=[]
  for x2 in x1:
    ze+=torch.flatten(x2).tolist()
  coord.append(ze)



x_train, x_test, y_train, y_test= train_test_split(coord, Y, train_size=0.8)



dataset=[(torch.tensor(x),torch.tensor(y)) for x,y in zip(x_train,y_train) ]


class MLP(pl.LightningModule):
  
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(taille, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 1),
      nn.Sigmoid()
    )
    self.ce = nn.BCELoss()
    
  def forward(self, x):
    return self.layers(x)
  
  def training_step(self, batch, batch_idx):
    x, y = batch
    x = x.view(x.size(0), -1)
    y_hat = self.layers(x)
    loss = self.ce(y_hat, y.unsqueeze(1).float())
    self.log('train_loss', loss)
    return loss
  
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
    return optimizer
  
  
if __name__ == '__main__':
  mlp = MLP()
  trainer = pl.Trainer(gpus=0, deterministic=True, max_epochs=11)
  trainer.fit(mlp, DataLoader(dataset,batch_size=64,num_workers=12))


def score_test(x,y):
  prediction = mlp(torch.tensor(x,dtype=torch.float32))
  print(y)
  print(prediction)
  acc = ((prediction.detach().numpy().round()==y).mean())
  return acc

print("score =",score_test(x_test,y_test))