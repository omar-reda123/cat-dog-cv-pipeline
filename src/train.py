#imports
import torch
import torch.nn as nn
import torch.optim as optim
from src.model import CatDogModel
from src.dataset import train_loader,device
from pathlib import Path

#initializations
model=CatDogModel().to(device=device)
criterion=nn.CrossEntropyLoss()
parameters = model.model.classifier.parameters()
lr=0.01
optimizer=optim.AdamW(params=parameters,lr=lr)
epochs=10

for epoch in range(epochs):
    for batch_idx,(images,labels) in enumerate(train_loader):
        #data
        images=images.to(device)
        labels=labels.to(device)
        #train loop
        optimizer.zero_grad()
        pred=model(images)
        loss=criterion(pred,labels)
        loss.backward()
        optimizer.step()
        #notice
        if batch_idx%10==0:    
            print(f"batch {batch_idx}:loss {loss}")

#saving model
models_dir = Path("models")
models_dir.mkdir(parents=True, exist_ok=True)  
model_path = models_dir / "cat_dog_model.pth"
torch.save(model.state_dict(),model_path)

        
        
        
    

