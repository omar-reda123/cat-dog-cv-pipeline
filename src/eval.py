# imports
import torch
import torch.nn as nn
from pathlib import Path
from src.model import CatDogModel
from src.dataset import val_loader, device

# init
state_dict_path = Path("models") / "cat_dog_model.pth"
model = CatDogModel().to(device=device)  
model.load_state_dict(torch.load(state_dict_path, map_location=device))

# eval mode
model.eval()

# eval loop
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        # data
        images = images.to(device)
        labels = labels.to(device)
        
        # predictions
        outs = model(images)  # [batch_size, 2]
        
        # eval
        _, predicted = torch.max(outs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Validation Accuracy: {accuracy:.2f}%")
#got Validation Accuracy: 95.24%
