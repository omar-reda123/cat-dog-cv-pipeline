#imports
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import pathlib
from pathlib import Path
import torchvision.transforms as transforms
import os
#initializations
device="cuda" if torch.cuda.is_available() else "cpu"
print(f"device is : {device}")

cat_train=Path("/home/omar/Projects/pcv/data/splitted-data/train/Cat")
cat_val=Path("/home/omar/Projects/pcv/data/splitted-data/val/Cat")
dog_train=Path("/home/omar/Projects/pcv/data/splitted-data/train/Dog")
dog_val=Path("/home/omar/Projects/pcv/data/splitted-data/val/Dog")

labels={
    cat_train.name:0,
    dog_train.name:1
}
print(f"labels: {list(labels.items())}")

#dataset
class MyDataset(Dataset):
    def __init__(self,*args:pathlib.PosixPath,transforms=None):
        super().__init__()
        self.image_paths=[]
        self.labels=[]
        for path in args:
            for image_path in path.iterdir():
                self.image_paths.append(image_path)
                self.labels.append(labels[image_path.parent.name])
       
        self.transforms = transforms
            
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path=self.image_paths[index]
        label=torch.tensor(self.labels[index],dtype=torch.long)
        
        image=cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)        
        if self.transforms:
            image=self.transforms(image)
        return image,label

#instances
train_transforms = transforms.Compose([
    transforms.ToPILImage(), 
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = MyDataset(cat_train, dog_train, transforms=train_transforms)
val_dataset = MyDataset(cat_val, dog_val, transforms=val_transforms)

#dataloader
workers = min(4, os.cpu_count() or 1) 
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=workers,pin_memory=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False, num_workers=workers,pin_memory=True)
    
    
            
            
        




