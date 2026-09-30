import torch
from torchvision import datasets,transforms
from torch.utils.data import random_split,DataLoader,Subset
from training_testing import training
import torch.nn as nn
import animal_model


train_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(p=0.4),
    transforms.TrivialAugmentWide(),
    transforms.ColorJitter(brightness=0.1,contrast=0.2,saturation=0.2,hue=0.03),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4508, 0.4392, 0.3964],std=[0.2799, 0.2714, 0.2758])
])



test_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4508, 0.4392, 0.3964],std=[0.2799, 0.2714, 0.2758])
])


train_full = datasets.ImageFolder(root=("datasets/animals"),transform=train_transform)
test_full = datasets.ImageFolder(root=("datasets/animals"),transform=test_transform)

n_total = len(train_full)
n_test = int(n_total*0.2)
n_val = int((n_total - n_test) * 0.15)
n_train = n_total - (n_test+n_val)
print(len(train_full.classes))
g = torch.Generator().manual_seed(42)
train_idx , test_idx , val_idx = random_split(range(len(train_full)),[n_train,n_test,n_val],generator=g)

train_data = Subset(train_full,list(train_idx))
test_data = Subset(test_full,list(test_idx))
val_data = Subset(test_full,list(val_idx))

train_loader = DataLoader(dataset=train_data,batch_size=32,shuffle=True)
test_laoder = DataLoader(dataset=test_data,batch_size=32,shuffle=False)
val_loader = DataLoader(dataset=val_data,batch_size=32,shuffle=False)

model = animal_model.AnimalModel()
optimizer = torch.optim.Adam(params=model.parameters(),lr=0.001)
loss_fn = nn.CrossEntropyLoss()

result = training(loss_fn=loss_fn,optimizer=optimizer,model=model,train_loader=train_loader,test_loader=val_loader,epochs=20)

print(result)