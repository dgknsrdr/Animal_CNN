import torch
from torchvision import datasets
from torch.utils.data import random_split,DataLoader,Subset


def Data_Loader(train_transform , test_transform):
    train_full = datasets.ImageFolder(root=("../datasets/animals"),transform=train_transform)
    test_full = datasets.ImageFolder(root=("../datasets/animals"),transform=test_transform)
    class_names = train_full.classes


    n_total = len(train_full)
    n_test = int(n_total*0.2)
    n_val = int((n_total - n_test) * 0.15)
    n_train = n_total - (n_test+n_val)



    g = torch.Generator().manual_seed(42)
    train_idx , test_idx , val_idx = random_split(range(len(train_full)),[n_train,n_test,n_val],generator=g)

    train_data = Subset(train_full,list(train_idx))
    test_data = Subset(test_full,list(test_idx))
    val_data = Subset(test_full,list(val_idx))

    train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=False)
    val_loader = DataLoader(dataset=val_data, batch_size=32, shuffle=False)

    return train_loader,test_loader,val_loader ,class_names