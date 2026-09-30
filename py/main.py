import torch
from training_testing import training
import torch.nn as nn
from transfer_learning import EfficienModel
from utils import save_model
from py.data_load import Data_Loader
from Transforms import Get_Transform
from py.animal_model import AnimalModel


#------ Device ------#

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#------- Models ------#

my_own_model = AnimalModel().to(device)
efficient_model = EfficienModel()
efficient_model = efficient_model.to(device)

#------- Data Loading ------#

train_transform , test_transform ,model1_train_transform , model1_test_transform= Get_Transform()
train_loader ,test_loader ,val_loader ,class_names= Data_Loader(train_transform=model1_train_transform,test_transform=model1_test_transform)

#-------Optimizer and Loss ------#

optimizer1 = torch.optim.Adam(efficient_model.classifier.parameters(),lr=0.001)
optimizer = torch.optim.Adam(params=my_own_model.parameters(),lr=0.001)
loss_fn = nn.CrossEntropyLoss()

#-------- Training --------#

result,processed_model = training(loss_fn=loss_fn,optimizer=optimizer1,model=efficient_model,train_loader=train_loader,test_loader=test_loader,epochs=6,device=device)

#--------- Save --------#
save_model(model=processed_model, target_dir="../models", model_name="EfficientNet_model", class_names=class_names)