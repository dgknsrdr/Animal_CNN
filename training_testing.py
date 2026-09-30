import torch

def training_data(loss_fn:torch.nn.Module,optimizer:torch.optim.Optimizer,model:torch.nn.Module,train_loader:torch.utils.data.DataLoader):
    model.train()
    train_loss , train_accuracy ,total_sample = 0,0,0
    for batch , (X,y) in enumerate(train_loader):
        y_pred = model(X)
        loss = loss_fn(y_pred,y)
        train_loss+=loss.item()
        batch_size = y.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_label = torch.argmax(y_pred,dim=1)
        train_accuracy += (y_pred_label==y).sum().item()/len(y_pred)
        total_sample +=batch_size
    train_loss = train_loss/total_sample
    train_accuracy = train_accuracy/total_sample

    return train_loss ,train_accuracy



def testing_data(loss_fn:torch.nn.Module,optimizer:torch.optim.Optimizer,model:torch.nn.Module,test_loader:torch.utils.data.DataLoader):
    model.eval()
    test_loss , test_accuracy , total_sample = 0,0,0
    with torch.inference_mode():
        for X,y in test_loader:
            test_pred = model(X)
            loss = loss_fn(test_pred,y)
            test_loss += loss.item()
            batch_size = y.size(0)
            test_pred_label = torch.argmax(test_pred,dim=1)
            test_accuracy += (test_pred_label==y).sum().item()/len(test_pred)
            total_sample += batch_size
        test_loss /= total_sample
        test_accuracy /= total_sample
    return test_loss,test_accuracy




def training (loss_fn:torch.nn.Module,
              optimizer:torch.optim.Optimizer,
              model:torch.nn.Module,
              train_loader:torch.utils.data.DataLoader,
              test_loader:torch.utils.data.DataLoader,
              epochs:int):
    results = {
        "train loss": [],
        "test loss": [],

        "train accuracy": [],
        "test accuracy": []
    }

    for epoch in range(epochs):
        train_loss , train_accuracy = training_data(loss_fn,optimizer,model,train_loader)
        test_loss ,test_accuracy = testing_data(loss_fn,optimizer,model,test_loader)


        print(
            f"Epoch: {epoch + 1} | "
            f"train loss: {train_loss:.4f} | "
            f"train accuracy: {train_accuracy:.4f} | "
            f"test loss: {test_loss:.4f} | "
            f"test accuracy: {test_accuracy:.4f}"
        )

        results["train loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["train accuracy"].append(train_accuracy.item() if isinstance(train_accuracy, torch.Tensor) else train_accuracy)
        results["test loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["test accuracy"].append(test_accuracy.item() if isinstance(test_accuracy, torch.Tensor) else test_accuracy)

    return results