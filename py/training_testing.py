import torch

def training_data(loss_fn:torch.nn.Module,
                  optimizer:torch.optim.Optimizer,
                  model:torch.nn.Module,
                  train_loader:torch.utils.data.DataLoader,
                  device:torch.device):


    model.train()


    if hasattr(model, "features"):
        if all(not p.requires_grad for p in model.features.parameters()):
            model.features.eval()


    train_loss , train_accuracy ,total_sample = 0,0,0

    for batch , (X,y) in enumerate(train_loader):

        X = X.to(device)
        y= y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred,y)

        batch_size = y.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_label = torch.argmax(y_pred,dim=1)

        train_loss += loss.item() * batch_size
        train_accuracy += (y_pred_label == y).sum().item()
        total_sample +=batch_size

    train_loss = train_loss/total_sample
    train_accuracy = train_accuracy/total_sample

    return train_loss ,train_accuracy



def testing_data(loss_fn:torch.nn.Module,
                 optimizer:torch.optim.Optimizer,
                 model:torch.nn.Module,
                 test_loader:torch.utils.data.DataLoader,
                 device:torch.device):

    model.eval()

    test_loss , test_accuracy , total_sample = 0,0,0

    with torch.inference_mode():
        for X,y in test_loader:

            X = X.to(device)
            y = y.to(device)

            test_pred = model(X)
            loss = loss_fn(test_pred, y)

            batch_size = y.size(0)

            test_loss += loss.item() * batch_size
            test_pred_label = test_pred.argmax(dim=1)
            test_accuracy += (test_pred_label == y).sum().item()
            total_sample += batch_size

        test_loss /= total_sample
        test_accuracy /= total_sample
    return test_loss,test_accuracy




def training(
    loss_fn,
    optimizer,
    model,
    train_loader,
    test_loader,
    epochs,
    device,
    patience=5,
    min_delta=0.001
):
    results = {
        "train loss": [],
        "test loss": [],
        "train accuracy": [],
        "test accuracy": []
    }

    best_val_loss = float("inf")
    best_weights = None
    best_epoch = 0

    patience_reference = float("inf")
    no_improvement = 0

    for epoch in range(epochs):
        train_loss, train_accuracy = training_data(
            loss_fn, optimizer, model, train_loader, device
        )

        val_loss, val_accuracy = testing_data(
            loss_fn, optimizer, model, test_loader, device
        )

        results["train loss"].append(train_loss)
        results["test loss"].append(val_loss)
        results["train accuracy"].append(train_accuracy)
        results["test accuracy"].append(val_accuracy)

        print(
            f"Epoch: {epoch + 1} | "
            f"train loss: {train_loss:.4f} | "
            f"train accuracy: {train_accuracy:.4f} | "
            f"val loss: {val_loss:.4f} | "
            f"val accuracy: {val_accuracy:.4f}"
        )


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_weights = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }


        if val_loss < patience_reference - min_delta:
            patience_reference = val_loss
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement >= patience:
            print(
                f"Early stopping: {patience} epoch boyunca "
                "validation loss yeterince iyileşmedi."
            )
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)
        model.eval()

        print(
            f"En iyi ağırlıklar yüklendi: epoch {best_epoch}, "
            f"val loss {best_val_loss:.4f}"
        )

    return results, model