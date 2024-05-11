import os  
from tqdm import tqdm  
import torch
from torch.utils.data import DataLoader
from utils.result_save import train_save

def train_step(model, optimizer, train_loader, loss_fun, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    scaler = torch.cuda.amp.GradScaler()  
    model.train()
    avg_loss = []
    for x,y in tqdm(train_loader):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(): 
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = loss_fun(y_pred, y)
            # next three lines are unchanged for all the tasks
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        avg_loss.append(loss.item())
    return sum(avg_loss)/len(avg_loss)  # ave_loss for current epoch


def predict(model, test_loader,loss_fun, arg_flag:bool=True,train_process:bool=True, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    avg_pre_loss = []
    with torch.no_grad():
        for x,y in tqdm(test_loader):
            x = x.to(device)
            y = y.to(device)  # truth
            pred = model(x)   
            pre_loss = loss_fun(pred, y)
            if arg_flag == True:
                y_pred = torch.argmax(pred, dim=1)
            else:
                y_pred = pred
            total_preds = torch.cat((total_preds, y_pred.cpu()), 0)  # prediction 
            total_labels = torch.cat((total_labels, y.cpu()), 0) #true label
            avg_pre_loss.append(pre_loss.item())
    if train_process:
        return total_labels.numpy().flatten(),total_preds.numpy().flatten(),sum(avg_pre_loss)/len(avg_pre_loss)
    else:
        return total_labels.numpy().flatten(),total_preds.numpy().flatten()
    

def train(model, dataset, val_dataset, result_savepath, loss_fun, lr=0.0001, num_epoch:int=100, batch_size:int=10,
          output_name:str='', optim_ada:bool=True, arg_flag =True,
          device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    
    if optim_ada:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    
    batch_size = batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = model.to(device)
    model_file_name = output_name
    max_pre_loss = 2000   
    pre_loss =[2000]     

    for epoch in range(num_epoch):
        epoch_train_loss = train_step(model, optimizer, dataloader,loss_fun) 
        G,P,epoch_pre_loss = predict(model, val_dataloader, loss_fun,arg_flag, train_process=True)
        print(f'epoch={epoch},train_loss={epoch_train_loss},test_loss={epoch_pre_loss}')
        if epoch_pre_loss < max_pre_loss:
            max_pre_loss = epoch_pre_loss
            torch.save(model.state_dict(), model_file_name)
            train_save(epoch,epoch_train_loss,epoch_pre_loss,G,P,result_savepath)
        else:
            scheduler.step()

        # early stop
        if epoch_pre_loss <= pre_loss[0]:  
            pre_loss.clear()  
            pre_loss.append(epoch_pre_loss)
        else:
            pre_loss.append(epoch_pre_loss)
        if len(pre_loss) > 30:
            print('early stop current fold')
            break
            
def load_model(model, output_name:str=''):
    # load the weight
    model_file_name = output_name
    if os.path.isfile(model_file_name):
        checkpoint = torch.load(model_file_name)
        model.load_state_dict(checkpoint)
    return model


    