from torch.utils.data import Dataset,DataLoader
from torch.optim import Adam
from torch import nn
import torch
from torchvision import transforms
import datetime
from data import *
from model import *
from eval import *


train='pics0/'
gt='labels0/'
val='val_pics0/'
val_gt='labels0/'
width=256
height=144
start=0
epochs=5000
lr=0.00001
model_path='./models2/'
use_cuda = torch.cuda.is_available()


mydata=MyData(train,gt,width,height)
dataloader=DataLoader(mydata,batch_size=32,shuffle=True)

mcnn=MCNN()
if use_cuda:
    mcnn = mcnn.cuda()

if start > 0:
    mcnn.load_state_dict(torch.load('./models2/'+str(start)+'.pth'))
optimizer=Adam(mcnn.parameters(),lr=lr)
criterion=nn.MSELoss()
if use_cuda:
    criterion = criterion.cuda()

for epoch in range(epochs):
    log_loss=0.
    log_mae=0.
    log_mse=0.
    for data,labels in dataloader:
        data=data.to(torch.float32)
        labels=labels.to(torch.float32)
        if use_cuda:
            data = data.cuda()
            labels = labels.cuda()
        output=mcnn(data)
        loss=criterion(output,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        log_loss += loss.item()*len(data)
        mae,mse=eval_model(labels,output)
        log_mae += mae*len(data)
        log_mse += mse*len(data)
    if (epoch + 1) % 10 == 0:
        print(datetime.datetime.now(),start+epoch+1,log_loss/len(dataloader.dataset),
            log_mae/len(dataloader.dataset),
            log_mse/len(dataloader.dataset))
        torch.save(mcnn.state_dict(),model_path+str(start+epoch+1)+'.pth')
    '''
    model=MCNN()
    model.load_state_dict(torch.load('./models/'+str(epoch)+'.pth'))
    model.eval()
    '''
