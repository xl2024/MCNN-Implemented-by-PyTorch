import cv2
import numpy as np
import torch
from torchvision import transforms
import os
from model import *


model_name='./models/3000.pth'
test_pic='./pics0/0save_name.jpg'
width=256
height=144


def get_model(model_name='./models/3000.pth'):
    mcnn=MCNN()
    mcnn.load_state_dict(torch.load(model_name))
    mcnn.eval()
    return mcnn


def get_output(model,test_pic='./pics0/0save_name.jpg',out_name='output.jpg'):
    img=cv2.imread(test_pic)
    img=cv2.resize(img, (width,height))
    pic=np.float32(img)
    if np.max(pic)!=np.min(pic) and np.std(pic)!=0:
        pic     = (pic-np.min(pic))/(np.max(pic)-np.min(pic))
        pic     = (pic-np.mean(pic))/np.std(pic)
    pic=transforms.ToTensor()(pic)
    pic=torch.reshape(pic,(-1,3,height,width))
    output=model(pic)
    output=output.detach().numpy()[0][0]
    cv2.imwrite(out_name,output*255)
    return np.sum(output)


def save_txt(name,content):
    txt=open(name,'w')
    content='\n'.join(str(i) for i in content)
    txt.write(content)
    txt.close()


if __name__=='__main__':
    test_dir='val_pics0/'
    gt_dir='labels0/'
    mcnn=get_model('./models/2148.pth')
    num_out=np.array([])
    num_label=np.array([])
    mae,mse,cnt=0.,0.,0
    for i in os.listdir(test_dir):
        num=get_output(mcnn,os.path.join(test_dir,i),'./out_dir/'+i)
        num_out=np.append(num_out,(i,num))
        txt=open(os.path.join(gt_dir,'{}{}'.format(i[0:-13],'save_name.txt')),'r')
        #txt=open(os.path.join(gt_dir,'labels_'+i[0:3]+'.txt'),'r')
        n=0
        for line in txt:
            n+=1
        txt.close()
        num_label=np.append(num_label,(i,n))
        mae+=abs(num-n)
        mse+=mae*mae
        cnt+=1
    save_txt('num_labels.txt',num_label)
    save_txt('num_out.txt',num_out)
    mae=mae/cnt
    mse=np.sqrt(mse)/cnt
    print(mae,mse)
