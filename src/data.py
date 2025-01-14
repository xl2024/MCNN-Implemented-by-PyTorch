from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import numpy as np
import os
import datetime
import cv2
import random
from scipy.ndimage import filters


class MyData(Dataset):

    def __init__(self,train_dir,gt_dir,width,height):
        self.train_dir=train_dir
        self.gt_dir=gt_dir
        self.width=width
        self.height=height
        self.dataset,self.labels=self.get_labels(self.train_dir,self.gt_dir,self.width,self.height)


    def __getitem__(self, index):
        data_ind=transforms.ToTensor()(self.dataset[index])   #torch_data=torch.FloatTensor(np_data)
        label_ind=transforms.ToTensor()(self.labels[index])
        return data_ind,label_ind


    def __len__(self):
        return len(self.dataset)


    def data_norm(self,dataset,labels):
        _dataset=[]
        _labels=[]
        for i in range(len(dataset)):
            pic=np.float32(dataset[i])
            if np.max(pic)!=np.min(pic) and np.std(pic)!=0:
                pic     = (pic-np.min(pic))/(np.max(pic)-np.min(pic))
                pic     = (pic-np.mean(pic))/np.std(pic)
                _dataset.append(pic)
                _labels.append(labels[i])
        h,w,c=pic.shape[0],pic.shape[1],pic.shape[2]
        _dataset=(np.array(_dataset)).reshape((-1,h,w,c))
        h,w=labels.shape[1:3]
        _labels=(np.array(_labels)).reshape((-1,h,w,1))
        return _dataset,_labels


    def den(self,img_shape=None,gt=None,width=320,height=180):        #produce the density map
        beta=0.3
        dist=np.array([])
        w,h,c=int(width/4+0.5),int(height/4+0.5),img_shape[2]
        den_map=np.zeros((h,w),dtype=np.float32)
        n=gt.shape[0]                 #the number of the labeled heads
        scal=width/img_shape[1]
        for i in range(n):
            for j in range(n):        #find the two nearest heads from the i-th head
                if i != j:
                    dist=np.append(dist,np.sqrt(np.sum(np.square(gt[i]-gt[j]))))
            dist.sort()
            dist=dist[0:2]            #set m=2
            dist=np.mean(dist)*scal/4
            x,y=gt[i]*scal/4
            x,y=int(x+0.5),int(y+0.5)
            if x==h:
                x=x-1
            if y==w:
                y=y-1
            k = int(beta*dist+0.5)
            if k==0:
                k=1
            dk=int(k/2)
            while x-dk<0 or x+dk>=h or y-dk<0 or y+dk>=w:
                dk-=1
            k=dk*2+1
            kn=np.zeros((dk*2+1,dk*2+1),dtype=np.float64)
            kn[dk][dk]=1.
            den_map[x-dk:x+dk+1,y-dk:y+dk+1]+=filters.gaussian_filter(kn,k)
        return den_map


    def get_labels(self,train,gt,width,height,count=0):
        train=self.train_dir
        gt=self.gt_dir
        width=self.width
        height=self.height
        dataset=np.array([],dtype=float)
        labels=np.array([],dtype=float)
        for lis in os.listdir(train):
            count=count+1
            if count % 10 == 0 :
                print(datetime.datetime.now(),'processing the '+str(count)+'th image...')
            img=cv2.imread(os.path.join(train, lis))
            label=np.array([],dtype=int)
            txt=open(os.path.join(gt,'{}{}'.format(lis[0:-13],'save_name.txt')),'r')
            #txt=open(os.path.join(gt,'labels_'+lis[0:3]+'.txt'),'r')
            for line in txt:
                label=np.append(label,eval(line))
            txt.close()
            label=label.reshape((-1,2))
            den_map=self.den(img.shape,label,width,height)
            cv2.imwrite('./den_maps/'+lis,den_map*255)
            labels=np.append(labels, den_map)
            img=cv2.resize(img, (width,height))
            dataset=np.append(dataset, img)
        channel=img.shape[2]
        dataset=dataset.reshape((-1,height,width,channel))
        labels=labels.reshape((-1,int(height/4+0.5),int(width/4+0.5),1))
        dataset,labels=self.data_norm(dataset,labels)
        #cv2.imwrite('last_data.jpg',(dataset[-1]+1)*127.5)
        #cv2.imwrite('last_label.jpg',labels[-1]*255)
        return dataset,labels


if __name__=='__main__':
    train='pics1'
    gt='labels1'
    width=256
    height=144
    mydata=MyData(train,gt,width,height)
    print(mydata)
