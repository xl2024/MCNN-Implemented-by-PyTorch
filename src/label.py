import cv2
import os

path='./pics'                 #the path of file fold containing pictures
r=5                      #error occurs with the size of eraser beyond r
bound=240                #the number of RGB cannot be 255
save_path='./labels'
save_name='label.txt'

'''
#colorful2grey
for lis in os.listdir(path):  
    img=cv2.imread(os.path.join(path,lis))
    print(img.max(),end=',')
    for i in range(img.shape[0]):
        for j in range(img.shape[1]): 
            for k in range(img.shape[2]):
                if img[i][j][k]>=bound-20:
                    img[i][j][k]=bound-20
    cv2.imwrite(os.path.join(save_path,lis),img)            #modify when used
    print(img.max(),end='.')
                  
'''

'''
#save labels as .txt
path='./labels'
save_path='./labels2'
for lis in os.listdir(path): 
    print(lis)
    label=[]
    img=cv2.imread(os.path.join(path,lis))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]): 
            exi=False
            if img[i][j][2]>bound and img[i][j][1]>bound and img[i][j][0]>bound:
                if i>0 and j>0 and not (img[i-1][j][2]>bound and img[i-1][j][1]>bound and img[i-1][j][0]>bound) and not (img[i][j-1][2]>bound and img[i][j-1][1]>bound and img[i][j-1][0]>bound):
                    for m in range(r*2):
                        for n in range(r*2):
                            if (i-r+m,j-r+n) in label:
                                exi=True
                                break
                    if exi==False:
                        it,jt=i,j
                    for x in range(2*r):
                        for y in range(2*r):
                            xi,yj=i+x,j+y
                            if xi+1<img.shape[0] and yj+1<img.shape[1] and not (img[xi+1][yj][2]>bound and img[xi+1][yj][1]>bound and img[xi+1][yj][0]>bound) and not (img[xi][yj+1][2]>bound and img[xi][yj+1][1]>bound and img[xi][yj+1][0]>bound) and (img[xi][yj][2]>bound and img[xi][yj][1]>bound and img[xi][yj][0]>bound):
                                for m in range(r*2):
                                    for n in range(r*2):
                                        if (xi-r+m,yj-r+n) in label:
                                            exi=True
                                            break
                                if exi==False:
                                    label.append((round((it+xi)/2),round((jt+yj)/2)))
                                    exi=True
                    if exi==False:
                        label.append((it+r,jt+r))
    txt=open(os.path.join(save_path,'{}{}'.format(lis[0:-4],'.txt')),'w')
    #txt=open(os.path.join(save_path,'{}{}{}'.format('labels_',lis[0:3],'.txt')),'w')
    label='\n'.join(str(i) for i in label)  
    txt.write(label)
    txt.close()  

    txt=open(os.path.join(save_path,'{}{}{}'.format('labels_',lis[0:3],'.txt')),'r')
    for line in txt:
        print(lis,eval(line))
    txt.close()
'''
