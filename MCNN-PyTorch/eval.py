import numpy as np
from model import *


def eval_model(gt,pred):
    #gt=gt.detach().numpy()
    gt = gt.detach().cpu().numpy()
    pred=pred.detach().cpu().numpy()
    mae=0.
    mse=0.
    for i in range(len(gt)):
        sum_gt=np.sum(gt[i])
        sum_pred=np.sum(pred[i])
        diff=sum_gt-sum_pred
        mae+=abs(diff)
        mse+=diff*diff
    mae=mae/len(gt)
    mse=np.sqrt(mse)/len(gt)
    return mae,mse
