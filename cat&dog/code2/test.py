from getdata import DogsVSCatsDataset as DVCD
from network import Net
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import random
import os
import getdata

dataset_dir ='homework2/code2/kaggle' # 数据集路径
model_file = './model/model.pth'                # 模型保存路径
N = 10
# new version
def test():

    # setting model
    model = Net()                                      
    # model.cuda()                                       
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_file))       
    model.eval()                                        

    # get data
    files = random.sample(os.listdir(dataset_dir), N)   # 随机获取N个测试图像
    imgs = []           # img
    imgs_data = []      # img data
    for file in files:
        img = Image.open(dataset_dir + file)           
        img_data = getdata.dataTransform(img)          

        imgs.append(img)                               
        imgs_data.append(img_data)                     
    imgs_data = torch.stack(imgs_data)                 

    # calculation
    out = model(imgs_data)                             
    out = F.softmax(out, dim=1)                        
    out = out.data.cpu().numpy()                       

    # pring results         显示结果
    for idx in range(N):
        plt.figure()
        if out[idx, 0] > out[idx, 1]:
            plt.suptitle('cat:{:.1%},dog:{:.1%}'.format(out[idx, 0], out[idx, 1]))
        else:
            plt.suptitle('dog:{:.1%},cat:{:.1%}'.format(out[idx, 1], out[idx, 0]))
        plt.imshow(imgs[idx])
    plt.show()


if __name__ == '__main__':
    test()