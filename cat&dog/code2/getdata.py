import os
import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

IMAGE_SIZE = 200

# 定义一个转换关系，用于将图像数据转换成PyTorch的Tensor形式
dataTransform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),                          # 将图像按比例缩放至合适尺寸
    transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),        # 从图像中心裁剪合适大小的图像
    transforms.ToTensor()   
])


class DogsVSCatsDataset(data.Dataset):     
    def __init__(self, mode, dir):          
        self.mode = mode
        self.list_img = []                  
        self.list_label = []                
        self.data_size = 0                  
        self.transform = dataTransform      # 转换关系

        if self.mode == 'train':           
            dir = dir + '/train/'           
            for file in os.listdir(dir):    
                self.list_img.append(dir + file)       
                self.data_size += 1                    
                name = file.split(sep='.')              
               
                if name[0] == 'cat':
                    self.list_label.append(0)         
                else:
                    self.list_label.append(1)        
        elif self.mode == 'test':           
            dir = dir + '/test/'            
            for file in os.listdir(dir):
                self.list_img.append(dir + file)    # 添加图片路径至image list
                self.data_size += 1
                self.list_label.append(2)       # 添加2作为label，实际未用到，也无意义
        else:
            print('Undefined Dataset!')

    def __getitem__(self, item):            # 重载data.Dataset父类方法，获取数据集中数据内容
        if self.mode == 'train':                                        
            img = Image.open(self.list_img[item])                       
            label = self.list_label[item]                               
            return self.transform(img), torch.LongTensor([label])       
        elif self.mode == 'test':                                       
            img = Image.open(self.list_img[item])
            return self.transform(img)                                  # 只返回image
        else:
            print('None')

    def __len__(self):
        return self.data_size               # 返回数据集大小