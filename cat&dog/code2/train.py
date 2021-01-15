

from getdata import DogsVSCatsDataset as DVCD
from torch.utils.data import DataLoader as DataLoader
from network import Net
import torch
from torch.autograd import Variable
import torch.nn as nn
from torchvision import models

dataset_dir = 'homework2/code2/kaggle' # 数据集路径

model_cp = 'model'               # 网络参数保存位置
workers = 10                        # PyTorch读取数据线程数量
batch_size = 16                     # batch_size大小
lr = 0.0001                         # 学习率
nepoch = 10


def train():
    datafile = DVCD('train', dataset_dir)                                                     
    dataloader = DataLoader(datafile, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)    

    print('Dataset loaded! length of train set is {0}'.format(len(datafile)))

    # model = Net()   
    model = models.resnet34(pretrained = True)   
    model.fc = nn.Linear(512,2)                
    # model = model.cuda()                
    # model = nn.DataParallel(model)
    model.train()                      

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)       

    criterion = torch.nn.CrossEntropyLoss()                       

    cnt = 0            
    for epoch in range(nepoch):
        
        for img, label in dataloader:                                           
            # img, label = Variable(img).cuda(), Variable(label).cuda()           
            out = model(img)                           
            loss = criterion(out, label.squeeze())     
            loss.backward()                            
            optimizer.step()                           
            optimizer.zero_grad()                      
            cnt += 1

            print('Epoch:{0},Frame:{1}, train_loss {2}'.format(epoch, cnt*batch_size, loss/batch_size))         

        torch.save(model.state_dict(), '{0}/model.pth'.format(model_cp))            # 训练所有数据后，保存网络的参数


if __name__ == '__main__':
    train()