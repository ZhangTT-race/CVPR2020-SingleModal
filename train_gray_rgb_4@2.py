from models.gray_rgb import gray_rgbnet,BasicBlock
from os import path

from datas.gray_rgb_casia import gray_rgb_casia

import torch
from torch.utils.data import Dataset,DataLoader,RandomSampler
from torch.optim import Adam
from torch import nn

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

data_root = "/Users/wdh/Downloads/CASIA-CeFA"
train_label_path = path.join(data_root, "4@2_train.txt")

checkpoints_root = "./checkpoints/gray_4@2"

channels = 3
rows = 256
cols = 256
batch_size = 32
epochs = 3000
lr =  0.0001

data_set = gray_rgb_casia(train_label_path, data_root, rows, cols, channels)
sampler = RandomSampler(data_set, replacement=True, num_samples=len(data_set))
data_loader = DataLoader(dataset=data_set, batch_size=batch_size, sampler=sampler, drop_last=True)

net = gray_rgbnet(BasicBlock, [2, 2, 2, 2], num_classes=4)
net.to(device)

optimizer = Adam(net.parameters(),lr)
criterion = nn.BCELoss()

running_loss = 0.0
step_count = 0
for epoch in range(epochs):
    for id, (inputs, labels, paths) in enumerate(data_loader):
        # print(inputs.shape,rank_pooling.shape,labels,paths)
        step_count += 1

        optimizer.zero_grad()

        res = net.forward(inputs.to(device))
        loss = criterion(res,labels.to(device))
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if step_count % 10 == 0:
            print('step:%d, epoch:%d, turn:%4d loss:%.4f' %
                  (step_count,epoch+1, id+1 , running_loss ))
        if step_count % 1000 == 0:
            torch.save(net.state_dict(),path.join(checkpoints_root,"%d_loss_%.4f.pth"%(step_count,running_loss)))
        running_loss = 0.0

    # adjust learning rate
    if epoch >= epochs / 2 and (epoch+1) % 20 == 0:
        for p in optimizer.param_groups:
            p['lr'] *= 0.5
