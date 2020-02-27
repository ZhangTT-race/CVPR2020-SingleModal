from models.gray_rgb import gray_rgbnet,BasicBlock
from os import path

from datas.gray_rgb_casia import gray_rgb_casia

import torch
from torch.utils.data import DataLoader,RandomSampler
from torch.optim import Adam
from torch import nn

glb_name = "4@2"
cuda_id = 1
device = torch.device("cuda:%d"%(cuda_id) if torch.cuda.is_available() else "cpu")
print(device)

data_root = "/nfs/private/wdh/cefa-train/dataset/CASIA-CeFA"
# data_root = "/Users/wdh/Downloads/CASIA-CeFA"
train_label_path = path.join(data_root, "%s_train.txt"%(glb_name))

checkpoints_root = "./checkpoints/gray%s"%(glb_name)

channels = 4
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
        step_count += 1

        optimizer.zero_grad()

        res = net.forward(inputs[0].to(device),inputs[1].to(device))
        loss = criterion(res,labels.to(device))
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if step_count % 10 == 0:
            print('step:%d, epoch:%d, turn:%4d loss:%.4f' %
                  (step_count,epoch+1, id+1 , running_loss ))

        if step_count % 1000 == 0:
            torch.save(net.state_dict(),path.join(checkpoints_root,"%05d_loss_%.4f.pth"%(step_count,running_loss)))
        running_loss = 0.0

    # adjust learning rate
    if epoch >= epochs / 2 and (epoch+1) % 20 == 0:
        for p in optimizer.param_groups:
            p['lr'] *= 0.5
