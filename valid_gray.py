import os
from models.gray_rgb import gray_rgbnet,BasicBlock

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

from os import path
import numpy as np
import cv2

global_model = "4@3"
global_action = "dev"
data_root = "/nfs/private/wdh/cefa-train/dataset/CASIA-CeFA"
#data_root = "/Users/wdh/Downloads/CASIA-CeFA/"
# train_label_path = path.join(data_root,"4@1_train.txt")

# dev_res_name = "4@1_test_res.txt"
# res_name = "4@1_dev_res.txt"
res_name = "%s_%s_res.txt"%(global_model,global_action)


# dev_label_path = path.join(data_root, res_name)
# dev_label_path = "/Users/wdh/fsdownload/myval.txt"

# checkpoints_root = "/Users/wdh/PycharmProjects/cefa_torch/checkpoints"
checkpoints_root = "/nfs/private/wdh/cefa-train/cefa-code-torch/checkpoints/gray%s"%global_model
# checkpoints_root = "/Users/wdh/fsdownload/"

channels = 3
rows = 256
cols = 256
batch_size = 4
now = 1 # 1,2,3
times = 1
thres = 0.5

net = gray_rgbnet(BasicBlock, [2, 2, 2, 2], num_classes=4)
#net.load_state_dict(torch.load(path.join(checkpoints_root,'23000_loss_0.0020.pth'),map_location=torch.device('cpu')))
#net.load_state_dict(torch.load(path.join(checkpoints_root,'30000_loss_0.0010.pth')))
#net.load_state_dict(torch.load(path.join(checkpoints_root,'35000_loss_0.0020.pth')))
net.load_state_dict(torch.load(path.join(checkpoints_root,'25000_loss_0.0029.pth')))
net.eval()
net.to(device)


# lines = open(dev_label_path).readlines()
lines = open(path.join(data_root,res_name)).readlines()

paths = tuple(line.split(" ")[0] for line in lines)
# label = tuple(line.split(" ")[1] for line in lines)

x = np.zeros((1, rows, cols, channels), dtype=np.float)
# y = np.zeros((1, 2), dtype=np.float)

res_lines = []

for index,rpath in enumerate(paths[0:]):
    rpath = rpath.strip("\n")
    video_path = path.join(data_root, rpath)
    rgb_root = path.join(video_path, "profile")
    # print(rgb_root)

    filenames = os.listdir(rgb_root)
    frame_count = len(filenames)

    cur_res_dir = []
    for t in range(times):
        # frame_choses = np.random.choice(filenames,batch_size)
        id1 = np.random.randint(1, frame_count // 4 + 1)
        if global_action == "dev":
            id2 = id1  # 间隔至少1/4 frame_count [0,3] [2-5,7]
        else: 
            id2 = np.random.randint(id1 + frame_count // 2, frame_count + 1)  # 间隔至少1/4 frame_count [0,3] [2-5,7]
        # id2 = np.random.randint(id1 + frame_count // 4,id1 + frame_count // 4)  # 间隔至少1/4 frame_count [0,3] [2-5,7]
        ids = [id1, id2]
        # 决定是否翻转
        flag1 = np.random.randint(0, 2)  # 决定是否翻转
        flag2 = np.random.randint(0, 2)  # 决定是否直方图均衡化
        imgs = []
        for k,id in enumerate(ids):
            img = cv2.resize(cv2.imread(path.join(rgb_root,"%04d.jpg"%id)), (cols, rows))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # if flag1 != 0:
            #     img = cv2.flip(img, 1)
            # if flag2 != 0:
            img = cv2.equalizeHist(img)
            # img = progress(img)  # 模拟帧间差异

            img = (img / 255.0).astype("float32")
            img = np.expand_dims(img, axis=0)

            imgs.append(img)
            if k == 0:
                imgs.append(img)

        cur_tensor = np.concatenate(imgs, 0)
        cur_tensor = np.expand_dims(cur_tensor,axis=0)
        cur_tensor = torch.tensor(cur_tensor,device=device)
        # print(cur_tensor.shape)
        # print(cur_rankpooling.shape)

        res = net(cur_tensor)
        #print(res)
        if global_action == "dev":
            p_real = res.cpu().detach().numpy()[0, 1]
        else:
            p_real = res.cpu().detach().numpy()[0, 3]# + res.cpu().detach().numpy()[0, 3]
        # print(p_real)

        cur_res_dir.append(p_real)

    p_real_dir = np.mean(cur_res_dir)

    res_line = rpath + " " + str(p_real_dir)

    print(res_line)
    # print(res_line)
    res_lines.append(res_line + "\n")

# print("FP:%d,TP:%d,TN:%d,FN:%d apcer:%.4f,bpcer:%.4f,acer:%.4f" % (FP,TP,TN,FN,apcer,bpcer,acer))
f = open(res_name.replace("_res","_gray_res"), "w+")
f.writelines(res_lines)
f.close()
