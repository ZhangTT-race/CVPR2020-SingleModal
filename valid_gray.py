import os
from models.gray_rgb import gray_rgbnet,BasicBlock

import torch

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu") #set cuda number
print(device)

from os import path
import numpy as np
import cv2

global_models = ["4@1","4@2","4@3"]

model_names = ["36000_loss_0.0006.pth","36000_loss_0.0041.pth","36000_loss_0.0060.pth"] #set name of your model here

global_actions = ["dev","test"] #"test" or "dev"

data_root = "/nfs/private/wdh/cefa-train/dataset/CASIA-CeFA" #dataset path
# data_root = "/Users/wdh/Downloads/CASIA-CeFA/"

for id_model,global_model in enumerate(global_models):
    for global_action in global_actions:
        checkpoints_root = "./checkpoints/gray%s"%global_model
        res_name = "%s_%s_res.txt"%(global_model,global_action)
        model_name = model_names[id_model]

        channels = 3
        rows = 256
        cols = 256

        times = 1

        net = gray_rgbnet(BasicBlock, [2, 2, 2, 2], num_classes=4)

        net.load_state_dict(torch.load(path.join(checkpoints_root,model_name),map_location=device))
        net.eval()
        net.to(device)

        lines = open(path.join(data_root,res_name)).readlines()
        paths = tuple(line.split(" ")[0] for line in lines)

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
                id1 = 1
         
                if global_action == "dev":
                    id2 = id1
                else:
                    id2 = frame_count

                ids = [id1, id2]
                imgs = []
                img_y = None
                for k,id in enumerate(ids):
                    try:
                        img = cv2.resize(cv2.imread(path.join(rgb_root,"%04d.jpg"%id)), (cols, rows))
                    except:
                        print(path.join(rgb_root,"%04d.jpg"%id))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    if global_model == "4@1" and global_action == "test":
                        gamma = 0.5
                        invGamma = 1.0 / gamma 
                        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                        img = cv2.LUT(np.array(img, dtype=np.uint8), table) 
                    
                    img = cv2.equalizeHist(img)
                    img = (img / 255.0).astype("float32")
                    img = np.expand_dims(img, axis=0)

                    imgs.append(img)
                    if k == 0:
                        img_y = img.copy()

                cur_tensor = np.concatenate(imgs, 0)
                cur_tensor = np.expand_dims(cur_tensor,axis=0)
                cur_tensor_y = np.expand_dims(img_y,axis=0)
                cur_tensor = torch.tensor(cur_tensor,device=device)
                cur_tensor_y = torch.tensor(cur_tensor_y,device=device)
                # print(cur_tensor.shape)

                res = net(cur_tensor,cur_tensor_y)

                if global_action == "dev":
                    p_real = res.cpu().detach().numpy()[0, 1]
                else:
                    p_real = res.cpu().detach().numpy()[0, 3]

                cur_res_dir.append(p_real)

            p_real_dir = np.mean(cur_res_dir)
            res_line = rpath + " " + str(p_real_dir)
            print(res_line)

            res_lines.append(res_line + "\n")

        f = open(res_name.replace("_res","_gray_res"), "w+")
        f.writelines(res_lines)
        f.close()
