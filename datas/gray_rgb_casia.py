from torch.utils.data import Dataset,DataLoader,RandomSampler
from os import path
import cv2
import numpy as np

class gray_rgb_casia(Dataset):
    def __init__(self,label_path, data_root, rows, cols, channels):
        super(gray_rgb_casia,self).__init__()

        self.rows = rows
        self.cols = cols
        self.channels = channels

        self.label_path = label_path
        self.data_root = data_root

        lines = open(label_path).readlines()

        print("reading files")

        #解析所有路径和
        self.video_paths = set()
        self.video_infos = dict()

        for line in lines[0:]:
            frame_path, label = line.split(" ")[0:2]
            video_path = frame_path[0:frame_path.rfind("/") + 1] #到profile
            frame_name = frame_path[frame_path.rfind("/") + 1:]
            frame_id = int(frame_name.split(".")[0]) # frome 1 to n

            self.video_paths.add(video_path)
            frame_buf = np.asarray(bytearray(open(path.join(data_root, frame_path), "rb").read()), "uint8")

            if video_path in self.video_infos:
               self.video_infos[video_path][frame_id] = (frame_buf, int(label))
            else:
                self.video_infos[video_path] = {frame_id:(frame_buf, int(label))}

        self.video_paths = tuple(self.video_paths)

        print("init done")

    def __getitem__(self, index): # index 范围是video的数量
        video_path = self.video_paths[index]
        frame_count = len(self.video_infos[video_path])
        video_info = self.video_infos[video_path]

         #label 不由标签决定，而是在此处构造
        label_ori = video_info[1][1] #原始label
        label = np.random.randint(0,2)
        imgs = []
        if label == 0: #0是负样本
            id = np.random.randint(1, frame_count + 1)
            ids = [id,id,id] #取同一帧

            flag1 = np.random.randint(0, 2) #决定是否翻转
            flag2 = 1 #np.random.randint(0,2) #决定是否直方图均衡化

            img_ori = cv2.resize(cv2.imdecode(video_info[id][0], cv2.IMREAD_COLOR),
                             (self.cols, self.rows))
            img_ori = cv2.cvtColor(img_ori,cv2.COLOR_BGR2GRAY) #灰度图
            #随机翻转翻转
            if flag1 != 0:
                img_ori = cv2.flip(img_ori,1)
            if flag2 !=0:
                img_ori = cv2.equalizeHist(img_ori)

            for k,id in enumerate(ids):
                if k == 0:
                    img_channel0 = progress(img_ori.copy(), isnoise=False)
                    img_channel0 = (img_channel0 / 255.0).astype("float32")
                    img_channel0 = np.expand_dims(img_channel0, axis=0)
                    imgs.append(img_channel0)

                img = progress(img_ori) #模拟帧间差异
                img = (img / 255.0).astype("float32")
                img = np.expand_dims(img, axis=0)

                imgs.append(img)
        else:
            id1 = np.random.randint(1, frame_count // 3 + 1)
            id2 = np.random.randint(id1+frame_count //3, 2*frame_count // 3 + 1)
            id3 = np.random.randint(id2 + frame_count //3, frame_count + 1)

            ids = [id1,id2,id3]
            # 决定是否翻转
            flag1 = np.random.randint(0, 2)  # 决定是否翻转
            flag2 = 1 #np.random.randint(0, 2)  # 决定是否直方图均衡化
            for k,id in enumerate(ids):
                img = cv2.resize(cv2.imdecode(video_info[id][0], cv2.IMREAD_COLOR), (self.cols, self.rows))
                img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                if flag1 != 0:
                    img = cv2.flip(img, 1)
                if flag2 != 0:
                    img = cv2.equalizeHist(img)
                if k == 0:
                    img_channel0 = progress(img.copy(), isnoise=False)
                    img_channel0 = (img_channel0 / 255.0).astype("float32")
                    img_channel0 = np.expand_dims(img_channel0, axis=0)
                    imgs.append(img_channel0)

                img = progress(img)  # 模拟帧间差异

                img = (img / 255.0).astype("float32")
                img = np.expand_dims(img,axis=0)

                imgs.append(img)

        # cur_tensor = np.concatenate(imgs,2)
        cur_tensor = np.concatenate(imgs, 0)  # 拼成三个gray通道
        cur_label = label_ori + 2*label

        return cur_tensor,np.eye(4,dtype="float32")[cur_label],path.join(video_path,"%04d.jpg,%04d.jpg,%04d.jpg"%(ids[0],ids[1],ids[2]))

    def __len__(self):
        return len(self.video_paths)

def progress(img,gamma=1.0,mean=0,var=0.001,israndom=True,isnoise=True):
    #模拟帧间差异
    rows,cols = img.shape[0:2]
    #输入是uint8的bgr图
    flag = 1

    #直方图均衡化 也要提到外面帧间差异 没有这么大
    # if israndom:
    #     flag = np.random.randint(0, 2)
    # if flag != 0:
    #     img = cv2.equalizeHist(img)
    #gamma矫正
    if israndom:
        flag = np.random.randint(0,2)
    if flag != 0:
        #拉亮或者压暗 gamma校正 gamma大于0
        gamma += 0.1 - 0.2*np.random.rand() #随机叠加强度
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        img = cv2.LUT(np.array(img, dtype=np.uint8), table)

    #高斯噪声
    if israndom:
        flag = np.random.randint(0,2)
    if flag != 0 and isnoise:
        img = np.array(img / 255, dtype="float32")
        noise = np.random.normal(mean, var ** 0.5, img.shape)
        out = img + noise
        if out.min() < 0:
            low_clip = 0.
        else:
            low_clip = 0.
        out = np.clip(out, low_clip, 1.0)
        img = np.uint8(out * 255)

    # 反转在外面做 以防止两张方向不对
    # if israndom:
    #     flag = np.random.randint(0,2)
    # if flag != 0:
    #     img = cv2.flip(img,1)

    # 小幅度偏移 旋转 放射变换
    if israndom:
        flag = np.random.randint(0,2)
    if flag != 0:
        cols_wise = int(cols*0.025)
        rows_wise = int(rows*0.025)
        x_move = np.random.randint(-cols_wise,cols_wise)
        y_move = np.random.randint(-rows_wise,rows_wise)
        x_scale = 1.025 - 0.05*np.random.rand()
        y_scale = 1.025 - 0.05*np.random.rand()
        x_rotate = 0.025 - 0.05*np.random.rand()
        y_rotate = 0.025 - 0.05*np.random.rand()

        H = np.float32([[x_scale, x_rotate,x_move ], [y_rotate, y_scale, y_move]])
        rows, cols = img.shape[:2]
        img = cv2.warpAffine(img, H, (cols, rows))

    return img

def rect_img(im): #裁剪负样本人脸
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(imgray,1,255,cv2.THRESH_BINARY)

    kernel = np.ones((5,5),np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # cv2.imwrite("binary.jpg",thresh)
    # print(thresh)
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # cv2.RETR_EXTERNAL 定义只检测外围轮廓

    # print(contours)
    # cnts = contours[0]
    cnt = contours[np.argmax([cv2.contourArea(cnt) for cnt in contours[0]])]
    # 外接矩形框，没有方向角
    x, y, w, h = cv2.boundingRect(cnt[0])


    return (x,y,w,h)

if __name__ == "__main__":
    # israndom = True
    # # img = cv2.imread("/Users/wdh/Downloads/CASIA-CeFA/train/1_031_1_1_1/profile/0001.jpg")
    # img = cv2.imread("/Users/wdh/Downloads/CASIA-CeFA/dev/003062/profile/0003.jpg")
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # new_img = progress(img,1.,israndom=israndom)
    #
    # show_img1 = np.concatenate([img,new_img],axis=1)
    # cv2.imwrite("showimg1.jpg",show_img1)
    #
    # img = cv2.imread("/Users/wdh/Downloads/CASIA-CeFA/dev/003063/profile/0003.jpg")
    # (x,y,w,h) = rect_img(img)
    # img = img[y:y+h,x:x+w]
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # new_img = progress(img,1.,israndom=israndom)
    #
    # show_img2 = np.concatenate([img, new_img], axis=1)
    # cv2.imwrite("showimg2.jpg",show_img2)
    #
    # img = cv2.imread("/Users/wdh/Downloads/CASIA-CeFA/phase2/test/000010/profile/0023.jpg")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # new_img = progress(img,1.8,israndom=israndom)
    #
    # show_img3 = np.concatenate([img, new_img], axis=1)
    # cv2.imwrite("showimg3.jpg", show_img3)

    data_root = "/Users/wdh/Downloads/CASIA-CeFA/"
    train_label_path = path.join(data_root, "4@1_train.txt")

    channels = 1
    rows = 256
    cols = 256

    data_set = gray_rgb_casia(train_label_path, data_root, rows, cols, channels)
    sampler = RandomSampler(data_set, replacement=True, num_samples=30)
    data_loader = DataLoader(dataset=data_set, batch_size=1, sampler=sampler, drop_last=True)

    # iter_data = iter(data_loader)
    print(len(data_set))
    for i in range(1):
        for id, (inputs,labels, paths) in enumerate(data_loader):
            img = inputs.numpy()[0]
            img = np.concatenate([img[0,:,:],img[1,:,:],img[2,:,:],img[3,:,:]],axis=1)
            label = labels.numpy()[0]
            cv2.imwrite("/Users/wdh/fsdownload/imshow/%d_%d%d%d%d.jpg"%(id,label[0],label[1],label[2],label[3]),img*255)
            print(id,inputs.shape,labels,paths)

