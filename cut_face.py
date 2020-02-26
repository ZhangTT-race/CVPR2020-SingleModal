import cv2
import numpy as np
import os
from os import path
import shutil
import imutils

global_action = "dev" #"train" or "dev"
data_root = "/Users/wdh/Downloads/CASIA-CeFA/%s"%global_action

def rect_img(im):
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(imgray,1,255,cv2.THRESH_BINARY)

    kernel = np.ones((5,5),np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # cv2.RETR_EXTERNAL 定义只检测外围轮廓
    cnts = contours[1] if imutils.is_cv3() else contours[0]

    cnt = cnts[np.argmax([cv2.contourArea(cnt) for cnt in cnts])]

    # 外接矩形框，没有方向角
    x, y, w, h = cv2.boundingRect(cnt)
    return (x,y,w,h)

video_list = os.listdir(data_root)
total = len(video_list)
for id,video_name in enumerate(video_list[0:]):
    if global_action == "train" and video_name[6:] != "3_1_4":
        continue
    elif int(video_name) % 2 == 0:
       continue

    print(video_name,"%d / %d" %(id+1,total))
    video_path = path.join(data_root,video_name)

    work_profile = path.join(video_path, "profile")
    work_ir = path.join(video_path, "ir")
    work_depth = path.join(video_path, "depth")

    bak_profile = path.join(video_path, "profile_bak")
    bak_ir = path.join(video_path, "ir_bak")
    bak_depth = path.join(video_path, "depth_bak")

    # move src to dst
    if not path.exists(bak_profile):
        shutil.move(work_profile, bak_profile)
    if not path.exists(bak_ir):
        shutil.move(work_ir, bak_ir)
    if not path.exists(bak_depth):
        shutil.move(work_depth, bak_depth)

    # make new src
    if not path.exists(work_profile):
        os.makedirs(work_profile)
    if not path.exists(work_ir):
        os.makedirs(work_ir)
    if not path.exists(work_depth):
        os.makedirs(work_depth)

    image_list = os.listdir(bak_profile)
    #print(image_list,len(image_list))
    for k,image_name in enumerate(image_list[0:]):
        profile = cv2.imread(path.join(bak_profile,image_name))
        ir = cv2.imread(path.join(bak_ir,image_name))
        depth = cv2.imread(path.join(bak_depth,image_name))

        profile_h,profile_w = profile.shape[0:2]
        ir_h,ir_w = ir.shape[0:2]
        depth_h,depth_w = depth.shape[0:2]

        (x,y,w,h) = rect_img(profile)
        #print(x,y,w,h)
        profile = profile[y:y+h,x:x+w]
        ir = ir[int(y*ir_h/profile_h):int((y+h)*ir_h/profile_h),int(x*ir_w/profile_w):int((x+w)*ir_w/profile_w)]
        depth = depth[int(y*depth_h/profile_h):int((y+h)*depth_h/profile_h),int(x*depth_w/profile_w):int((x+w)*depth_w/profile_w)]

        cv2.imwrite(path.join(work_profile,image_name),profile)
        cv2.imwrite(path.join(work_ir,image_name),ir)
        cv2.imwrite(path.join(work_depth,image_name),depth)
