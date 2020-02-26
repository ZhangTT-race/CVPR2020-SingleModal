# CVPR2020-SingleModal

Code for competition : Chalearn Single-modal Cross-ethnicity Face anti-spoofing Recognition Challenge@CVPR2020

we use RGB data for training.

## Prerequisites

We use Anaconda3 with python 3.7 , we use  :

> opencv-python  4.2.0

>  pytorch 1.4.0

>  imutils 0.5.3

>  numpy 1.18.1

in your enviroment.

## Data pre-progress

Edit cut_face.py ,set your data path. 

```python
global_action = "dev" #"train" or "dev"
data_root = "/Users/wdh/Downloads/CASIA-CeFA/%s"%global_action
```

Change global_action = "train" ,to cut negative data in train

```shell
python cut_face.py
```

Change global_action = "dev" ,to cut **negative** data in dev

```python
python cut_face.py
```

#### Final data index tree

like this

```shell
├── data
    ├── dev
        ├── 003001
        		├── depth
    				├── depth_bak
    				├── ir
    				├── ir_bak
    				├── profile
    				├── profile_bak	
 ... ...
```

## Train

Set your datapath and checkpoint path and coda 

```shell
python train_gray_rgb_4@1.py
python train_gray_rgb_4@2.py
python train_gray_rgb_4@3.py
```

## Test

set your global_model 、global_action 、data_root 、 checkpoints_root and model_name in valid_gray.py

```python
global_model = "4@1" # "4@1"、"4@2"、"4@3"
global_action = "test" #"test" or "dev"

data_root = "/Users/wdh/Downloads/CASIA-CeFA/"
checkpoints_root = "./checkpoints/gray%s"%global_model
model_name = "23000_loss_0.0020.pth"
```

run valid_gray.py each time after you change global_model and  global_action , or you change the code to make it automatically.

```shell
python valid_gray.py
```

you will  get six txt file :

 ```shell
4\@1_dev_gray_res.txt 4\@1_test_gray_res.txt 4\@2_dev_gray_res.txt 4\@2_test_gray_res.txt 4\@3_dev_gray_res.txt 4\@3_test_gray_res.txt
 ```

​	then ：

```shell
./merge.sh
```

finally get the submission.txt 