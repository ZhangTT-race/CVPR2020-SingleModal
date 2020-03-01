# CVPR2020-SingleModal

Code for competition : Chalearn Single-modal Cross-ethnicity Face anti-spoofing Recognition Challenge@CVPR2020

we use RGB data for training.

## commit

The date of last commmit  is 03/01/2020 ,our result is comes from the last version.

## Prerequisites

We use Anaconda3 with python 3.7 , we use  :

> opencv-python  4.2.0

>  pytorch 1.4.0

>  imutils 0.5.3

>  numpy 1.18.1

in the enviroment.

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

Set your datapath and checkpoint path and coda ，in train_gray_rgb_4@1.py train_gray_rgb_4@2.py train_gray_rgb_4@3.py  

```shell
./train.sh
```

## Test

set your global_models、model_names 、global_actions and data_root  valid_gray.py

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