#!/bin/bash
nohup python -u train_gray_rgb_4\@1.py > log_train_4@1.nohup &
nohup python -u train_gray_rgb_4\@2.py > log_train_4@2.nohup &
nohup python -u train_gray_rgb_4\@3.py > log_train_4@3.nohup &