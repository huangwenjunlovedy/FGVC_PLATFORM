"""
@author:yangxb23
@github:https://github.com/xuebin-yang/
@data:25/12/2021
@func:dataset preprocess
"""


import torch
import numpy as np
import os
import shutil

def fun():
    path = "G:\\github_files\\datasets\\Aircraft\\"
    files = os.listdir(path)
    f = open('./files/aircraft_images.txt', 'r+')
    for i in files:
        path_2 = path + i   # G:\github_files\datasets\CUB\train
        files_2 = os.listdir(path_2)
        for j in files_2:
            path_3 = path_2 + '\\' + j   # # G:\github_files\datasets\CUB\\train\\186.Cedar_Waxwing
            img_files = os.listdir(path_3)
            for m in img_files:
                img_path = path_3 + '\\' + m
                f.write(img_path + '\n')
                # shutil.copy(img_path, 'G:\github_files\datasets\CUB_images')
    f.close()




if __name__ == "__main__":
    fun()