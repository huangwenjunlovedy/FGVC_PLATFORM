# -*- coding: utf-8 -*-
"""
@author:yangxb23
@github:https://github.com/xuebin-yang/
@data:02/12/2021
"""

import torch
import os
import numpy as np

def read_cub_img():
    with open("G:\\github_files\\FGVC_platform\\files\\cub_images.txt", "r") as f:  # 打开文件
        cub_images = f.readlines()  # 读取文件
        # print(cub_images)
        return cub_images

def read_car_img():
    with open("G:\\github_files\\FGVC_platform\\files\\car_images.txt", "r") as f:  # 打开文件
        car_images = f.readlines()  # 读取文件
        # print(car_images)
        return car_images

def read_aircraft_img():
    with open("G:\\github_files\\FGVC_platform\\files\\aircraft_images.txt", "r") as f:  # 打开文件
        aircraft_images = f.readlines()  # 读取文件
        # print(aircraft_images)
        return aircraft_images