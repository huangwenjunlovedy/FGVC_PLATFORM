import os
import shutil
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

path = 'G:\\github_files\\datasets\\CAR\\train'
data_path = 'G:\\github_files\\datasets\\Aircraft\\test'

class_list = os.listdir(data_path)
print(class_list)
count = 1
with open("./aircraft_classes.txt", 'w') as f:
    for i in class_list:
        f.write(("%03d" % count) + '.' + i)  # 将 1 变成 001
        f.write('\n')
        count += 1


