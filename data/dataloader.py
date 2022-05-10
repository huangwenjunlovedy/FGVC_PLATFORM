"""
@author:yangxb23
@github:https://github.com/xuebin-yang/
@data:02/12/2021
"""
import torch
import os
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


def get_data_loader_init(train_transforms, test_transforms, img_name):
    dataset_name = img_name.split('/')[-4]
    if dataset_name == 'CUB':
        root_train = 'G:/github_files/datasets/CUB/train'
        root_test = 'G:/github_files/datasets/CUB/test'
    elif dataset_name == 'CAR':
        root_train = 'G:/github_files/datasets/CAR/train'
        root_test = 'G:/github_files/datasets/CAR/test'
    elif dataset_name == 'Aircraft':
        root_train = 'G:/github_files/datasets/Aircraft/train'
        root_test = 'G:/github_files/datasets/Aircraft/test'
    else:
        raise ValueError("not support dataset!!!")
    # print('root_train', root_train)
    # print('root_test', root_test)
    train_dataset = my_Dataset(root_train, train_transforms)
    CLASS_train = train_dataset.name2label
    # print('the relationship between train data lable and filename:', CLASS_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=48, shuffle=True)

    test_dataset = my_Dataset(root_test, test_transforms)
    CLASS_test = test_dataset.name2label
    # print('the relationship between test data lable and filename', CLASS_train)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=48, shuffle=True)
    return CLASS_train, CLASS_test, train_loader, test_loader


class my_Dataset(Dataset):
    def __init__(self, root, augmentation):
        super(my_Dataset, self).__init__()
        self.root = root
        self.augmentation = augmentation
        self.all_images = {}  # 所有图像名
        self.classes = []   # 类别名
        self.name2label = {}  # 类别名的量化

        # 返回指定目录下的文件列表，并对文件列表进行排序，os.listdir每次返回目录下的文件列表顺序会不一致，排序是为了每次返回文件列表顺序一致
        for name in sorted(os.listdir(os.path.join(root))):
            # 过滤掉非目录文件
            if not os.path.isdir(os.path.join(root, name)):
                continue
            # 构建字典，名字：0~4数字
            self.name2label[name] = len(self.name2label.keys())

        if os.path.isdir(self.root):
            for fname in os.listdir(self.root):
                self.classes.append(fname)
        print("exist {} classes！".format(len(self.classes)))
        for index_cls, a in enumerate(self.classes):
            each_label = os.listdir(os.path.join(self.root, a))   # root/001.xxxxx/xxxx.jpg
            for single_img in each_label:
                self.all_images[single_img] = index_cls
        print("exist {} images!".format(len(self.all_images.keys())))

    def __len__(self):
        return len(self.all_images.keys())

    def __getitem__(self, idx):
        img_path, label = self.root + '/' + self.classes[list(self.all_images.values())[idx]] + '/' + list(self.all_images.keys())[idx], list(self.all_images.values())[idx]
        img_np = Image.open(img_path).convert('RGB')
        img_np = self.augmentation(img_np)
        label = torch.tensor(label)

        return img_np, label, img_path