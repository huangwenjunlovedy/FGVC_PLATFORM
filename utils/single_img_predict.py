# -*- coding: utf-8 -*-
"""
@author:yangxb23
@github:https://github.com/xuebin-yang/
@data:02/12/2021
"""

import time
import torch
import torchvision
import PIL
import cv2
from PIL import Image
import torchvision.transforms as transforms

def get_cub_class():
    with open("G:\\github_files\\FGVC_platform\\files\\cub_classes.txt", "r", encoding='UTF-8') as f:
        classes = f.readlines()
        classes_copy = []
        for i in classes:
            classes_copy.append(i.split(".")[-1].strip())
    return classes_copy


def get_car_class():
    with open("G:\\github_files\\FGVC_platform\\files\\car_classes.txt", "r", encoding='UTF-8') as f:
        classes = f.readlines()
        classes_copy = []
        for i in classes:
            classes_copy.append(i.split(".")[-1].strip())
    return classes_copy


def get_aircraft_class():
    with open("G:\\github_files\\FGVC_platform\\files\\aircraft_classes.txt", "r", encoding='UTF-8') as f:
        classes = f.readlines()
        classes_copy = []
        for i in classes:
            classes_copy.append(i.split(".")[-1].strip())
    return classes_copy


def augmentation_fun(img):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    augmentation = transforms.Compose([
        transforms.Resize(size=600),
        transforms.CenterCrop(size=448),
        transforms.ToTensor(),
        normalize
    ])
    return augmentation(img)

def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition  # broadcast

def get_pred_labels(dataset_name, index):
    if dataset_name == 'CUB':
        cub_labels = get_cub_class()
        return cub_labels[index]
    elif dataset_name == 'CAR':
        car_labels = get_car_class()
        return car_labels[index]
    elif dataset_name == 'Aircraft':
        aircraft_labels = get_aircraft_class()
        return aircraft_labels[index]
    else:
        raise("not support dataset!!!")


def eval_single_img(img_name, model):   #img->one picture
    # start = time.time()
    dataset_name = img_name.split('/')[-4]  # dataset name

    img = Image.open(img_name).convert('RGB')
    img = augmentation_fun(img)
    img = img.unsqueeze(0).cuda()
    # model = load_pretraned_model(dataset_name).cuda()
    model.cuda().eval()

    logits = model(img)
    confi = softmax(logits)

    pred_cls_index = confi.argmax(dim=1)   #
    pre_cls_value = confi.max(dim=1)  # biggest confidence

    pred_class = get_pred_labels(dataset_name, pred_cls_index)
    return pred_class, pre_cls_value[0].item()

def load_pretraned_model(dataset_name):
    # load the weight of models
    if dataset_name == 'CUB':
        path_model = 'G:\\github_files\\fine_grained_classfication_platform\model\\cub_pretrained_finetune_500ep.tar'
        num_cls = 200
    elif dataset_name == 'CAR':
        pass
    elif dataset_name == 'Aircraft':
        pass
    else:
        raise('not support models!!!')

    model = torchvision.models.resnet50(pretrained=False)
    # change output_features
    num_fc_ftr = model.fc.in_features
    model.fc = torch.nn.Linear(num_fc_ftr, num_cls)
    checkpoint = torch.load(path_model, map_location="cpu")
    state_dict = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module'):
            # remove prefix
            state_dict[k[len("module."):]] = state_dict[k]

    msg = model.load_state_dict(state_dict, strict=False)
    print(set(msg.missing_keys))
    return model


if __name__ == "__main__":
    img_name = "G:\\github_files\\datasets\\CUB\\train\\001.Black_footed_Albatross\\Black_Footed_Albatross_0017_796098.jpg"
    predict_class = eval_single_img(img_name)
    print(predict_class)







