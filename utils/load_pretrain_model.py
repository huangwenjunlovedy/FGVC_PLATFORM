# -*- coding: utf-8 -*-
"""
@author:yangxb23
@github:https://github.com/xuebin-yang/
@data:02/12/2021
"""


import torch
import torchvision
import numpy as np

def load_model(dataset_name):
    model = torchvision.models.resnet50(pretrained=False)
    if dataset_name == 'CUB':
        # chapter_4
        path_model = 'G:\\github_files\\FGVC_platform\models\\cub_pgd_1200ep_448_best.pth.tar'  # the first pretrained model

        # change output_features
        num_fc_ftr = model.fc.in_features
        model.fc = torch.nn.Linear(num_fc_ftr, 200)
        checkpoint = torch.load(path_model, map_location="cpu")
        state_dict = checkpoint['state_dict']
    elif dataset_name == 'CAR':     # 这里代码都有问题，因为和 car 和 aircraft 模型不匹配
        # chapter_4
        path_model = 'G:\\github_files\\FGVC_platform\\models\\pgd_car_1200ep_448_best.pth.tar'

        # change output_features
        num_fc_ftr = model.fc.in_features
        model.fc = torch.nn.Linear(num_fc_ftr, 196)
        checkpoint = torch.load(path_model, map_location="cpu")
        state_dict = checkpoint['state_dict']
    elif dataset_name == 'Aircraft':
        # chapter_4
        path_model = 'G:\\github_files\\FGVC_platform\\models\\aircraft_pgd_1200_448_best.tar'  # the first pretrained model

        # change output_features
        num_fc_ftr = model.fc.in_features
        model.fc = torch.nn.Linear(num_fc_ftr, 100)
        checkpoint = torch.load(path_model, map_location="cpu")
        state_dict = checkpoint['state_dict']
    else:
        raise ('not support dataset!!!')

    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module'):
            # remove prefix
            state_dict[k[len("module."):]] = state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    msg = model.load_state_dict(state_dict, strict=False)
    print(set(msg.missing_keys))
    return model