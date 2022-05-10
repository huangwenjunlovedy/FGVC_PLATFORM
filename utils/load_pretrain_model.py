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

        path_model = 'G:\\github_files\\FGVC_platform\models\\cub_pgd_1400ep_448_best.pth.tar'  # the first pretrained model
        # path_model = 'G:\\github_files\\demonstration_platform\model\\cub_baseline.tar'   # baseline

        # chapter_3
        ### baseline
        # path_model = 'C:\\Users\\YXB\\Desktop\sim_ft_models\CUB\\resnet_epochs_300_size_224_alpha_1.0\\model_best.pth.tar' # b-78.1
        # path_model = 'C:\\Users\\YXB\Desktop\\sim_ft_models\\CUB\\resnet_epochs_300_size_448_alpha_1.0\\model_best.pth.tar'  # b-81.9

        ### BD
        # path_model = 'C:\\Users\\YXB\\Desktop\\sim_ft_models\\CUB\\simre50_epochs_300_size_224_alpha_1.0\\model_best.pth.tar' # 224b-77.9
        # path_model = 'C:\\Users\\YXB\Desktop\\sim_ft_models\\CUB\\simres50_epochs_300_size_448_alpha_1.0.tar'  # b-81.84

        ### AD

        # path_model = 'C:\\Users\\YXB\\Desktop\\sim_ft_models\\AD_CUB\\ad_epochs_300_size_448_alpha_1.0\\model_best.pth.tar'  # 81.9
        # path_model = 'C:\\Users\\YXB\\Desktop\\sim_ft_models\\AD_CUB\\ad_epochs_300_size_224_alpha_1.0\\model_best.pth.tar'

        # change output_features
        num_fc_ftr = model.fc.in_features
        model.fc = torch.nn.Linear(num_fc_ftr, 200)
        checkpoint = torch.load(path_model, map_location="cpu")
        state_dict = checkpoint['state_dict']
    elif dataset_name == 'CAR':     # 这里代码都有问题，因为和 car 和 aircraft 模型不匹配
        # chapter_4
        path_model = 'G:\\github_files\\FGVC_platform\\models\\pgd_car_1400ep_448_best.pth.tar'
        # chapter_3
        ### baseline
        # path_model = 'C:\\Users\\YXB\Desktop\\sim_ft_models\\CAR\\resnet50_epochs_300_size_224_alpha_1.0\\model_best.pth.tar' # c-92.09
        # path_model = 'C:\\Users\\YXB\\Desktop\\sim_ft_models\\CAR\\renset50_epochs_300_size_448_alpha_1.0\\checkpoint.pth.tar'  # b-92.17

        ### BD

        # path_model = 'C:\\Users\\YXB\Desktop\\sim_ft_models\\CAR\\simres_epochs_300_size_224_alpha_1.0\\checkpoint.pth.tar'   # b-92.5 c-92.3
        # path_model = 'C:\\Users\\YXB\\Desktop\\sim_ft_models\\CAR\\simres_epochs_300_size_448_alpha_1.0\\model_best.pth.tar'  # b-92.46 c-92.17

        ### AD

        # path_model = 'C:\\Users\\YXB\\Desktop\\sim_ft_models\AD_car\\ad_epochs_300_size_448_alpha_1.0\\model_best.pth.tar' # 92.68
        # path_model = 'C:\\Users\\YXB\\Desktop\\sim_ft_models\AD_car\\ad_epochs_300_size_224_alpha_1.0\\model_best.pth.tar' # 92.14

        # change output_features
        num_fc_ftr = model.fc.in_features
        model.fc = torch.nn.Linear(num_fc_ftr, 196)
        checkpoint = torch.load(path_model, map_location="cpu")
        state_dict = checkpoint['state_dict']
    elif dataset_name == 'Aircraft':
        # chapter_4
        path_model = 'G:\\github_files\\FGVC_platform\\models\\aircraft_pgd_1400_448_bestl.tar'  # the first pretrained model

        # chapter_3
        ### baseline
        # path_model = 'C:\\Users\\YXB\\Desktop\\sim_ft_models\\Aircraft\\resnet_epochs_300_size_224_alpha_1.0\\model_best.pth.tar'  # b-88.419
        # path_model = 'C:\\Users\\YXB\\Desktop\\sim_ft_models\\Aircraft\\resnet_epochs_300_size_448_alpha_1.0\\model_best.pth.tar' # 90.2

        ### BD
        # path_model = 'C:\\Users\\YXB\\Desktop\\sim_ft_models\\Aircraft\\simres_epochs_300_size_224_alpha_1.0_\\model_best.pth.tar'  # b-88.77
        # path_model = 'C:\\Users\\YXB\Desktop\\sim_ft_models\\Aircraft\\simres_epochs_300_size_448_alpha_1.0\\model_best.pth.tar' # b-90.6

        ### AD
        # path_model = 'C:\\Users\\YXB\\Desktop\\sim_ft_models\\AD_Aircraft\\ad_epochs_300_size_448_alpha_1.0\\model_best.pth.tar'  # 90.45
        # path_model = 'C:\\Users\\YXB\\Desktop\\sim_ft_models\\AD_Aircraft\\ad_epochs_300_size_224_alpha_1.0\\model_best.pth.tar' # 88.5

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