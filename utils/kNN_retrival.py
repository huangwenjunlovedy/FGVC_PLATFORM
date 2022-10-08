"""
@author:yangxb23
@github:https://github.com/xuebin-yang/
@data:02/12/2021
"""

import os
import torch
import numpy as np
import torchvision
import torch.backends.cudnn as cudnn
from data.config import get_dataset_cls, get_augmentation_config
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from data.dataloader import get_data_loader_init
from PIL import Image


def feature_extract(data_loader, model):
    model.cuda()
    model.eval()
    features = {'data': [], 'target': [], 'all_img_path': []}
    with torch.no_grad():
        for i, (input, target, img_path) in enumerate(data_loader):
            inputs = input.cuda()
            output = model(inputs)
            # print(output.size())  # [48, 2048, 1, 1]
            output = np.squeeze(output)
            for j in range(output.size(0)):
                features['data'].append(output[j].cpu().numpy())
                features['target'].append(target[j].cpu().numpy())
                features['all_img_path'].append(img_path[j])

            if i % 10 == 0:
                print("{}/{} finished".format(i, len(data_loader)))
    return features, img_path


def query_img_feature_extract(image_path, model):
    model.cuda()
    model.eval()
    feature = {'data': []}
    with torch.no_grad():
        img = Image.open(image_path).convert('RGB')
        img = np.array(img)
        img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).float()
        input = img.cuda()
        output = model(input)
        # print(output.size())  # [1, 2048, 1, 1]
        output = np.squeeze(output)
        feature['data'].append(output.cpu().numpy())
    return feature

def knn_retrival(image_name, model):
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # close the warning

    # setting the model
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    print(model)

    # get the features
    query_feature = query_img_feature_extract(image_name, model)

    train_transforms, test_transforms = get_augmentation_config()
    _, _, train_data_loader, val_data_loader = get_data_loader_init(train_transforms, test_transforms, image_name)

    # 改变模型需要重新提取和保存训练集特征
    # train_features, img_path = feature_extract(train_data_loader, model)
    # dir = './features/{}'.format(image_name.split('/')[-4])
    # if not os.path.exists(dir):
    #     os.makedirs(dir)
    # np.save('{}/train_set_feature.npy'.format(dir), train_features)

    # load the saved features
    dir = 'G:/github_files/fine_grained_classfication_platform/features/{}'.format(image_name.split('/')[-4])
    train_features = np.load(dir + '/train_set_feature.npy', allow_pickle=True).item()
    print('train_features_path=', dir + '/train_set_feature.npy')
    X_train = train_features['data']
    x_img_path = train_features['all_img_path']

    ks = [1, 5, 10, 20, 50]
    topk_correct = {}

    distances = cosine_distances(query_feature['data'], X_train)   # [1, 5994]
    indices = np.argsort(distances)  # sort from small to big

    for k in ks:
        topk_list = []
        top_k_indices = indices[:, :k]
        for ind in top_k_indices:
            for ind_ind in ind:
                # print(ind_ind)
                topk_list.append(str(x_img_path[ind_ind]))
        topk_correct[str(k)] = topk_list
    return topk_correct['20']

if __name__ == "__main__":
    image_name = "G:/github_files/datasets/CUB/train/001.Black_footed_Albatross/Black_Footed_Albatross_0017_796098.jpg"
    model = torchvision.models.resnet50(pretrained=True)
    knn_retrival(image_name, model)