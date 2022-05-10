from torchvision import transforms

def get_dataset_cls(image_name):
    dataset_name = image_name.split('/')[-4]
    if dataset_name == 'CUB':
        num_class = 200
    elif dataset_name == 'CAR':
        num_class = 196
    elif dataset_name == 'Aircraft':
        num_class = 100
    else:
        raise ValueError('Unknown dataset!!!')
    return num_class


def get_augmentation_config():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose([
        transforms.Resize(size=600),
        transforms.RandomResizedCrop(size=448),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(size=600),
        transforms.CenterCrop(size=448),
        transforms.ToTensor(),
        normalize
    ])
    return train_transforms, test_transforms