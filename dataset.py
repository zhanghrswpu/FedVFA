import json
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from numpy import float32
from torch.utils import data
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, EMNIST, Caltech101
from torchvision.transforms import Lambda, transforms


def mnist(mix_data=False, test_batch_size=64):
    """
    :param mix_data:是否将测试数据和train数据混合到一起组成x，y。
    :return:
    """
    train_loader = MNIST("./data/mnist_data", train=True, download=True,
                         transform=torchvision.transforms.Compose([
                             torchvision.transforms.ToTensor(),
                             torchvision.transforms.Normalize(mean=(0.1307,), std=(0.3081,)), ]))

    test_loader = MNIST("./data/mnist_data", train=False, download=True,
                        transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
                        ]), target_transform=
                        Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(
                            y), value=1)))
    train_x, train_y = [], []
    for x, y in train_loader:
        train_x.append(x.numpy())
        train_y.append(torch.tensor(y))

    if mix_data:
        for x, y in test_loader:
            train_x.append(x.numpy())
            train_y.append(torch.argmax(y))
        return np.array(train_x), np.array(train_y)

    return np.array(train_x), np.array(train_y), DataLoader(test_loader, batch_size=test_batch_size,
                                                            sampler=SubsetRandomSampler(list(range(len(test_loader)))))


def fashion_mnist(mix_data=False, test_batch_size=64):
    train_loader = FashionMNIST("./data/fashion_mnist_data", train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize([0.28604063, ], [0.32045463, ])]))

    test_loader = FashionMNIST("./data/fashion_mnist_data", train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize([0.28604063, ], [0.32045463, ]),
                               ]), target_transform=
                               Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(
                                   y), value=1)))
    train_x, train_y = [], []
    for x, y in train_loader:
        train_x.append(x.numpy())
        train_y.append(torch.tensor(y))

    if mix_data:
        for x, y in test_loader:
            train_x.append(x.numpy())
            train_y.append(torch.argmax(y))
        return np.array(train_x), np.array(train_y)

    return np.array(train_x), np.array(train_y), DataLoader(test_loader, batch_size=test_batch_size,
                                                            sampler=SubsetRandomSampler(list(range(len(test_loader)))))


def cifar10(mix_data=False, test_batch_size=64):
    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_loader = CIFAR10("./data/cifar10", train=True, download=True,
                           transform=transform_train)

    test_loader = CIFAR10("./data/cifar10", train=False, download=True,
                          transform=transform_test, target_transform=
                          Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(
                              y), value=1)))
    train_x, train_y = [], []
    for x, y in train_loader:
        train_x.append(x.numpy())
        train_y.append(torch.tensor(y))

    if mix_data:
        for x, y in test_loader:
            train_x.append(x.numpy())
            train_y.append(torch.argmax(y))
        return np.array(train_x), np.array(train_y)

    return np.array(train_x), np.array(train_y), DataLoader(test_loader, batch_size=test_batch_size)


def cifar100(mix_data=False, test_batch_size=64):
    mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),  # 数据增强
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # 加载训练集和测试集
    trainset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=True, download=True,
                                             transform=transform_train)

    testset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=False, download=True,
                                            transform=transform_test, target_transform=
                                            Lambda(
                                                lambda y: torch.zeros(100, dtype=torch.float).scatter_(0, torch.tensor(
                                                    y), value=1)))

    train_x, train_y = [], []
    for x, y in trainset:
        train_x.append(x.numpy())
        train_y.append(torch.tensor(y))

    if mix_data:
        for x, y in testset:
            train_x.append(x.numpy())
            train_y.append(torch.argmax(y))
        return np.array(train_x), np.array(train_y)

    return np.array(train_x), np.array(train_y), DataLoader(testset, batch_size=test_batch_size)


# emnist is a subset of By-merge which is balanced.
# In By-merge, because of some letters, uppercase and lowercase handwriting is basically difficult to distinguish,
# so the uppercase and lowercase of these letters are merged here to form a new classification.
# There are 15 types of merged letters [C, I, J ,K,L,M,O,P,S,U,V,W,X,Y,Z], so there are 47 categories left
def emnist(mix_data=False, test_batch_size=64):
    train_loader = EMNIST(root="./data/emnist", split="balanced", download=True,
                          train=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])
                          )
    test_loader = EMNIST(root="./data/emnist", split="balanced", download=True,
                         train=False, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ]),
                         target_transform=
                         Lambda(lambda y: torch.zeros(47, dtype=torch.float).scatter_(0, torch.tensor(
                             y), value=1)))

    train_x, train_y = [], []
    for x, y in train_loader:
        train_x.append(x.numpy())
        train_y.append(torch.tensor(y))

    if mix_data:
        for x, y in test_loader:
            train_x.append(x.numpy())
            train_y.append(torch.argmax(y))
        return np.array(train_x), np.array(train_y)

    return np.array(train_x), np.array(train_y), DataLoader(test_loader, batch_size=test_batch_size,
                                                            sampler=SubsetRandomSampler(list(range(len(test_loader)))))


def svhn(mix_data=False, test_batch_size=64):
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(),
            transforms.Normalize([0.43768448, 0.44376868, 0.4728041],
                                 [0.12008653, 0.123137444,
                                  0.10520427])])
    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.43768448, 0.44376868, 0.4728041],
                                                     [0.12008653, 0.123137444,
                                                      0.10520427])])
    train_loader = torchvision.datasets.SVHN('./data/svhn/', split='train', download=True, transform=transform_train)
    test_loader = torchvision.datasets.SVHN('./data/svhn/', split='test', download=True, transform=transform_test,
                                            target_transform=
                                            Lambda(
                                                lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(
                                                    y), value=1)))

    train_x, train_y = [], []
    for x, y in train_loader:
        train_x.append(x.numpy())
        train_y.append(torch.tensor(y))

    if mix_data:
        for x, y in test_loader:
            train_x.append(x.numpy())
            train_y.append(torch.argmax(y))
        return np.array(train_x), np.array(train_y)

    return np.array(train_x), np.array(train_y), DataLoader(test_loader, batch_size=test_batch_size,
                                                            sampler=SubsetRandomSampler(list(range(len(test_loader)))))


class FEMNIST(Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, ):
        super(FEMNIST, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        train_clients, train_groups, train_data_temp, test_data_temp = read_data("./data/FEMNIST/train",
                                                                                 "./data/FEMNIST/test")
        if self.train:
            self.dic_users = {}
            train_data_x = []
            train_data_y = []
            for i in range(len(train_clients)):
                self.dic_users[i] = set()
                l = len(train_data_x)
                cur_x = train_data_temp[train_clients[i]]['x']
                cur_y = train_data_temp[train_clients[i]]['y']
                for j in range(len(cur_x)):
                    self.dic_users[i].add(j + l)
                    train_data_x.append(np.array(cur_x[j]).reshape(28, 28))
                    train_data_y.append(cur_y[j])
            self.data = train_data_x
            self.label = train_data_y
        else:
            test_data_x = []
            test_data_y = []
            for i in range(len(train_clients)):
                cur_x = test_data_temp[train_clients[i]]['x']
                cur_y = test_data_temp[train_clients[i]]['y']
                for j in range(len(cur_x)):
                    test_data_x.append(np.array(cur_x[j]).reshape(28, 28))
                    test_data_y.append(cur_y[j])
            self.data = test_data_x
            self.label = test_data_y

    def __getitem__(self, index):
        img, target = self.data[index], self.label[index]
        img = np.array([img])
        img = Image.fromarray(img, mode='L')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)

    def get_client_dic(self):
        if self.train:
            return self.dic_users
        else:
            exit("The test dataset do not have dic_users!")


def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data


def read_data(train_data_dir, test_data_dir):
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data


class FER2013(data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        df = pd.read_csv(root + '/icml_face_data.csv')
        if train:
            self.data = df[df['Usage'] == 'Training']
        else:
            self.data = pd.concat([df[df['Usage'] == 'PublicTest'], df[df['Usage'] == 'PrivateTest']])

        self._samples = [
            (
                torch.tensor([int(idx) for idx in row["pixels"].split()], dtype=torch.uint8).reshape(48, 48),
                int(row["emotion"]) if "emotion" in row else None,
            )
            for index, row in self.data.iterrows()
        ]

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, idx):
        image_tensor, target = self._samples[idx]
        image = Image.fromarray(image_tensor.numpy())

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target


def fer2013(mix_data, test_batch_size=64):
    train_loader = FER2013("./data/fer2013", train=True,
                           transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ]))
    test_loader = FER2013("./data/fer2013", train=False,
                          transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ]),
                          target_transform=
                          Lambda(lambda y: torch.zeros(7, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))

    train_x, train_y = [], []
    for x, y in train_loader:
        train_x.append(x.numpy())
        train_y.append(torch.tensor(y))

    if mix_data:
        for x, y in test_loader:
            train_x.append(x.numpy())
            train_y.append(torch.argmax(y))
        return np.array(train_x), np.array(train_y)

    return np.array(train_x), np.array(train_y), DataLoader(test_loader, batch_size=test_batch_size,
                                                            sampler=SubsetRandomSampler(list(range(len(test_loader)))))


def caltech101(mix_data=False, test_batch_size=4):
    RGB_MEAN = [0.5429, 0.5263, 0.4994]
    RGB_STD = [0.2422, 0.2392, 0.2406]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(256, (.8, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(RGB_MEAN, RGB_STD),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(RGB_MEAN, RGB_STD)
    ])
    train_dataset = torchvision.datasets.ImageFolder("./data/caltech101/train", transform=train_transform)
    test_dataset = torchvision.datasets.ImageFolder("./data/caltech101/test", transform=test_transform,
                                                    target_transform=
                                                    Lambda(lambda y: torch.zeros(101, dtype=torch.float).scatter_(0,
                                                                                                                  torch.tensor(
                                                                                                                      y),
                                                                                                                  value=1)))

    train_x, train_y = [], []
    for x, y in train_dataset:
        train_x.append(x.numpy())
        train_y.append(torch.tensor(y))

    if mix_data:
        for x, y in test_dataset:
            train_x.append(x.numpy())
            train_y.append(torch.argmax(y))
        return np.array(train_x), np.array(train_y)

    return np.array(train_x), np.array(train_y), DataLoader(test_dataset, batch_size=test_batch_size,
                                                            sampler=SubsetRandomSampler(list(range(len(test_dataset)))))


def caltech101_split():
    import os
    from os import path
    import numpy as np
    from sklearn.model_selection import train_test_split

    data_dir = path.join('data', 'caltech101')  # where the raw data are
    categories = os.listdir(data_dir)  # get 101 categories
    for i in range(len(categories)):

        category = categories[i]  # fetch a specific category
        cat_dir = path.join(data_dir, category)

        images = os.listdir(cat_dir)  # get all images under the category

        # images split by their names
        images_train, images_test = train_test_split(images, test_size=0.2)
        image_sets = images_train, images_test
        labels = 'train', 'test'

        # move to corresponding folders
        for image_set, label in zip(image_sets, labels):
            dst_folder = path.join(data_dir, label, category)  # create folder
            os.makedirs(dst_folder)
            for image in image_set:
                src_dir = path.join(cat_dir, image)
                dst_dir = path.join(dst_folder, image)
                os.rename(src_dir, dst_dir)  # move

        os.rmdir(cat_dir)  # remove empty folder


def number_classes(labels, num_classes):
    class_idcs = [np.argwhere(labels[range(len(labels))] == y).flatten()
                  for y in range(num_classes)]
    class_dict = {}
    for index, c in enumerate(class_idcs):
        class_dict[index] = len(c)
    print(class_dict)


def getStat(train_data, channel=1):
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(channel)
    std = torch.zeros(channel)
    for X, _ in train_loader:
        for d in range(channel):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


if __name__ == '__main__':
    X, y, test = cifar100()
    print(X)
