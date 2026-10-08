import json
from collections import defaultdict

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam

import utils
from selector import RandomSampler, ClusterSampler
import model
from aggregation import FedAvg, FedNA, FedCATDD
from client import Client, MoonClient, FedDynClient, FedLossClient, FedSubLossClient, FedFAClient, FedVFAClient
from dataset import mnist, fashion_mnist, emnist, cifar10, svhn, caltech101, cifar100
from loss import FedLoss, FedProx, FedLC, MoonLoss, FedDynLoss, FedRs, FedSubLoss, FedFA, FedVFA


def prepare_testloader(args, test_loader):
    class_count = defaultdict(int)
    for x, y in test_loader:
        _, true_label = torch.max(y.data, 1)
        for label in true_label:
            class_count[int(label.numpy())] += 1
    # 记录测试集中各个类别的数据分布
    args.test_class_count = class_count


def get_feature_dim(model):
    if model == "ResNet18":
        return 512
    elif model == "Net":
        return 128
    elif model == "LeNet":
        return 84
    elif model == "ResNet20":
        return 64
    elif model == "ResNet32":
        return 64
    elif model == "MobileNetV2":
        return 1280
    elif model == "CIFAR100Model":
        return 256
    else:
        raise Exception("model not exists")


def create_client(X, y, args):
    n_dim = get_feature_dim(args.model)
    if args.enable_dirichlet:
        client_data = utils.split_noniid(X, y, alpha=args.dirichlet_alpha, n_clients=args.client_count,
                                         draw_data_distribution=args.draw_data_distribution)
    else:
        client_data = utils.split_sequence(
            X, y, count=args.each_class_count, n_clients=args.client_count)
    clients = []
    zero_count = 0
    max_count, min_count = 0, 100000
    if args.loss == FedSubLoss or args.loss == FedVFA:
        with open("./test/feature_mean_{}_{}.json".format(n_dim, args.num_classes), "r") as f:
            means = json.load(f)
            means = torch.tensor(means, requires_grad=False).to(args.device)
    for i in range(args.client_count):
        if len(client_data[i]) <= 0:
            # 如果客户端没有数据，不加入训练
            zero_count += 1
            continue
        max_count = max(max_count, len(client_data[i]))
        min_count = min(min_count, len(client_data[i]))
        if args.loss == MoonLoss:
            clients.append(MoonClient(i, client_data[i], args))
        elif args.loss == FedDynLoss:
            clients.append(FedDynClient(i, client_data[i], args))
        elif args.loss == FedLoss:
            clients.append(FedLossClient(i, client_data[i], args))
        elif args.loss == FedSubLoss:
            clients.append(FedSubLossClient(
                i, client_data[i], means, args))
        elif args.loss == FedVFA:
            clients.append(FedVFAClient(
                i, client_data[i], means, args))
        elif args.loss == FedFA:
            clients.append(FedFAClient(
                i, client_data[i], n_dim, args))
        else:
            clients.append(Client(i, client_data[i], args))
    return clients


def create_model(args):
    return eval("model.{}({},{})".format(args.model, args.channel, args.num_classes))


def create_dataset(args):
    if args.dataset == "mnist":
        args.channel = 1
        args.num_classes = 10
        return mnist()
    elif args.dataset == "fmnist":
        args.channel = 1
        args.num_classes = 10
        return fashion_mnist()
    elif args.dataset == "emnist":
        args.channel = 1
        args.num_classes = 47
        return emnist()
    elif args.dataset == "cifar10":
        args.channel = 3
        args.num_classes = 10
        return cifar10()
    elif args.dataset == "svhn":
        args.channel = 3
        args.num_classes = 10
        return svhn()
    elif args.dataset == "cifar100":
        args.channel = 3
        args.num_classes = 100
        return cifar100()
    else:
        raise Exception("dataset does not exist")


def create_algorithm(args, clients):
    if args.algorithm == "FedAvg":
        return FedAvg(clients)
    elif args.algorithm == "FedNA":
        return FedNA(clients)
    elif args.algorithm == "FedCA-TDD":
        return FedCATDD(clients)
    else:
        raise Exception("algorithm does not exist")


def prepare_loss(args):
    if args.loss == "CE":
        args.loss = CrossEntropyLoss
    elif args.loss == "FedLoss":
        args.loss = FedLoss
    elif args.loss == "FedLC":
        args.loss = FedLC
    elif args.loss == "FedProx":
        args.loss = FedProx
    elif args.loss == "MoonLoss":
        args.loss = MoonLoss
    elif args.loss == "FedDyn":
        args.loss = FedDynLoss
    elif args.loss == "FedRs":
        args.loss = FedRs
    elif args.loss == "FedSubLoss":
        args.loss = FedSubLoss
    elif args.loss == "FedVFA":
        args.loss = FedVFA
    elif args.loss == "FedFA":
        args.loss = FedFA
    else:
        raise Exception("loss function does not exist")


def create_sampler(args, clients, X):
    if args.sampler == "random":
        return RandomSampler(clients, args.choice_count)
    elif args.sampler == "cluster":
        return ClusterSampler(clients, len(X), args.choice_count)
    else:
        raise Exception("loss function does not exist")


def prepare_optimizer(args):
    if args.optimizer == "sgd":
        args.optimizer = SGD
        args.optimizer_parameters = {"lr": args.learning_rate, "momentum": args.momentum,
                                     "weight_decay": args.weight_decay}
    elif args.optimizer == "adam":
        args.optimizer = Adam
        args.optimizer_parameters = {"lr": args.learning_rate,
                                     "weight_decay": args.weight_decay}
    else:
        raise Exception("optimizer does not exist")
