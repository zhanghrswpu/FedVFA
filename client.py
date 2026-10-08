import copy
import json
from collections import defaultdict

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, SubsetRandomSampler

import utils
from loss import FedLC, FedProx, MoonLoss, FedDynLoss, FedRs, FedFA
from loss import generated_feature_data



class Client:

    def __init__(self, id, datasets, args):
        self.weights = None
        self.id = id
        self.batch_size = args.batch_size
        X, y = list(zip(*datasets))
        train_X, train_y, test_X, test_y = utils.data_split(
            np.array(X), np.array(tuple(tensor.numpy() for tensor in y)))  # np.array(y)
        self.classes_count = utils.statistic_classes_count(train_y)
        self.data_count = len(train_X)

        self.train_dataloader = DataLoader(dataset=list(zip(train_X, train_y)), batch_size=args.batch_size,
                                           sampler=SubsetRandomSampler(list(range(len(train_X)))))
        self.test_dataloader = DataLoader(dataset=list(
            zip(test_X, test_y)), batch_size=args.batch_size)

        self.optimizer = None
        self.device = args.device
        self.args = args

    def get_model_parameters(self):
        return dict(self.model.state_dict())

    def train(self, global_model, epoch):
        self.model = copy.deepcopy(global_model)

        self.model.train()
        # 将模型转移到gpu
        self.model.to(self.args.device)

        if self.args.loss == CrossEntropyLoss:
            self.loss_func = self.args.loss()
        elif self.args.loss == FedLC:
            self.loss_func = self.args.loss(
                self.args.fedlc_tau, list(self.classes_count.values()))
        elif self.args.loss == FedProx:
            for param in global_model.parameters():
                # 由于FedProx要在loss中使用到此前的model，所以进行这个设置，否则浪费资源
                # 计算完成之后需要重新设置为True
                param.requires_grad = False
            global_model = global_model.to(self.args.device)
            self.loss_func = self.args.loss(
                global_model, mu=self.args.fedprox_mu)
        elif self.args.loss == FedRs:
            self.loss_func = self.args.loss(
                self.classes_count, alpha=self.args.fedrs_alpha)
        else:
            raise Exception("the loss dose not exists:", self.args.loss)

        self.optimizer = self.args.optimizer(
            self.model.parameters(), **self.args.optimizer_parameters)

        loss, correct = utils.batch_train(self.model, self.optimizer, self.loss_func,
                                          self.train_dataloader,
                                          self.args.local_epochs,
                                          self.device,
                                          self.args.algorithm != "FedAvg",
                                          False)
        if self.args.loss == FedProx:
            for param in global_model.parameters():
                param.requires_grad = True
            global_model.to("cpu")

        self.clients_weight_update = self.get_model_parameters()
        self.optimizer = None
        return correct, loss

    def test(self, global_model=None, test_dataloader=None):
        model = global_model
        model.eval()
        model.to(self.args.device)
        class_acc_dict = defaultdict(int)
        for x, y in test_dataloader:
            x, y = x.to(self.args.device), y.to(self.args.device)
            _, out = model(x)
            for i, label in enumerate(y):
                label = torch.argmax(label)
                if torch.argmax(out[i]) == label:
                    class_acc_dict[int(label)] += 1

        for label, count in self.args.test_class_count.items():
            class_acc_dict[label] /= count
        model.to("cpu")
        return sum(class_acc_dict.values()) / len(class_acc_dict.keys()), class_acc_dict


class FedFAClient(Client):
    def __init__(self, id, datasets, n_dim, args):
        super(FedFAClient, self).__init__(id, datasets, args)
        self.anchor_feature = torch.tensor(utils.generate_matrix(n_dim, args.num_classes)).to(args.device)

    def train(self, global_model, epoch):
        self.model = copy.deepcopy(global_model)
        self.model.train()
        # 将模型转移到gpu
        self.model.to(self.args.device)
        global_model.to(self.args.device)

        self.loss_func = FedFA(self.model, self.anchor_feature)
        self.optimizer = self.args.optimizer(
            self.model.parameters(), **self.args.optimizer_parameters)

        loss, features, correct = utils.batch_train(self.model, self.optimizer, self.loss_func,
                                                    self.train_dataloader,
                                                    self.args.local_epochs,
                                                    self.device, return_features=False,
                                                    label_count=self.args.num_classes)
        self.anchor_feature = list(features.values())
        global_model.cpu()

        self.clients_weight_update = self.get_model_parameters()
        self.optimizer = None
        return correct, loss


class MoonClient(Client):

    def __init__(self, id, datasets, args):
        super(MoonClient, self).__init__(id, datasets, args)
        self.pre_model = None

    def train(self, global_model, epoch):
        self.model = copy.deepcopy(global_model)
        self.model.train()
        # 将模型转移到gpu
        self.model.to(self.args.device)
        global_model.to(self.args.device)

        self.loss_func = MoonLoss(
            tau=self.args.fedmoon_tau, mu=self.args.fedmoon_mu)

        self.optimizer = self.args.optimizer(
            self.model.parameters(), **self.args.optimizer_parameters)

        if self.pre_model is None:
            self.pre_model = global_model

        self.pre_model.to(self.args.device)

        loss, correct = utils.moon_batch_train(self.model, self.pre_model, global_model,
                                               self.optimizer, self.loss_func,
                                               self.train_dataloader,
                                               self.args.local_epochs,
                                               self.device)

        self.pre_model = copy.deepcopy(self.model).cpu()
        self.pre_model.eval()
        global_model.cpu()

        self.clients_weight_update = self.get_model_parameters()
        self.optimizer = None
        return correct, loss


class FedLossClient(Client):
    def __init__(self, id, datasets, args):
        super(FedLossClient, self).__init__(id, datasets, args)

    def feature_mean_covariance(self, features_statistics, feature_shape, args):
        res = {}
        with torch.no_grad():
            for class_ in range(args.num_classes):
                if class_ not in features_statistics.keys():
                    res[class_] = [torch.zeros(size=feature_shape, device=self.args.device),
                                   torch.zeros(size=feature_shape, device=self.args.device)]
                else:
                    v = features_statistics[class_]
                    v = torch.stack(v)
                    mean = torch.mean(v, dim=0)
                    if args.std_required:
                        if len(v) == 1:
                            std = 0
                        else:
                            std = torch.sqrt(
                                torch.sum((v - mean) ** 2, dim=0) / (v.size()[0] - 1))
                    if args.std_required:
                        res[int(class_)] = [mean, std]
                    else:
                        res[int(class_)] = [mean]
            return res

    def train(self, global_model, epoch, feature_statistics):
        self.model = copy.deepcopy(global_model)
        self.model.train()
        # 将模型转移到gpu
        self.model.to(self.args.device)
        global_model.to(self.args.device)

        if epoch > 0:
            features, labels = generated_feature_data(feature_statistics, self.args.num_classes, self.args.std_required,
                                                      None)
            labels = torch.zeros((len(labels), self.args.num_classes)).scatter_(
                1, labels.long().reshape(-1, 1), 1)
            labels = labels.to("cuda")
            features = features.to("cuda")
        else:
            features, labels = None, None

        self.loss_func = self.args.loss(
            self.model, self.args.beta, epoch, features, labels)

        self.optimizer = self.args.optimizer(
            self.model.parameters(), **self.args.optimizer_parameters)

        loss, features, correct = utils.batch_train(self.model, self.optimizer, self.loss_func,
                                                    self.train_dataloader,
                                                    self.args.local_epochs,
                                                    self.device, return_features=True)

        feature_shape = list(features.values())[0][0].shape
        self.features_statistics = self.feature_mean_covariance(
            features, feature_shape, self.args)

        self.clients_weight_update = self.get_model_parameters()
        self.optimizer = None
        return correct, loss


class FedDynClient(Client):

    def __init__(self, id, datasets, args):
        super(FedDynClient, self).__init__(id, datasets, args)
        self.nabla = None

    def train(self, global_model, epoch):
        if self.nabla is None:
            self.nabla = self.vectorize(global_model).detach().clone().zero_()

        self.nabla = self.nabla.to(self.args.device)
        self.model = copy.deepcopy(global_model)
        self.model.train()
        # 将模型转移到gpu
        self.model.to(self.args.device)

        self.loss_func = FedDynLoss(
            global_model, self.nabla, self.args.feddyn_alpha)

        self.optimizer = self.args.optimizer(
            self.model.parameters(), **self.args.optimizer_parameters)

        loss, correct = utils.batch_train(self.model,
                                          self.optimizer, self.loss_func,
                                          self.train_dataloader,
                                          self.args.local_epochs,
                                          self.device)

        with torch.no_grad():
            # 当前的梯度等于上一次训练的梯度，减去这一次的模型更新的值乘上alpha
            self.nabla = self.nabla - self.loss_func.alpha * (
                    self.loss_func.vectorized_curr_params - self.loss_func.vectorized_global_params
            )

        self.nabla.to("cpu")
        self.clients_weight_update = self.get_model_parameters()
        self.optimizer = None
        return correct, loss

    def vectorize(self, src):
        return torch.cat([param.flatten() for param in src.parameters()])


class FedSubLossClient(Client):
    def __init__(self, id, datasets, means, args):
        super(FedSubLossClient, self).__init__(id, datasets, args)
        self.means = means

    def train(self, global_model, epoch):
        def custom_compute(x, y, model, loss_func):
            feature, logit = model(x)
            loss = loss_func(feature, logit, y)
            return loss, logit

        self.model = copy.deepcopy(global_model)
        self.model.train()
        # 将模型转移到gpu
        self.model.to(self.args.device)

        self.loss_func = self.args.loss(self.model, self.args.alpha, self.args.beta, self.args.anchor_count,
                                        self.args.variance,
                                        self.means, epoch)

        self.optimizer = self.args.optimizer(
            self.model.parameters(), **self.args.optimizer_parameters)

        loss, correct = utils.batch_train(self.model, self.optimizer, self.loss_func,
                                          self.train_dataloader,
                                          self.args.local_epochs,
                                          self.device, custom_loss_compute=custom_compute,
                                          return_features=False)

        # 设置模型更新到client中
        self.clients_weight_update = self.get_model_parameters()
        self.optimizer = None
        return correct, loss


class FedVFAClient(Client):
    def __init__(self, id, datasets, means, args):
        super(FedVFAClient, self).__init__(id, datasets, args)
        self.means = means

    def train(self, global_model, epoch):
        def custom_compute(x, y, model, loss_func):
            feature, logit = model(x)
            loss = loss_func(feature, logit, y)
            return loss, logit

        self.model = copy.deepcopy(global_model)
        self.model.train()
        # 将模型转移到gpu
        self.model.to(self.args.device)

        self.loss_func = self.args.loss(self.model, self.args.alpha, self.args.beta, self.means,
                                        self.args.enable_after_adjust)

        self.optimizer = self.args.optimizer(
            self.model.parameters(), **self.args.optimizer_parameters)

        loss, correct = utils.batch_train(self.model, self.optimizer, self.loss_func,
                                          self.train_dataloader,
                                          self.args.local_epochs,
                                          self.device, custom_loss_compute=custom_compute,
                                          return_features=False)

        # 设置模型更新到client中
        self.clients_weight_update = self.get_model_parameters()
        self.optimizer = None
        return correct, loss
