import itertools
import random
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import AgglomerativeClustering, KMeans

from utils import get_similarity


def group_generate(classes):
    # 自动化生成组合
    groups = []
    for n in range(1, len(classes) + 1):
        combinations = list(itertools.combinations(classes, n))
        for comb in combinations:
            groups.append(list(comb))
    return groups


# 定义分层抽样函数

class RandomSampler:

    def __init__(self, clients, count, epoch=1000, init_sequence=False):
        self.clients = clients
        self.count = count
        self.p = self.init_p(clients)
        self.init_sequence = init_sequence
        if init_sequence:
            self.epoch = epoch
            self.__init_sequence()

    def init_p(self, clients):
        client_count = [sum(client.classes_count.values()) for client in clients]
        sum_count = sum(client_count)
        return [client_count[i] / sum_count for i in range(len(client_count))]

    def __init_sequence(self):
        client_index = range(0, len(self.clients))
        self.sequence = []
        # 初始化好每次选择的序列
        for i in range(self.epoch):
            self.sequence.append(np.random.choice(client_index, self.count, replace=False, p=self.p))
        self.cur_index = 0

    def run(self):
        if self.init_sequence:
            self.cur_index += 1
            res = []
            for i in self.sequence[self.cur_index - 1]:
                res.append(self.clients[i])
            return res
        return np.random.choice(self.clients, self.count, replace=False, p=self.p)


class ClusterSampler:

    def __init__(self, clients, total_count, count):
        self.total_count = total_count
        self.count = count
        self.clients = clients
        self.init_cluster()

    def init_cluster(self):
        label_distributions = []
        for client in self.clients:
            counts = [0] * len(client.classes_count.keys())
            for k, v in client.classes_count.items():
                counts[k] = v / self.total_count
            label_distributions.append(counts)
        # model = AgglomerativeClustering(n_clusters=self.count, compute_distances=True)
        model = KMeans(n_clusters=self.count, max_iter=30000)
        model.fit(label_distributions)
        # 各个簇的客户端下标
        self.cluster = [[] for i in range(self.count)]
        self.weights = [[] for i in range(self.count)]
        for i, label in enumerate(model.labels_):
            self.cluster[label].append(i)
        for i, clients in enumerate(self.cluster):
            for id in clients:
                self.weights[i].append(sum(self.clients[id].classes_count.values()))
            sum_weights = sum(self.weights[i])
            for j in range(len(self.weights[i])):
                self.weights[i][j] /= sum_weights

    def run(self):
        select_clients = []
        for i in range(self.count):
            # , p=self.weights[i]
            # p no p
            select_clients.append(
                self.clients[np.random.choice(self.cluster[i], 1, replace=False, p=self.weights[i])[0]])
        return select_clients


def statistics_classes_count(select_clients, num_class):
    count = [0] * num_class
    for client in select_clients:
        for k, v in client.classes_count.items():
            count[k] += v
    return count


class Sampler:

    def __init__(self, classes):
        self.groups = group_generate(classes)

    def run(self, clients):
        label_distributions = []
        for client in clients:
            label_distributions.append(tuple(client.classes_count.keys()))

        return self.stratified_sampling(label_distributions, self.groups)

    def stratified_sampling(self, label_distribution, groups):
        """
        执行分层抽样，每个组选取一定比例的人

        :param label_distribution: list, 客户端的标签拥有情况
        :param groups: list, 所有组合
        :return: list, 抽样得到的人的列表
        """
        sampled_clients = []
        selected_clients = set()
        for group in groups:
            # 找到属于这个组合的所有人
            group_clients = [i for i, labels in enumerate(label_distribution) if set(group).issubset(set(labels))]
            group_size = len(group_clients)
            if group_size == 0:
                continue
            # 计算这个组合的人数占总人数的比例
            group_proportion = group_size / len(label_distribution)
            # 计算这个组合应该抽取的人数
            group_sample_size = round(group_proportion * len(label_distribution))
            if group_sample_size == 0:
                group_sample_size = 1
            # 进行简单随机抽样
            selected = set(random.sample(group_clients, group_sample_size))
            for client in selected:
                if client not in selected_clients:
                    sampled_clients.append(client)
                    selected_clients.add(client)
        return sampled_clients
