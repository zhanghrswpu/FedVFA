import json
import os
import random
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator, FuncFormatter
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset
from matplotlib import rcParams

from loss import FedProx, FedDynLoss, FedFA, FedVFA

config = {
    "font.family": 'serif',
    "font.size": 18,
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
}

rcParams.update(config)


def save_log(path, filename=None, dict=None):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + filename, "w") as f:
        json.dump(dict, f)


def save_model(path, filename, model):
    if not os.path.exists(path):
        os.makedirs(path)

    torch.save(model.state_dict(), "./{}/{}.pth".format(path, filename))


def draw_plot(data, labels, title="", xlable="", ylable=""):
    plt.figure(figsize=(5, 5))
    for index, d in enumerate(data):
        plt.plot(d, label=labels[index])
    plt.ylabel(ylable)
    plt.title(title)
    plt.legend()
    plt.show()


def statistic_classes_count(y):
    """
    统计各个值的个数，传入的y必须是经过了one_hot之后的
    :param y:
    :return:
    """
    if len(y) == 0:
        return []
    count = dict()
    for c in range(len(y[0])):
        count[c] = 0
    for i in y:
        # count[np.argmax(i).numpy().tolist()] += 1
        count[np.argmax(i).tolist()] += 1
    for k, v in count.items():
        count[k] = max(1e-8, v)
    return count


def one_hot_recover(labels):
    recover_label = []
    for label in labels:
        recover_label.append(torch.argmax(label))
    return recover_label


def batch_train(model, optimizer, loss_func, data_loader, epochs, device, custom_loss_compute=None,
                return_features=False, label_count=10):
    model.train()

    loss_value = 0
    features = defaultdict(list)
    correct = 0
    total = 0
    # 这个值给fedfa用
    momentum_feature_means = {label: torch.zeros(list(model.parameters())[-2].shape[1]).to(device) for label in
                              range(label_count)}

    # 在这个方法内部，将local_train_batch视为本地训练的epoch的数量，每个epoch用全部的本地数据来进行训练
    for epoch in range(epochs):
        for index, (x, y) in enumerate(data_loader):
            optimizer.zero_grad()
            labels = one_hot_recover(y)
            x, y = x.to(device), y.to(device)
            if len(x) == 1:
                continue
            total += len(x)

            if custom_loss_compute:
                loss, logit = custom_loss_compute(x, y, model, loss_func)
            else:
                feature, logit = model(x)

                if isinstance(loss_func, FedFA):
                    # 复制一下再处理
                    copied_feature = feature.clone().detach()
                    copied_feature.requires_grad = False

                    momentum = 0.5
                    # 这两个变量给FedFA用
                    feature_sums = {}  # 存储特征求和
                    feature_counts = {}  # 存储每个类别的特征数量
                    for i in range(len(labels)):
                        label = labels[i].item()  # 获得当前样本的类别
                        cur_feature = copied_feature[i]
                        # 如果该类别的特征未被计算过，初始化求和和计数
                        if label not in feature_sums:
                            feature_sums[label] = cur_feature
                            feature_counts[label] = 1
                        else:
                            # 将当前样本的特征添加到对应类别的特征求和中，并增加计数
                            feature_sums[label] += cur_feature
                            feature_counts[label] += 1
                    feature_means = {label: feature_sums[label] / feature_counts[label] for label in feature_sums}

                    for label in feature_means:
                        m_mean = momentum_feature_means[label]
                        current_mean = feature_means[label]
                        updated_mean = m_mean * momentum + (1 - momentum) * current_mean
                        momentum_feature_means[label] = updated_mean

                if return_features and epoch == epochs - 1:  # 获得更高的准确率
                    feature = feature.detach()
                    feature.requires_grad = False
                    for i in range(len(feature)):
                        features[int(labels[i])].append(feature[i])

                if isinstance(loss_func, FedProx) or isinstance(loss_func, FedDynLoss):
                    loss = loss_func(logit, y, model)
                elif isinstance(loss_func, FedFA):
                    loss = loss_func(feature, logit, y)
                else:
                    loss = loss_func(logit, y)

            correct += (logit.argmax(1) == y.argmax(1)).type(
                torch.float).sum().float().detach().cpu().numpy().tolist()

            loss.backward()
            loss_value += loss.item()
            optimizer.step()

            if hasattr(loss_func, 'enable_after_adjust') and loss_func.enable_after_adjust:
                optimizer.zero_grad()
                loss = loss_func.classifier_train()
                loss.backward()
                optimizer.step()

    if return_features:
        return loss_value, features, correct / total
    elif isinstance(loss_func, FedFA):
        return loss_value, momentum_feature_means, correct / total
    else:
        return loss_value, correct / total


def moon_batch_train(model, pre_model, global_model, optimizer, loss_func, data_loader, epochs, device):
    model.train()

    loss_value = 0
    correct = 0
    total = 0
    for epoch in range(epochs):
        for index, (x, y) in enumerate(data_loader):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)

            total += len(x)
            feature, logit = model(x)
            pre_feature, pre_logit = pre_model(x)
            global_feature, global_logit = global_model(x)

            correct += (logit.argmax(1) == y.argmax(1)).type(
                torch.float).sum().float().detach().cpu().numpy().tolist()
            loss = loss_func(logit, y, feature, global_feature, pre_feature)

            loss.backward()
            loss_value += loss.item()
            optimizer.step()

    return loss_value, correct / total


def data_split(X, y, test_ratio=0.2):
    """
    将数据集拆分成训练集和测试集
    :param X:   X
    :param y:   y
    :param test_ratio:  测试集的ratio
    :return:
    """
    permutation = np.random.permutation(X.shape[0])  # 利用np.random.permutaion函数，获得打乱后的行数，输出permutation
    X = X[permutation]  # 得到打乱后数据a
    y = y[permutation]

    test_X = X[0:int(len(X) * test_ratio)]
    train_X = X[int(len(X) * test_ratio):]
    test_y = y[0:int(len(X) * test_ratio)]
    train_y = y[int(len(X) * test_ratio):]
    return train_X, train_y, test_X, test_y


def split_noniid(train_data, train_labels, alpha=0.1, n_clients=30, draw_data_distribution=False):
    """

    :param train_data:  训练数据，也就是x
    :param train_labels:    没有one_hot的训练数据的标签，
    :param alpha:   迪利克雷分布的参数
    :param n_clients:   需要将数据划分给多少个client
    :return:
    """
    n_classes = train_labels.max() + 1
    # 每个client每个样本占多少数据的分布,纵轴的长度代表标签的个数,横轴的长度代表client的个数
    # 第i行的第j的位置的数据表示第j个client能得到第i个label的数据的多少
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    # 每个类别的class的下标
    class_idcs = [np.argwhere(train_labels[range(len(train_labels))] == y).flatten()
                  for y in range(n_classes)]

    client_data = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        # np.split 把一个数组从左到右按指定的规则切分
        for i, idcs in enumerate(np.split(c, np.ceil(np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_data[i] += [idcs]

    if draw_data_distribution:
        client_noniid_draw(label_distribution * (len(train_labels) / n_classes))

    one_hot_label = []
    for l in train_labels:
        one_hot_label.append(one_hot(l, n_classes))

    one_hot_label = torch.tensor(np.array(one_hot_label))

    client_data_res = []
    for idcs in client_data:
        idcs = np.concatenate(idcs)
        client_data_res.append(list(zip(train_data[idcs], one_hot_label[idcs])))

    return client_data_res


def split_sequence(train_data, train_labels, count=1, n_clients=30, print_client_data=False):
    """
    :param count: 每个client最多拥有的class数量
    """
    n_classes = train_labels.max() + 1
    class_data = []
    for i in range(n_classes):
        indices = np.where(train_labels[:] == i)[0]
        class_data.append(indices)

    assert n_classes % count == 0
    group_num = n_classes / count
    group_num_client = int(n_clients / group_num)

    assign_data = []
    for class_d in class_data:
        diff = len(class_d) % group_num_client
        if diff != 0:
            diff = group_num_client - diff
            class_d = np.append(class_d, random.sample(class_d.tolist(), int(diff)), axis=0).astype(np.int64)
        assign_data.append(np.split(class_d, group_num_client))

    client_data = []
    for group in range(int(group_num)):
        for id_client in range(group_num_client):
            data = []
            label = []
            for class_ in range(count):
                data.extend(assign_data[count * group + class_][id_client])
                label.extend([count * group + class_] * len(assign_data[count * group + class_][id_client]))
            client_data.append(list(zip(train_data[data], torch.tensor(one_hot(train_labels[data], n_classes)))))

    return client_data


def one_hot(arr, num_classes):
    return np.eye(num_classes)[arr]


def client_noniid_draw(label_distribution):
    plt.figure(dpi=300)
    plt.figure(figsize=(14, 6))
    count = len(label_distribution[0])
    index = np.arange(1, count + 1, 1)

    bottom = [0] * len(label_distribution[0])
    for i, dis in enumerate(label_distribution):
        plt.bar(x=index, height=dis, bottom=bottom, label="label {}".format(i))
        for i in range(len(bottom)):
            bottom[i] += dis[i]
    index = np.arange(0, count + 1, 5)
    index[0] = 1
    plt.yticks(fontsize=20)
    plt.xticks(index, index, fontsize=20)
    plt.xlabel("client ID", fontsize=25)
    plt.ylabel("count", fontsize=25)
    plt.legend(fontsize=15)
    # plt.savefig("./image/" + "client_data" + ".pdf")
    plt.show()


def statistics_classes_count(clients, data_classes_count):
    classes_count_client = []
    for client in clients:
        temp = [0] * data_classes_count
        for k, v in client.classes_count.items():
            temp[k] = v
        classes_count_client.append(temp)
    return classes_count_client


def get_similarity(grad_1, grad_2, distance_type="L1"):
    if distance_type == "L1":

        norm = 0
        for g_1, g_2 in zip(grad_1, grad_2):
            norm += np.sum(np.abs(g_1 - g_2))
        return norm

    elif distance_type == "L2":
        norm = 0
        for g_1, g_2 in zip(grad_1, grad_2):
            norm += np.sum((g_1 - g_2) ** 2)
        return norm

    elif distance_type == "cosine":
        norm, norm_1, norm_2 = 0, 0, 0
        for i in range(len(grad_1)):
            norm += np.sum(grad_1[i] * grad_2[i])
            norm_1 += np.sum(grad_1[i] ** 2)
            norm_2 += np.sum(grad_2[i] ** 2)

        if norm_1 == 0.0 or norm_2 == 0.0:
            return 0.0
        else:
            norm /= np.sqrt(norm_1 * norm_2)

            return np.arccos(norm)


def draw_bar(acc_distribution, class_distribution, label, title):
    for i, d in enumerate(acc_distribution):
        sorted_d = dict(sorted(d.items(), key=lambda x: x[0]))
        plt.plot(list(sorted_d.keys()), list(sorted_d.values()), label=label[i])
    if class_distribution is not None:
        total_count = list(class_distribution.values())
        count_sum = sum(total_count)
        for i in range(len(total_count)):
            total_count[i] /= count_sum
        plt.bar(list(class_distribution.keys()), total_count)
    plt.title(title)
    plt.legend()
    plt.show()


def generate_matrix(d=100, classes=10):
    identity_matrix = torch.eye(d)

    # 抽样列向量作为特征锚点，可以选择单位矩阵的随机列或者简单地选择前n列
    # 假设我们需要的特征锚点数量是 n
    n = classes  # 示例，我们需要10个特征锚点

    # 随机抽样列向量作为特征锚点
    indices = torch.randperm(d)[:n]  # 随机排列后选取前n个索引
    feature_anchors = identity_matrix[indices]

    return feature_anchors


def visualize_high_dimensional_vectors(vectors, labels=None, perplexity=9, n_iter=1000):
    """
    使用 t-SNE 可视化高维向量。

    参数:
        vectors (np.ndarray): 高维向量数组。
        labels (list): 与向量对应的标签列表，用于可视化时的颜色编码。
        perplexity (int): t-SNE 的复杂度参数，默认为30。
        n_iter (int): t-SNE 的优化迭代次数，默认为1000。
    """
    # 初始化 t-SNE 模型
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter)

    # 执行 t-SNE 降维
    vectors_2d = tsne.fit_transform(vectors)

    # 绘制结果
    plt.figure(figsize=(10, 8))

    # 如果有标签，则按标签绘制带颜色的点
    if labels is not None:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            indices = [i for i, l in enumerate(labels) if l == label]
            plt.scatter(vectors_2d[indices, 0], vectors_2d[indices, 1], label=str(label))
        plt.legend()
    # 否则只绘制点
    else:
        plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1])

    plt.title('High Dimensional Vectors Visualization using t-SNE')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()


def load_data():
    file = "./test/feature_mean_128_47_old.json"
    with open(file, "r") as f:
        ff = json.load(f)
    return np.array(ff)


def compute_cka_for_layers(model1, model2, dataloader, device=None):
    def hsic(centered_kernel_x, centered_kernel_y):
        return torch.sum(centered_kernel_x * centered_kernel_y)

    def centering(kernel_matrix):
        n = kernel_matrix.size(0)
        unit = torch.ones(n, n, device=kernel_matrix.device) / n
        centered_kernel = kernel_matrix - unit @ kernel_matrix - kernel_matrix @ unit + unit @ kernel_matrix @ unit
        return centered_kernel

    def compute_cka(kernel_x, kernel_y):
        centered_kernel_x = centering(kernel_x)
        centered_kernel_y = centering(kernel_y)
        hsic_value = hsic(centered_kernel_x, centered_kernel_y)
        norm_x = hsic(centered_kernel_x, centered_kernel_x).sqrt()
        norm_y = hsic(centered_kernel_y, centered_kernel_y).sqrt()
        cka_value = hsic_value / (norm_x * norm_y)
        return cka_value.item()

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set both models to evaluation mode
    model1.to(device).eval()
    model2.to(device).eval()

    activations1 = []
    activations2 = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            inputs1 = inputs
            inputs2 = inputs
            # Pass inputs through each model layer by layer
            a1 = []
            a2 = []

            for layer1, layer2 in zip(model1.layers, model2.layers):
                inputs1, inputs2 = layer1(inputs1), layer2(inputs2)
                a1.append(inputs1)
                a2.append(inputs2)

            # Convert activations to appropriate format for CKA computataion
            a1 = [x.view(x.size(0), -1) for x in a1]  # Flatten and keep as tensor
            a2 = [x.view(x.size(0), -1) for x in a2]
            activations1.append(a1)
            activations2.append(a2)

    # Now compute the CKA for each layer
    cka_values = []
    for i in range(len(model1.layers)):
        features_1 = torch.cat([x[i] for x in activations1], dim=0)
        features_2 = torch.cat([x[i] for x in activations2], dim=0)

        kernel_x = torch.mm(features_1, features_1.t())
        kernel_y = torch.mm(features_2, features_2.t())

        cka_values.append(compute_cka(kernel_x, kernel_y))

    return cka_values


def calculate_model_difference(model1, model2):
    # 确保两个模型在相同的设备上
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model1.to(device)
    model2.to(device)

    # 初始化参数差值的列表
    params_diff = []

    # 遍历模型1的参数
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if name1 != name2:
            raise ValueError("模型结构不匹配: {} 和 {}".format(name1, name2))

        # 计算差值并保存
        param_diff = param1 - param2
        params_diff.append(param_diff.view(-1))

    # 将所有差值连接成一个一维向量
    difference_vector = torch.cat(params_diff)
    return difference_vector


def plot_accuracy_comparison_with_distribution(current_classifier_acc, previous_classifier_acc, class_total,
                                               filename=None):
    fig, ax1 = plt.subplots(figsize=(7, 5))
    font = {'family': 'Times New Roman', 'size': 18}

    # Set up the bar chart for class distribution on ax1 (the primary axis)
    classes = list(class_total.keys())
    counts = [class_total[class_id] for class_id in classes]
    # ax1.bar(classes, counts, color='black', alpha=0.5, label='Number of samples')
    ax1.bar(classes, counts, color='black', alpha=0.5, label='样本数量')

    # Set up the ax2 (the secondary axis) for plotting accuracies
    ax2 = ax1.twinx()
    current_acc_values = [current_classifier_acc.get(class_id, 0) for class_id in classes]
    previous_acc_values = [previous_classifier_acc.get(class_id, 0) for class_id in classes]

    ax2.plot(classes, current_acc_values, label='训练后', marker='o', color='b')
    ax2.plot(classes, previous_acc_values, label='训练前', marker='x', color='r')
    # ax2.plot(classes, current_acc_values, label='After local training', marker='o', color='b')
    # ax2.plot(classes, previous_acc_values, label='Before local training', marker='x', color='r')

    # Set labels, titles, and grid
    ax1.set_xlabel('类别')
    ax1.set_ylabel('样本数量', color='black')
    ax2.set_ylabel('准确率', color='blue')
    # ax1.set_xlabel('Class')
    # ax1.set_ylabel('Number of samples', color='black')
    # ax2.set_ylabel('Accuracy', color='blue')
    ax1.tick_params(axis='y', labelcolor='black')
    ax2.tick_params(axis='y', labelcolor='blue')

    # ax1.set_title('Comparison of Classifier Accuracies and Class Distribution by Class')

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    # ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right',prop=font)
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right')

    ax = plt.gca()

    # 设置x轴和y轴的刻度字体
    for label in ax.get_xticklabels():
        label.set_fontproperties(font)

    for label in ax.get_yticklabels():
        label.set_fontproperties(font)

    fig.tight_layout()  # For better spacing
    plt.grid(True)
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


def plot_accuracy_comparison_with_distribution_2():
    fig, axs = plt.subplots(2, 3, figsize=(20, 9), gridspec_kw={'width_ratios': [1, 1, 1], 'height_ratios': [1, 1]})
    gs = GridSpec(2, 6)  # 创立2 * 6 网格
    gs.update(wspace=1, hspace=0.3)
    # 对第一行进行绘制
    ax1 = plt.subplot(gs[0, :2])  # gs(哪一行，绘制网格列的范围)
    ax2 = plt.subplot(gs[0, 2:4])
    ax3 = plt.subplot(gs[0, 4:6])
    # 对第二行进行绘制
    ax4 = plt.subplot(gs[1, 1:3])
    ax5 = plt.subplot(gs[1, 3:5])
    axs = [ax1, ax2, ax3, ax4, ax5]
    for i in range(5):
        with open("client_{}.json".format(i), "r") as f:
            data = json.load(f)
            current_classifier_acc = data["current"]
            previous_classifier_acc = data["pre"]
            class_total = data["class_count"]
            font = {'family': 'Times New Roman', 'size': 22}
            song_font = {'family': 'SimSun', 'size': 22}

            # Set up the bar chart for class distribution on ax1 (the primary axis)
            classes = list(class_total.keys())
            counts = [class_total[class_id] for class_id in classes]
            axs[i].bar(classes, counts, color='black', alpha=0.5, label='Number of samples')
            # axs[i].bar(classes, counts, color='black', alpha=0.5, label='样本数量')

            # Set up the ax2 (the secondary axis) for plotting accuracies
            ax2 = axs[i].twinx()
            current_acc_values = [current_classifier_acc.get(class_id, 0) for class_id in classes]
            previous_acc_values = [previous_classifier_acc.get(class_id, 0) for class_id in classes]

            # ax2.plot(classes, current_acc_values, label='训练后', marker='o', color='b')
            # ax2.plot(classes, previous_acc_values, label='训练前', marker='x', color='r')
            ax2.plot(classes, current_acc_values, label='After local training', marker='o', color='b')
            ax2.plot(classes, previous_acc_values, label='Before local training', marker='x', color='r')

            # axs[i].set_xlabel('类别', font=song_font)
            axs[i].set_xlabel('Class', font=font)
            # if i == 0 or i == 3:
            #     axs[i].set_ylabel('样本数量', color='black', font=song_font)
            # if i == 2 or i == 4:
            #     ax2.set_ylabel('准确率', color='blue', font=song_font)
            if i == 0 or i == 3:
                axs[i].set_ylabel('Number of samples', color='black', font=font)
            if i == 2 or i == 4:
                ax2.set_ylabel('Accuracy', color='blue', font=font)
            axs[i].tick_params(axis='y', labelcolor='black')
            ax2.tick_params(axis='y', labelcolor='blue')

            # 设置x轴和y轴的刻度字体
            for label in axs[i].get_xticklabels():
                label.set_fontproperties(font)

            for label in axs[i].get_yticklabels():
                label.set_fontproperties(font)
            for label in ax2.get_xticklabels():
                label.set_fontproperties(font)

            for label in ax2.get_yticklabels():
                label.set_fontproperties(font)
            lines_1, labels_1 = axs[i].get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            # ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right',prop=font)
            plt.grid(True)
    lenged_font = {'family': 'Times New Roman', 'size': 22}
    # lenged_font = {'family': 'SimSun', 'size': 22}
    plt.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', columnspacing=0.8, handletextpad=0.2,
               labelspacing=0.5, ncol=3, prop=lenged_font, bbox_to_anchor=(-0.22, 2.6))
    # plt.tight_layout()
    # plt.show()
    plt.savefig("pre_current_compare.pdf")


def check_accuracy_condition(cur_accuracy, pre_accuracy):
    cur_better = any(cur_accuracy[cls] > pre_accuracy[cls] for cls in cur_accuracy)
    pre_better = any(pre_accuracy[cls] > cur_accuracy[cls] for cls in pre_accuracy)

    return cur_better and pre_better


import torch


def krum(gradients, nb_to_keep=1):
    """
    实现Krum算法选择模型更新。

    :param gradients: 参与者的模型参数列表或梯度列表。
    :param nb_to_keep: 最终需要保留的模型的数量，默认为1。
    :return: 被选中的模型更新。
    """
    n = len(gradients)
    distances = []

    # 计算每个模型与其他所有模型之间的距离
    for i in range(n):
        distances.append([])
        for j in range(n):
            if i != j:
                distance = (gradients[i] - gradients[j]).norm()  # 计算L2范数
                distances[i].append(distance)
            else:
                distances[i].append(torch.tensor(float('inf')))  # 自己与自己的距离设置为无穷大

    # 计算距离得分
    scores = []
    for i in range(n):
        # 对每个模型的距离进行排序
        distances[i].sort()
        # 取出前(n-f-2)个最小距离的和，f为可以容忍的错误节点数
        score = torch.sum(torch.stack(distances[i][:-nb_to_keep - 2]))
        scores.append(score)

    # 选择得分最小的模型
    selected_indices = torch.argsort(torch.stack(scores))[:nb_to_keep]

    # 当nb_to_keep为1时，仅返回一个模型梯度，否则返回列表
    if nb_to_keep == 1:
        return gradients[selected_indices[0]]
    else:
        return [gradients[idx] for idx in selected_indices]


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# 创建一个函数来绘制热力图
def plot_heatmap(matrix, x_axis_title, y_axis_title):
    """
    绘制给定矩阵的热力图。

    参数:
    matrix (np.array): 一个二维的numpy数组。
    x_axis_title (str): x轴的标题。
    y_axis_title (str): y轴的标题。
    """
    sns.heatmap(matrix, annot=True, cmap='coolwarm')
    plt.xlabel(x_axis_title)
    plt.ylabel(y_axis_title)
    plt.colorbar()
    plt.show()


def extract_random_sampler(dataloader, num_samples_per_class):
    """
    从给定的 DataLoader 中随机抽取指定数量的数据，确保每个类别都抽取相同数量的数据。

    参数:
    dataloader (DataLoader): 要从中抽取数据的 DataLoader 对象。
    num_samples_per_class (int): 每个类别要抽取的样本数量。

    返回:
    DataLoader: 包含抽取数据的新 DataLoader 对象。
    """
    # 建立一个类别到样本索引列表的映射
    class_to_indices = defaultdict(list)
    for data, class_idx in dataloader.dataset:
        label = torch.argmax(class_idx).item()
        class_to_indices[label].append(data)

    # 对每个类别随机抽取指定数量的样本索引
    selected = []
    selected_label = []
    for class_idx, data in class_to_indices.items():
        if len(data) < num_samples_per_class:
            raise ValueError(f"Class {class_idx} does not have enough samples to draw {num_samples_per_class} samples.")
        selected.extend(data[0:num_samples_per_class])
        selected_label.extend([class_idx] * num_samples_per_class)
    # 根据选中的索引创建一个子数据集

    # 创建一个新的 DataLoader 对象以包含抽取的样本
    balanced_loader = DataLoader(dataset=list(zip(selected, selected_label)), batch_size=dataloader.batch_size,
                                 shuffle=True)

    return balanced_loader


def create_dataloader(x, y, batch_size=32, shuffle=True):
    """
    根据给定的特征张量x和标签张量y创建一个DataLoader。

    参数:
    x (Tensor): 特征张量。
    y (Tensor): 标签张量。
    batch_size (int, optional): 每个批次的大小。默认为32。
    shuffle (bool, optional): 是否在每个纪元之前打乱数据。默认为True。

    返回:
    DataLoader: 用于迭代的数据加载器。
    """

    # 确保x和y的第一维（样本数量）是相同的
    assert x.shape[0] == y.shape[0], "The number of samples in x and y must be equal."
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader



if __name__ == '__main__':
    # aaa()
    # matrix = generate_matrix(d=128)
    # matrix = load_data()
    # visualize_high_dimensional_vectors(matrix)
    plot_accuracy_comparison_with_distribution_2()
    # compare()
