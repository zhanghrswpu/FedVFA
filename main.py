import argparse
import copy
import json
import os
import random
from collections import defaultdict

from sklearn.manifold import TSNE
import numpy as np
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

import loss as ll
import utils
from client import FedVFAClient
from loss import FedLoss, FedProx, FedLC
from param import get_args
from prepare import *


def set_random_seed(seed=100, deterministic=True, benchmark=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
    if benchmark:
        torch.backends.cudnn.benchmark = True


def statistical_distribution(args, selected_client):
    args.statistical_client = []
    if not hasattr(args, "statistical_client"):
        args.statistical_client = []
    if not hasattr(args, "data_distribution"):
        args.data_distribution = defaultdict(int)
    for client in selected_client:
        if not client.id in args.statistical_client:
            args.statistical_client.append(client.id)
            for k, v in client.classes_count.items():
                args.data_distribution[k] += v


def average_feature_distribution(args, selected_clients, std_required):
    # 统计被选择的客户端的各个类别的数量之和
    # 客户端的各个类别的特征所占的比例
    factors = []
    each_class_count = [0] * len(selected_clients[0].classes_count.keys())
    for class_ in selected_clients[0].classes_count.keys():
        count = 0
        for client in selected_clients:
            count += client.classes_count[class_]
        each_class_count[class_] = count
        if count == 0:
            factor = [0] * len(selected_clients)
        else:
            factor = [
                client.classes_count[class_] / count for client in selected_clients
            ]
        factors.append(factor)

    feature_statistics = []
    for class_ in range(args.num_classes):
        mean_feature = (
            selected_clients[0].features_statistics[class_][0] * factors[class_][0]
        )

        if std_required:
            std_feature = (
                selected_clients[0].features_statistics[class_][1] * factors[class_][0]
            )
        for i, client in enumerate(selected_clients[1:]):
            mean_feature += (
                client.features_statistics[class_][0] * factors[class_][i + 1]
            )
            if std_required:
                std_feature += (
                    client.features_statistics[class_][1] * factors[class_][i + 1]
                )
        if std_required:
            feature_statistics.append([mean_feature, std_feature])
        else:
            feature_statistics.append([mean_feature])

    return feature_statistics


def classifier_train(model, feature_statistics, args):
    for param in list(model.parameters())[:-2]:
        param.requires_grad = False
    optimizer = args.optimizer(model.parameters(), **args.optimizer_parameters)

    for epoch in range(2):
        features, labels = loss.generated_feature_data(feature_statistics, 10)
        labels = torch.zeros((len(labels), args.num_classes)).scatter_(
            1, labels.long().reshape(-1, 1), 1
        )
        optimizer.zero_grad()
        _, feature_targets = model(features, start_layer_idx=-1)
        loss_func = F.cross_entropy(feature_targets, labels)
        print("classifier loss:", loss_func.item())
        loss_func.backward()
        optimizer.step()

    for param in list(model.parameters())[:-2]:
        param.requires_grad = True


def generated_feature_data(feature_statistics, count, p=None):
    features = []
    labels = []
    if p is not None:
        class_data = np.random.choice(
            list(range(len(feature_statistics))), count, replace=True, p=p
        )
    else:
        class_data = np.repeat(
            np.random.choice(
                list(range(len(feature_statistics))),
                len(feature_statistics),
                replace=False,
            ),
            count / len(feature_statistics),
        )

    class_data = np.sort(class_data)
    for class_ in class_data:
        v = feature_statistics[class_]
        feature = []
        for j in range(len(v[0])):
            feature.append(torch.normal(v[0][j], v[1][j]))
        features.append(feature)
        labels.append(class_)

    return torch.tensor(features), torch.tensor(labels)


def evaluate_accuracy_per_class(model, dataloader, device="cpu"):
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            features, outputs = model(inputs)

            for label, prediction in zip(labels, outputs):
                label = torch.argmax(label, 0)
                prediction = torch.argmax(prediction, 0)
                if label == prediction:
                    class_correct[label.item()] += 1
                class_total[label.item()] += 1

    # Calculate accuracy for each class
    class_accuracy = {
        class_id: class_correct[class_id] / class_total[class_id]
        for class_id in class_total
    }

    return class_accuracy


def compare_classifier(model, cur_classifier_param, dataloader, device="cpu"):
    model.eval()
    model.to(device)
    pre_classifier_param = copy.deepcopy(model.layers[-1].state_dict())
    accuracy_pre = evaluate_accuracy_per_class(model, dataloader, device)
    model.layers[-1].load_state_dict(cur_classifier_param)
    accuracy_cur = evaluate_accuracy_per_class(model, dataloader, device)
    model.layers[-1].load_state_dict(pre_classifier_param)
    model.to("cpu")
    return accuracy_cur, accuracy_pre


def compute_cka(clients_models, test_set):
    cka_matrix = [[0] * len(clients_models) for i in range(len(clients_models))]
    for i in range(len(clients_models)):
        for j in range(i + 1, len(clients_models)):
            cka_matrix[i][j] = utils.compute_cka_for_layers(
                clients_models[i], clients_models[j], test_set
            )
            cka_matrix[j][i] = cka_matrix[i][j]
    titles = ["client_{}".format(i) for i in range(len(clients_models))]
    utils.plot_heatmap(cka_matrix, titles, titles)
    return cka_matrix


def train(aggregate, clients, test_set, model, sampler, args):
    total_loss, total_acc, last_acc, last_loss = [], [], [], []

    # 主模型并不参与训练
    model.eval()
    for epoch in range(args.epochs):
        print(("-" * 10 + "  epoch {}  " + "-" * 10).format(epoch))
        loss = 0
        acc = 0
        # 随机采样一些client进行训练
        selected_clients = sampler.run()
        for index, client in enumerate(selected_clients):
            correct, client_loss = client.train(model, epoch)
            if args.pre_classifier_compare and epoch == 5:
                cur_classifier_param = client.model.layers[-1].state_dict()
                accuracy_current, accuracy_previous = compare_classifier(
                    model, cur_classifier_param, test_set, args.device
                )
                if utils.check_accuracy_condition(accuracy_current, accuracy_previous):
                    acc, class_acc_dict = clients[0].test(model, test_set)
                    print(acc)
                    with open("client_{}.json".format(index), "w") as f:
                        info = {
                            "current": accuracy_current,
                            "pre": accuracy_previous,
                            "class_count": client.classes_count,
                        }
                        json.dump(info, f)
                    utils.plot_accuracy_comparison_with_distribution(
                        accuracy_current,
                        accuracy_previous,
                        client.classes_count,
                        "client_{}.pdf".format(index),
                    )

            client.model.to("cpu")
            del client.model
            acc += correct
            loss += client_loss

        # compute_cka(clients_model, test_set)

        if args.log_client_loss:
            print(
                "client train total loss: {:.3f}, train acc: {:.3f}\n".format(
                    loss, acc / len(selected_clients)
                )
            )
            total_loss.append(loss)

        aggregate.run(model, selected_clients, epoch)

        for client in selected_clients:
            client.clients_weight_update = None

        if args.epochs - epoch <= args.last_acc_count:
            acc, class_acc_dict = clients[0].test(model, test_set)
            last_acc.append(acc)
            if epoch % args.print_acc_interval == args.print_acc_interval - 1:
                print("test average accuracy of the client: {}".format(acc))
                total_acc.append(acc)
            continue

        if epoch % args.print_acc_interval == args.print_acc_interval - 1:
            acc, class_acc_dict = clients[0].test(model, test_set)
            print("test average accuracy of the client: {}".format(acc))
            total_acc.append(acc)
            # res = {"acc": total_acc, "last": last_acc}
            # utils.save_log("log/{}/".format(args.dataset), "{}_{}.json".format(args.algorithm, args.loss), res)
    return total_loss, total_acc, last_acc, last_loss, model


def main():
    set_random_seed()
    args = get_args()
    args.choice_count = min(args.choice_count, args.client_count)
    print(args)
    config = copy.deepcopy(vars(args))
    loss_name = args.loss
    prepare_loss(args)
    prepare_optimizer(args)
    X, y, test_loader = create_dataset(args)
    # extract_X, extract_y = utils.extract_data(test_loader, 1)
    prepare_testloader(args, test_loader)
    model = create_model(args)
    clients = create_client(X, y, args)
    algorithm = create_algorithm(args, clients)
    sampler = create_sampler(args, clients, X)
    total_loss, total_acc, last_acc, last_loss, model = train(
        algorithm, clients, test_loader, model, sampler, args
    )
    res = {"acc": total_acc, "last": last_acc, "config": config}
    if args.log_file_name is not None:
        log_file_name = "{}.json".format(args.log_file_name)
    else:
        log_file_name = "{}_{}.json".format(args.algorithm, loss_name)
    if args.enable_dirichlet:
        utils.save_log(
            "log/{}/{}/".format(args.dataset, args.dirichlet_alpha), log_file_name, res
        )
    else:
        utils.save_log(
            "log/{}/{}/".format(args.dataset, args.each_class_count), log_file_name, res
        )
    utils.save_model(
        "log/trained_model",
        "{}_{}_{}_{}_{}".format(
            args.algorithm, loss_name, args.model, args.dataset, args.dirichlet_alpha
        ),
        model,
    )


def parameter_adjust():
    set_random_seed()
    betas = [0.05]
    alphas = [2, 3, 4, 5]

    args = get_args()
    args.dataset = "emnist"
    args.model = "Net"
    args.epochs = 500

    args.loss = "FedSubLoss"
    dir = "./log/svhn/FedAvg_FedVFA_adjust.json"
    if not os.path.exists(dir):
        with open(dir, "w+", encoding="utf-8") as f:
            json.dump([], f)

    config = copy.deepcopy(vars(args))

    prepare_loss(args)
    prepare_optimizer(args)

    X, y, test_loader = create_dataset(args)
    prepare_testloader(args, test_loader)

    with open(dir, "r") as f:
        res = json.load(f)

    for beta in betas:
        for alpha in alphas:
            set_random_seed()
            config["beta"] = beta
            config["alpha"] = alpha
            args.beta = beta
            args.alpha = alpha
            model = create_model(args)
            clients = create_client(X, y, args)
            algorithm = create_algorithm(args, clients)
            sampler = create_sampler(args, clients, X)
            print(config)
            total_loss, total_acc, last_acc, last_loss, model = train(
                algorithm, clients, test_loader, model, sampler, args
            )
            res.append({"acc": total_acc, "last": last_acc, "config": config})
            with open(dir, "w", encoding="utf-8") as f:
                json.dump(res, f)


if __name__ == "__main__":
    main()
