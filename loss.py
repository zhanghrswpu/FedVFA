import datetime
import json
import os.path

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def generated_feature_data(feature_statistics, count, std_exist, p=None):
    features = []
    labels = []
    if p is not None:
        class_data = np.random.choice(
            list(range(len(feature_statistics))), count, replace=True, p=p)
    else:
        class_data = np.repeat(
            list(range(len(feature_statistics))), count / len(feature_statistics))

    class_data = np.sort(class_data)
    for class_ in class_data:
        v = feature_statistics[class_]
        feature = []
        for j in range(len(v[0])):
            if std_exist:
                feature.append(torch.normal(v[0][j], v[1][j]))
            else:
                feature.append(v[0][j])
        features.append(feature)
        labels.append(class_)

    return torch.tensor(features), torch.tensor(labels)


class FedFA(nn.Module):
    def __init__(self, model, anchor_features, mu=0.1, enable_after_adjust=True):
        super(FedFA, self).__init__()
        self.classes_count = list(model.parameters())[-1].size()[0]
        self.anchor_features = anchor_features
        device = next(model.parameters()).device
        self.anchor_features.requires_grad = False
        self.anchor_labels = torch.tensor([i for i in range(len(anchor_features))],
                                          requires_grad=False).to(device)
        self.model = model
        self.mu = mu
        self.enable_after_adjust = enable_after_adjust

    def forward(self, features, logits, labels):
        centers_batch = self.anchor_features[torch.argmax(labels, dim=1)]
        dist_vec = torch.pow(features - centers_batch, 2).sum(dim=1)
        center_loss = dist_vec.sum() / len(features) * self.mu

        return F.cross_entropy(logits,
                               labels) + center_loss

    def classifier_train(self):
        feature, logit = self.model(self.anchor_features, -1)
        return F.cross_entropy(logit,
                               self.anchor_labels)


class FedLoss(nn.Module):

    def __init__(self, model, beta, epoch, features, labels):
        super(FedLoss, self).__init__()
        self.model = model
        self.beta = beta
        self.epoch = epoch
        self.features = features
        self.labels = labels

    def forward(self, inputs, targets, test=False):
        if self.epoch == 0 or test:
            return F.cross_entropy(inputs, targets)
        features, labels = self.features, self.labels
        _, feature_targets = self.model(features, start_layer_idx=-1)
        return F.cross_entropy(inputs, targets) + self.beta * F.cross_entropy(feature_targets, labels)


class FedLC(nn.Module):
    # Federated Learning with Label Distribution Skew via Logits Calibration
    def __init__(self, tau, count_classes):
        super(FedLC, self).__init__()
        self.tau = tau
        self.count_classes = count_classes
        for i in range(len(self.count_classes)):
            self.count_classes[i] = max(1e-8, self.count_classes[i])

    def forward(self, logits, targets):
        if not isinstance(self.count_classes, torch.Tensor):
            self.count_classes = torch.tensor(
                self.count_classes).to(logits.device)
            self.count_classes.requires_grad = False
        #
        # targets = torch.argmax(targets, dim=1)
        # cal_logit = torch.exp(logits - (
        #     self.tau
        #     * torch.pow(self.count_classes, -1 / 4)
        #     .unsqueeze(0)
        #     .expand((logits.shape[0], -1))
        # ))
        # y_logit = torch.gather(cal_logit, dim=-1, index=targets.unsqueeze(1))
        # loss = -torch.log(y_logit / cal_logit.sum(dim=-1, keepdim=True))
        # return loss.sum() / logits.shape[0]
        return F.cross_entropy(logits - (
                self.tau
                * torch.pow(self.count_classes, -1 / 4)
                .unsqueeze(0)
                .expand((logits.shape[0], -1))
        ), targets)


class FedProx(nn.Module):
    # 作者在不同的方法中使用不同的参数：0.001, 0.01, 0.1, 1
    def __init__(self, last_weight, mu=1e-2):
        super(FedProx, self).__init__()
        self.last_weight = last_weight
        self.mu = mu

    def forward(self, logits, labels, model):
        loss = torch.empty((1,), device=logits.device)
        for w, w_last in zip(model.parameters(), self.last_weight.parameters()):
            loss += torch.pow(torch.norm(w - w_last), 2)
        return F.cross_entropy(logits, labels) + self.mu / 2 * loss


class MoonLoss(nn.Module):

    def __init__(self, tau=0.5, mu=1):
        super(MoonLoss, self).__init__()
        self.tau = tau
        self.mu = mu

    def forward(self, logits, labels, feature, global_feature, pre_feature):
        loss_con = -torch.log(
            torch.exp(F.cosine_similarity(feature, global_feature) / self.tau)
            / (
                    torch.exp(F.cosine_similarity(pre_feature, feature) / self.tau)
                    + torch.exp(F.cosine_similarity(feature,
                                                    global_feature) / self.tau)
            )
        )
        return F.cross_entropy(logits, labels) + self.mu * torch.mean(loss_con)


class FedDynLoss(nn.Module):

    def __init__(self, global_model, nable, alpha=0.01, ):
        #  [.1, .01, .001];
        super(FedDynLoss, self).__init__()
        self.vectorized_curr_params = None
        self.nabla = nable
        self.alpha = alpha
        self.vectorized_global_params = self.vectorize(
            global_model).to(self.nabla.device)

    def forward(self, logits, labels, model):
        # The formula is from the paper
        loss = F.cross_entropy(logits, labels)
        # 模型参数向量化
        self.vectorized_curr_params = self.vectorize(model)
        loss -= torch.dot(self.nabla, self.vectorized_curr_params)
        loss += (self.alpha / 2) * torch.norm(
            self.vectorized_curr_params - self.vectorized_global_params
        )
        return loss

    def vectorize(self, src):
        return torch.cat([param.flatten() for param in src.parameters()])


class FedRs(nn.Module):

    def __init__(self, classes_count, alpha=0.5):
        super(FedRs, self).__init__()
        self.vectorized_curr_params = None
        self.alpha = alpha
        self.classes_count = classes_count

    def forward(self, logits, labels):
        m_logits = torch.ones_like(logits[0]).to(logits.device) * self.alpha
        for k, v in self.classes_count.items():
            if v > 1e-8:
                m_logits[k] = 1.0

        for i in range(len(logits)):
            logits[i] = torch.mul(logits[i], m_logits)

        return F.cross_entropy(logits, labels)


class FedSubLoss(nn.Module):

    def __init__(self, model, alpha, beta, anchor_count, variance, means, epoch):
        super(FedSubLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.anchor_count = anchor_count
        self.classes_count = list(model.parameters())[-1].size()[0]
        device = next(model.parameters()).device
        self.mean_anchor = means
        if self.anchor_count != 0:
            self.anchor_features, self.anchor_labels = self.generate_data(
                means, variance, self.anchor_count, device, epoch)
            self.anchor_features.requires_grad = False
        self.model = model

    def generate_data(self, means, std, count, device):
        """
        :param means:   各个类的均值
        :param std:     方差
        :param count:   各个类的样本个数
        :return:
        """
        labels = []
        with torch.no_grad():
            classes = []
            for class_, mean in enumerate(means):
                for i in range(count):
                    classes.append(torch.normal(mean, std))
                    labels.append(class_)
        return torch.stack(classes).to(device), torch.tensor(labels,
                                                             requires_grad=False).to(device)

    def forward(self, features, logits, labels):
        centers_batch = self.mean_anchor[torch.argmax(labels, dim=1)]
        dist_vec = torch.pow(features - centers_batch, 2).sum(dim=1)
        center_loss = dist_vec.sum() / len(features) * self.beta
        if self.anchor_count != 0:
            _, feature_targets = self.model(
                self.anchor_features, start_layer_idx=-1)

            classifier_loss = self.alpha * F.cross_entropy(feature_targets, self.anchor_labels)
        else:
            classifier_loss = 0

        return F.cross_entropy(logits,
                               labels) + center_loss + classifier_loss * self.alpha


class FedVFA(nn.Module):

    def __init__(self, model, alpha, beta, means, enable_after_adjust=True):
        super(FedVFA, self).__init__()
        self.classes_count = list(model.parameters())[-1].size()[0]
        self.mean_anchor = means
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.anchor_labels = torch.tensor(list(range(len(self.mean_anchor)))).to("cuda")
        self.enable_after_adjust = enable_after_adjust

    def forward(self, features, logits, labels):
        centers_batch = self.mean_anchor[torch.argmax(labels, dim=1)]
        dist_vec = torch.pow(features - centers_batch, 2).sum(dim=1)
        center_loss = dist_vec.sum() / len(features) * self.alpha
        with torch.no_grad():
            self.model.eval()
            feature, teachear_logits = self.model(self.mean_anchor, -1)
            self.model.train()
        return self.logist_(teachear_logits, logits, labels) + center_loss

    def logist_(self, teacher_logits, student_logits, labels):
        _teacher_logits = []
        for label in labels:
            _teacher_logits.append(teacher_logits[torch.argmax(label)])
        loss_func = DistillationLoss(alpha=self.beta)
        return loss_func(torch.stack(_teacher_logits), student_logits, labels)

    def classifier_train(self):
        feature, logit = self.model(self.mean_anchor, -1)
        return F.cross_entropy(logit, self.anchor_labels)


class DistillationLoss(nn.Module):
    def __init__(self, alpha=1):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        # batchmean
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, teacher_logits, student_logits, labels):
        """
        :param teacher_logits: 预测输出 from the teacher model
        :param student_logits: 预测输出 from the student model
        :param labels: 真实 labels of the data
        :return: combined distillation and cross-entropy loss
        """
        # Calculate the soft targets with temperature
        teacher_soft = F.softmax(teacher_logits, dim=1)
        student_soft = F.log_softmax(student_logits, dim=1)

        # Distillation loss，self.kl_div
        distillation_loss = F.kl_div(
            student_soft,
            teacher_soft
        )

        # Cross-entropy loss
        ce_loss = F.cross_entropy(student_logits, labels)

        # Final combined loss
        loss = ce_loss + distillation_loss * self.alpha

        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Compute the cross entropy loss (which includes log_softmax)
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Compute the model's probability of correct classification
        pt = torch.exp(-BCE_loss)

        # Compute the focal loss component
        alpha_factor = torch.ones(targets.size(0), device=targets.device) * self.alpha
        alpha_factor = torch.where(torch.eq(targets, 1), alpha_factor, 1. - alpha_factor)
        focal_loss = alpha_factor * (1 - pt) ** self.gamma * BCE_loss

        # Reduce the loss (mean or sum)
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x1, x2, label):
        # 计算x1和x2之间的欧几里得距离
        euclidean_distance = F.pairwise_distance(x1, x2)
        # 计算对比损失
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
