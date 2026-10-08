import json
import os.path

from torch import nn
import torch
from torch import optim
import numpy as np

seed = 100
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def feature_generated(n_distribution, n_dim, max_iters=10000):
    random_vectors = torch.randn(n_distribution, n_dim, device=device)
    param = nn.Parameter(random_vectors)
    temp_matrix = 1 - torch.eye(int(n_distribution), dtype=torch.float, requires_grad=False, device=device)
    optimizer = optim.RMSprop([param], lr=4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    for i in range(max_iters):
        optimizer.zero_grad()
        normed_x = param / param.norm(dim=1).unsqueeze(1)
        cov = torch.mm(normed_x, normed_x.t()) ** 2 / (n_distribution - 1)
        loss = torch.mean(cov * temp_matrix)
        # loss = torch.mean(cov)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        if i % 1000 == 0:
            print("Iter: {}, loss: {}".format(i, loss.item()))

    normed_n_mean = param / param.norm(dim=1).unsqueeze(1)
    return normed_n_mean


def generate_data(means, std, count):
    labels = []
    with torch.no_grad():
        classes = []
        for class_, mean in enumerate(means):
            for i in range(count):
                data = []
                for n in mean:
                    data.append(torch.normal(n, std))
                classes.append(torch.tensor(data))
                labels.append(class_)
    return classes, labels


def one_hot_generate(n_dim=100, classes=10):
    identity_matrix = torch.eye(n_dim)

    # 抽样列向量作为特征锚点，可以选择单位矩阵的随机列或者简单地选择前n列
    # 假设我们需要的特征锚点数量是 n
    n = classes  # 示例，我们需要10个特征锚点

    # 随机抽样列向量作为特征锚点
    indices = torch.randperm(n_dim)[:n]  # 随机排列后选取前n个索引
    feature_anchors = identity_matrix[indices]
    with open("feature_mean_{}_{}_onehot.json".format(n_dim, classes), "w") as f:
        json.dump(feature_anchors.numpy().tolist(), f)


if __name__ == '__main__':
    # one_hot_generate(84, 10)
    device = "cuda"
    n_dim = 64
    class_count = 10

    if not os.path.exists("feature_mean_{}_{}.json".format(n_dim, class_count)):
        mean = feature_generated(class_count, n_dim)
        with open("feature_mean_{}_{}.json".format(n_dim, class_count), "w") as f:
            json.dump(mean.cpu().detach().numpy().tolist(), f)
    else:
        with open("feature_mean_{}_{}.json".format(n_dim, class_count), "r") as f:
            mean = json.load(f)
            mean = torch.tensor(mean)
    each_class_count = 10
    data, labels = generate_data(mean, 0.05, each_class_count)

    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    tsne = TSNE(n_components=2, perplexity=min(30, len(data) - 1))
    X_tsne = tsne.fit_transform(torch.stack(data))

    plt.figure()

    for i in range(class_count):
        plt.scatter(X_tsne[i * each_class_count:i * each_class_count + each_class_count, 0],
                    X_tsne[i * each_class_count:i * each_class_count + each_class_count, 1])
    plt.legend()
    plt.show()
