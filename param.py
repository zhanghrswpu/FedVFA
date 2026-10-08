import argparse


def str2bool(str):
    return True if str.lower() == 'true' else False


def get_fedsubloss_param(parser):
    """
    通过调参来看，对比损失的占比大一些有利于最终准确率的提高，也有利于速度的提升
    例如alpha等于10的最终准确率比5高

    同时，分类器校准的占比高，准确率也会提高
    例如，在alpha等于5的时候，beta等于2，比等于0.5高
    :param parser:
    :return:
    """
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--variance", type=float, default=0.05)
    parser.add_argument("--anchor_count", type=int, default=6)


def get_fedvfaloss_param(parser):
    """
    通过调参来看，对比损失的占比大一些有利于最终准确率的提高，也有利于速度的提升
    例如alpha等于10的最终准确率比5高

    同时，分类器校准的占比高，准确率也会提高
    例如，在alpha等于5的时候，beta等于2，比等于0.5高
    :param parser:
    :return:
    """
    parser.add_argument("--alpha", type=float, default=0)
    parser.add_argument("--beta", type=float, default=0)
    parser.add_argument("--classifier_iter", type=int, default=0)
    parser.add_argument("--enable_after_adjust", type=str2bool, default=False)


def get_fedlc_param(parser):
    parser.add_argument("--fedlc_tau", type=float,
                        default=1, help="FedLC loss parameters")


def get_fedprox_param(parser):
    parser.add_argument("--fedprox_mu", type=float,
                        default=1, help="FedProx loss parameters")


def get_fedmoon_param(parser):
    parser.add_argument("--fedmoon_mu", type=float, default=1)
    parser.add_argument("--fedmoon_tau", type=float, default=0.5)


def get_fedrs_param(parser):
    # in paper 0.5
    parser.add_argument("--fedrs_alpha", type=float, default=0.5)


def get_feddyn_param(parser):
    #  in paper：[.1, .01, .001];
    parser.add_argument("--feddyn_alpha", type=float, default=0.01)


def get_args():
    """
    emnst 2000 epoch 0.005没有1000 0.01效果好
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--model", type=str, default="LeNet")
    parser.add_argument("--log_client_loss", type=bool, default=True)
    parser.add_argument("--print_acc_interval", type=int, default=10)
    parser.add_argument("--last_acc_count", type=int, default=20,
                        help="continuously record the last number of epochs")
    parser.add_argument("--enable_dirichlet", type=str2bool, default=True)
    parser.add_argument("--dirichlet_alpha", type=float, default=0.1)
    parser.add_argument("--each_class_count", type=int, default=2,
                        help="number of classes each client has when non-dirichlet distribution")
    parser.add_argument("--loss", type=str, default="CE")
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--algorithm", type=str, default="FedAvg")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float,
                        default=0.01, help="Local learning rate")
    parser.add_argument("--weight_decay", type=float, default=0, help="")
    parser.add_argument("--momentum", type=float,
                        default=0.9, help="sgd momentum")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--client_count", type=int,
                        default=5, help="total client")
    parser.add_argument("--sampler", type=str,
                        default="random", help="client sampler")
    parser.add_argument("--choice_count", type=int, default=5,
                        help="number of clients selected per iteration")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cpu", "cuda"], help="run device (cpu | cuda)")
    # exp
    parser.add_argument("--draw_data_distribution", type=bool, default=False)
    parser.add_argument("--print_before_after", type=bool, default=False)
    parser.add_argument("--enable_classifier_exp", type=bool, default=False)
    parser.add_argument("--log_file_name", default=None)
    parser.add_argument("--pre_classifier_compare", type=bool, default=True)

    loss = parser.parse_known_args()[0].loss
    if loss == "FedProx":
        get_fedprox_param(parser)
    elif loss == "FedLC":
        get_fedlc_param(parser)
    elif loss == "FedSubLoss":
        get_fedsubloss_param(parser)
    elif loss == "FedRs":
        get_fedrs_param(parser)
    elif loss == "FedDyn":
        get_feddyn_param(parser)
    elif loss == "MoonLoss":
        get_fedmoon_param(parser)
    elif loss == "FedVFA":
        get_fedvfaloss_param(parser)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print(args)
