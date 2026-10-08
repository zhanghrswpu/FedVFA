import copy
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import torch.nn.functional as F

import utils


class FedAvg:

    def __init__(self, clients):
        self.client_count = len(clients)

    def run(self, server_model, clients, epoch):
        clients_weight = []
        for client in clients:
            clients_weight.append(client.clients_weight_update)

        total_data_points = sum([client.data_count for client in clients])
        fed_avg_freqs = [client.data_count / total_data_points for client in clients]

        with torch.no_grad():
            avg_params = {}
            for key in clients_weight[0]:
                avg_params[key] = sum(w * p[key] for w, p in zip(fed_avg_freqs, clients_weight))

            server_model.load_state_dict(avg_params, strict=True)

