import os
import os.path as osp
from collections import OrderedDict

import fire
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
# from core.encoders import *

from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import sys
import json
from torch import optim
from tqdm import tqdm

from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from loader import BioDataset
from losses import *
from gin import Encoder
from evaluate_embedding import evaluate_embedding

from arguments import arg_parse
from infograph_model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def gnn_key_transformation(old_name, new_name):
    def internal_gnn_key_transformation(key):
        return key.replace(f"{old_name}.", f"{new_name}.")

    return internal_gnn_key_transformation


def rename_state_dict_keys(state_dict, key_transformation):
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        new_key = key_transformation(key)
        new_state_dict[new_key] = value
    return new_state_dict


class GcnInfomax(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, prior, alpha=0.5, beta=1., gamma=.1, ):
        super(GcnInfomax, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = prior

        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.encoder = Encoder(hidden_dim, num_gc_layers)

        self.local_d = FF(self.embedding_dim)
        self.global_d = FF(self.embedding_dim)
        # self.local_d = MI1x1ConvNet(self.embedding_dim, mi_units)
        # self.global_d = MIFCNet(self.embedding_dim, mi_units)

        if self.prior:
            self.prior_d = PriorDiscriminator(self.embedding_dim)

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        # batch_size = data.num_graphs
        if x is None:
            data.x = torch.ones(batch.shape[0]).to(device)

        y, M = self.encoder(data)

        g_enc = self.global_d(y)
        l_enc = self.local_d(M)

        mode = 'fd'
        measure = 'JSD'
        local_global_loss = local_global_loss_(l_enc, g_enc, edge_index, batch, measure)

        if self.prior:
            prior = torch.rand_like(y)
            term_a = torch.log(self.prior_d(prior)).mean()
            term_b = torch.log(1.0 - self.prior_d(y)).mean()
            PRIOR = - (term_a + term_b) * self.gamma
        else:
            PRIOR = 0

        return local_global_loss + PRIOR


def pretrain_infograph(run_seed=0,
                       pretrained_node_lvl_gnn_path="../model_gin/masking.pth",
                       lr=0.001,
                       epochs=3,
                       hidden_dim=300,
                       num_gc_layers=5,
                       local=False,
                       prior=False,
                       output_filename="unsupervised_pretrain_graph_lvl_infograph_epoch.pth"):
    torch.manual_seed(run_seed)
    np.random.seed(run_seed)
    accuracies = {'logreg': [], 'svc': [], 'linearsvc': [], 'randomforest': []}
    log_interval = 1
    batch_size = 128

    # set up dataset
    root_unsupervised = 'dataset/unsupervised'
    dataset = BioDataset(root_unsupervised, data_type='unsupervised')

    dataloader = DataLoader(dataset, batch_size=batch_size)

    model = GcnInfomax(hidden_dim, num_gc_layers, prior).to(device)
    if not pretrained_node_lvl_gnn_path == "":
        pretrain_model_state_dict = torch.load(pretrained_node_lvl_gnn_path, map_location='cuda:0')
        pretrain_model_state_dict = rename_state_dict_keys(pretrain_model_state_dict,
                                                           gnn_key_transformation(old_name="gnns", new_name="convs"))
        model.encoder.load_state_dict(pretrain_model_state_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print('================')
    print('lr: {}'.format(lr))
    # print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(hidden_dim))
    print('num_gc_layers: {}'.format(num_gc_layers))
    print('================')

    model.eval()
    # emb, y = model.encoder.get_embeddings(dataloader)
    # res = evaluate_embedding(emb, y)
    # accuracies['logreg'].append(res[0])
    # accuracies['svc'].append(res[1])
    # accuracies['linearsvc'].append(res[2])
    # accuracies['randomforest'].append(res[3])

    for epoch in range(1, epochs + 1):
        loss_all = 0
        model.train()
        for data in tqdm(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            loss = model(data)
            loss_all += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()

        print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader)))
    os.makedirs("result/unsupervised_graph_lvl_seed" + str(run_seed), exist_ok=True)

    if not output_filename == "":
        with open("result/unsupervised_graph_lvl_seed" + str(run_seed) + "/" + output_filename, 'wb') as f:
            torch.save(model.encoder.state_dict(), f)
        # if epoch % log_interval == 0:
        #     model.eval()
        #     emb, y = model.encoder.get_embeddings(dataloader)
        #     res = evaluate_embedding(emb, y)
        #     accuracies['logreg'].append(res[0])
        #     accuracies['svc'].append(res[1])
        #     accuracies['linearsvc'].append(res[2])
        #     accuracies['randomforest'].append(res[3])
        #     print(accuracies)

    tpe = ('local' if local else '') + ('prior' if prior else '')
    with open('new_log', 'a+') as f:
        s = json.dumps(accuracies)
        f.write('{},{},{},{},{},{},{}\n'.format("unsupervised", tpe, num_gc_layers, epochs, log_interval, lr, s))


if __name__ == '__main__':
    fire.Fire(pretrain_infograph)
