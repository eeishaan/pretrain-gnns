import wandb

wandb.init(project="pretrain-gnn", entity="lap1n")

import os

import fire
import numpy as np
import pickle
import torch
from torch import optim, nn
from torch.utils import data
from tqdm import tqdm

from batch import BatchFinetune
from loader import BioDataset
from siamese_model import SiameseModel

criterion = nn.MSELoss()


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss


def get_pair_of_graph_from_pair_id(graph_dataset, pair_id_batch):
    graphs_1 = []
    graphs_2 = []

    for pair_id in pair_id_batch:
        graph_1_idx = int(pair_id / len(graph_dataset))
        graph_2_idx = int(pair_id % len(graph_dataset))
        graphs_1.append(graph_dataset[graph_1_idx])
        graphs_2.append(graph_dataset[graph_2_idx])
    batch_graph_1 = BatchFinetune.from_data_list(graphs_1)
    batch_graph_2 = BatchFinetune.from_data_list(graphs_2)
    return batch_graph_1, batch_graph_2


def train_siamese_model(model, device, loader, graph_dataset, optimizer):
    model.train()
    train_loss_accum = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        graph_pair_id, wl_kernel_score = batch[0], batch[1]
        wl_kernel_score = wl_kernel_score.cuda()
        batch_graph_1, batch_graph_2 = get_pair_of_graph_from_pair_id(graph_dataset, graph_pair_id)
        batch_graph_1 = batch_graph_1.to(device)
        batch_graph_2 = batch_graph_2.to(device)
        wl_score_pred = model((batch_graph_1, batch_graph_2))
        optimizer.zero_grad()
        loss = criterion(wl_score_pred, wl_kernel_score)
        loss.backward()
        optimizer.step()
        train_loss_accum += float(loss.detach().cpu().item())

    return train_loss_accum / (step + 1)


def pretrain_unsupervised_graph_level(device=0,
                                      pretrained_node_lvl_gnn_path="./model_gin/masking.pth",
                                      batch_size=8,
                                      lr=0.001,
                                      decay=0,
                                      run_seed=0,
                                      num_epochs=5,
                                      num_layer=5,
                                      emb_dim=300,
                                      JK="last",
                                      dropout_ratio=0,
                                      gnn_type="gin",
                                      num_workers=8,
                                      data_type="unsupervised",
                                      base_dataset_path="./dataset",
                                      output_filename="unsupervised_pretrain_graph_lvl_50_epoch.pth"):
    torch.manual_seed(run_seed)
    np.random.seed(run_seed)
    device = torch.device("cuda:" + str(device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)

    wl_kernel_data_path = os.path.join(base_dataset_path, data_type, "processed/wl_scores.npy")
    wl_kernel_data = np.load(wl_kernel_data_path)
    x = torch.Tensor(np.arange(len(wl_kernel_data)))
    y = torch.Tensor(wl_kernel_data)
    wl_kernel_dataset = data.TensorDataset(x, y)  # create your datset
    wl_kernel_dataloader = data.DataLoader(wl_kernel_dataset, num_workers=num_workers,
                                           batch_size=batch_size)  # create your dataloader
    dataset_root_path = os.path.join(base_dataset_path, data_type)
    graph_dataset = BioDataset(dataset_root_path, data_type=data_type)

    model = SiameseModel(num_layer, emb_dim, JK=JK, drop_ratio=dropout_ratio, gnn_type=gnn_type)
    if not pretrained_node_lvl_gnn_path == "":
        pretrain_model_state_dict = torch.load(pretrained_node_lvl_gnn_path, map_location='cuda:0')
        model.model.load_state_dict(pretrain_model_state_dict)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

    for epoch in range(1, num_epochs + 1):
        print("====epoch " + str(epoch))
        train_loss = train_siamese_model(model, device, wl_kernel_dataloader, graph_dataset, optimizer)
        wandb.log({"train_loss": train_loss}, step=epoch)

    os.makedirs("result/unsupervised_graph_lvl_seed" + str(run_seed), exist_ok=True)

    if not output_filename == "":
        with open("result/unsupervised_graph_lvl_seed" + str(run_seed) + "/" + output_filename, 'wb') as f:
            torch.save(model.model.state_dict(), f)


if __name__ == "__main__":
    fire.Fire(pretrain_unsupervised_graph_level)
