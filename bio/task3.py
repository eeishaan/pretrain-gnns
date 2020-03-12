DEBUG = False

import os
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim

from model import GNN
from util import MaskEdge
from loader import BioDataset
from dataloader import DataLoaderMasking
from pretrain_masking import compute_accuracy
from splitters import random_split,species_split

if DEBUG is True:
    import multiprocessing
    multiprocessing.set_start_method('spawn', True)


def finetune_linear(model, linear, optimizer, criterion, train_loader, valid_loader,
                    device, savepath, valid_period=16, valid_steps=16):
    model.eval()
    linear.train()

    loss_accum = 0
    acc_accum = 0

    best_loss = 9999

    valid_iterator = iter(valid_loader)

    for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
        linear.train()
        batch = batch.to(device)

        node_rep = model(batch.x, batch.edge_index, batch.edge_attr)

        ### predict the edge types.
        masked_edge_index = batch.edge_index[:, batch.masked_edge_idx]
        edge_rep = node_rep[masked_edge_index[0]] + node_rep[masked_edge_index[1]]
        pred_edge = linear(edge_rep)

        #converting the binary classification to multiclass classification
        edge_label = torch.argmax(batch.mask_edge_label, dim = 1)

        acc_edge = compute_accuracy(pred_edge, edge_label)
        acc_accum += acc_edge

        optimizer.zero_grad()

        loss = criterion(pred_edge, edge_label)
        loss.backward()

        optimizer.step()

        loss_accum += float(loss.cpu().item())

        #validation iterations
        with torch.no_grad():
            if step % valid_period == 0:
                linear.eval()
                valid_acc_accum = 0
                valid_loss_accum = 0

                for _ in range(valid_steps):
                    try:
                        valid_batch = next(valid_iterator)
                    except StopIteration:
                        valid_iterator = iter(valid_loader)
                        valid_batch = next(valid_iterator)

                    valid_batch = valid_batch.to(device)

                    node_rep = model(valid_batch.x, valid_batch.edge_index, valid_batch.edge_attr)

                    ### predict the edge types.
                    masked_edge_index = valid_batch.edge_index[:, valid_batch.masked_edge_idx]
                    edge_rep = node_rep[masked_edge_index[0]] + node_rep[masked_edge_index[1]]
                    pred_edge = linear(edge_rep)

                    #converting the binary classification to multiclass classification
                    edge_label = torch.argmax(valid_batch.mask_edge_label, dim = 1)

                    valid_acc_edge = compute_accuracy(pred_edge, edge_label)
                    valid_acc_accum += valid_acc_edge

                    valid_loss = criterion(pred_edge, edge_label)
                    valid_loss_accum += float(valid_loss.cpu().item())


                if best_loss > valid_loss_accum:
                    best_loss = valid_loss_accum
                    torch.save(linear.state_dict(), savepath)

    return loss_accum/(step + 1), acc_accum/(step + 1)


def eval_accuracy(model, linear, test_loader, device):
    model.eval()
    linear.eval()

    acc_accum = 0

    for step, batch in enumerate(tqdm(test_loader, desc="Iteration")):
        batch = batch.to(device)

        node_rep = model(batch.x, batch.edge_index, batch.edge_attr)

        ### predict the edge types.
        masked_edge_index = batch.edge_index[:, batch.masked_edge_idx]
        edge_rep = node_rep[masked_edge_index[0]] + node_rep[masked_edge_index[1]]
        pred_edge = linear(edge_rep)

        #converting the binary classification to multiclass classification
        edge_label = torch.argmax(batch.mask_edge_label, dim = 1)

        acc_edge = compute_accuracy(pred_edge, edge_label)
        acc_accum += acc_edge

    return acc_accum/(step + 1)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--mask_rate', type=float, default=0.15,
                        help='dropout ratio (default: 0.15)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features are combined across layers. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--model_file', type=str, default = '', help='filename to output the model')
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--weights', type=str, default="model_gin/masking.pth",
                        help='path to weights checkpoint')
    parser.add_argument('--savepath', type=str, default=None,
                        help='path where the linear checkpoint is saved')
    parser.add_argument('--linear_weights', type=str, default=None,
                        help="if specified, evaluates accuracy of models instead of finetuning")
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # Set up dataset
    root_unsupervised = 'dataset/unsupervised'
    dataset = BioDataset(root_unsupervised,
                         data_type='unsupervised',
                         transform=MaskEdge(mask_rate=args.mask_rate))

    print(dataset)

    # Split the dataset into train-valid-test
    # To conserve ratios, train-val:test is split 43:7 between species
    species_list = [int(species) for species in dataset.raw_file_names]

    test_list = np.random.choice(species_list, size=7, replace=False).tolist()
    
    trainval_list = list(set(species_list) - set(test_list))
    trainval_list.sort()

    trainval_dataset, test_dataset = species_split(dataset,
                                                   trainval_list,
                                                   test_list)

    del dataset
    
    # Load the model weights
    model = GNN(args.num_layer,
                args.emb_dim,
                JK=args.JK,
                drop_ratio=args.dropout_ratio,
                gnn_type=args.gnn_type).to(device)

    model.load_state_dict(torch.load(args.weights, map_location=device))

    for p in model.parameters():
        p.requires_grad=False

    # Get the last layer, either through training or loading a checkpoint
    linear_pred_edges = torch.nn.Linear(args.emb_dim, 7).to(device)

    if args.linear_weights is None:
        print("Finetuning last layer using: {}".format(args.weights))
        # Train:val is randomly split 0.85:0.15
        train_dataset, valid_dataset, _ = random_split(trainval_dataset,
                                                   seed=args.seed,
                                                   frac_train=0.85,
                                                   frac_valid=0.15,
                                                   frac_test=0)

        del trainval_dataset

        train_loader = DataLoaderMasking(train_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        num_workers=args.num_workers)

        valid_loader = DataLoaderMasking(valid_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        num_workers=args.num_workers)

        optimizer_linear_pred_edges = optim.Adam(linear_pred_edges.parameters(),
                                                lr=args.lr,
                                                weight_decay=args.decay)
    
        criterion = nn.CrossEntropyLoss()
        
        loss, acc = finetune_linear(model,
                                    linear_pred_edges,
                                    optimizer_linear_pred_edges,
                                    criterion,
                                    train_loader,
                                    valid_loader,
                                    device,
                                    args.savepath)

        print("Final loss:", loss)
    
    elif os.path.exists(args.linear_weights) is False:
        print("Incorrect linear checkpoint path")
        acc = "invalid"
    
    else:
        print("Evaluating model: {} \nUsing last layer: {}".format(args.weights, args.linear_weights))
        test_loader = DataLoaderMasking(test_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.num_workers)

        linear_pred_edges.load_state_dict(torch.load(args.linear_weights, map_location=device))
        acc = eval_accuracy(model,
                            linear_pred_edges,
                            test_loader,
                            device)

    print("Final acc:", acc)
    