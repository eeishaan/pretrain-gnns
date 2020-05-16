'''
load the given model and prune the weights
save the new model along with the pruned weights mask
'''
import json
import os

import fire
import torch
import torch.nn as nn

from model import GNN, GNN_graphpred


def process_linear(obj, prune_ratio, invert_mask):
    weights = obj.weight.abs()
    smallest_element = int(weights.numel() * prune_ratio)
    kth_largest = weights.view(-1).kthvalue(smallest_element).values
    mask = (weights <= kth_largest)
    if invert_mask:
        mask = ~mask
    return mask


def process_children(obj, prune_ratio, invert_mask):
    mask_obj = []
    for child in obj.children():
        if isinstance(child, nn.Linear):
            child_mask_obj = process_linear(child, prune_ratio, invert_mask)
        else:
            child_mask_obj = process_children(child, prune_ratio, invert_mask)
        mask_obj.append(child_mask_obj)
    return mask_obj


def prune(model_path, mask_out, prune_ratio=0.25, mask_in=None, invert_mask=False):
    model_params = {
        'num_layer': 5,
        'emb_dim': 300,
        'JK': 'last',
        'drop_ratio': 0.15,
        'gnn_type': 'gin'
    }

    device = torch.device(
        "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model = GNN(**model_params).to(device)

    model.load_state_dict(torch.load(
        model_path, map_location=lambda storage, loc: storage))

    mask_obj = process_children(model, prune_ratio, invert_mask)
    dir_name = os.path.dirname(mask_out)
    os.makedirs(dir_name, exist_ok=True)
    torch.save(mask_obj, mask_out)


fire.Fire(prune)
