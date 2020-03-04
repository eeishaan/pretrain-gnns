import torch
import argparse
import numpy as np
from tqdm import tqdm
from model import GNN
from util import MaskEdge
from loader import BioDataset
from dataloader import DataLoaderMasking


def main(model, loader, device):
    pass


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
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
    parser.add_argument('--weights', type=str, default="model_gin/supervised_masking.pth")
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    #set up dataset
    root_unsupervised = 'dataset/unsupervised'
    dataset = BioDataset(root_unsupervised, data_type='unsupervised', transform = MaskEdge(mask_rate = args.mask_rate))

    print(dataset)

    loader = DataLoaderMasking(dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)

    model = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    main(model, loader, device)