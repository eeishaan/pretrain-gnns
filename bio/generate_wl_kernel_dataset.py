import os
import fire as fire
import torch
import numpy as np
from grakel import WeisfeilerLehman
from loader import BioDataset

counter = 0


def grakel_graph_format_transform(X):
    """
    :param X: object containing at least :
                x (Tensor, optional): Node feature matrix with shape :obj:`[num_nodes,
                    num_node_features]`. (default: :obj:`None`)
                edge_index (LongTensor, optional): Graph connectivity in COO format
                    with shape :obj:`[2, num_edges]`. (default: :obj:`None`)
                edge_attr (Tensor, optional): Edge feature matrix with shape
                    :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
    :return: tuple of :
                g : a valid graph format
                    Similar to intialization object (of ``__init__``).
                    set of edges  : (node_id_1,node_id_2)

                node_labels: dict, default=None
                    Node labels dictionary relevant to g format :
                    dictionary {node_id, node_label}


                edge_labels: dict, default=None
                    Edge labels dictionary relevant to g format.
                    dictionary {(node_id_1,node_id_2):edge_label}

    """
    edge_set = set()
    edge_index_numpy = X.edge_index.numpy().T
    for (node_1, node_2) in edge_index_numpy:
        edge_set.add((node_1, node_2))
    node_features = {}
    for node_id, node_value in enumerate(X.x.numpy().squeeze()):
        node_features[node_id] = node_value
    edge_features = {}
    for edge_id, edge_value in enumerate(X.edge_attr.numpy().squeeze()):
        node_pair = edge_index_numpy[edge_id]
        edge_features[(node_pair[0], node_pair[1])] = edge_value.tolist().index(max(edge_value))

    global counter
    counter += 1
    if counter % 100 == 0:
        print(f"{counter} graphs processed")
    return edge_set, node_features, edge_features


def generate_wl_kernel_dataset(num_workers=8, dataset_size=1000, data_type="unsupervised",
                               base_dataset_path="./dataset"):
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    dataset_root_path = os.path.join(base_dataset_path, data_type)
    wl_kernel = WeisfeilerLehman(n_jobs=num_workers, n_iter=2, normalize=True, verbose=True)
    dataset = BioDataset(dataset_root_path, data_type=data_type, transform=grakel_graph_format_transform)
    if dataset_size == -1:
        dataset_size = len(dataset)
    wl_similarity_scores = wl_kernel.fit_transform(iter(dataset[0:dataset_size]))
    np.save(os.path.join(dataset_root_path, "processed", "wl_scores.npy"), wl_similarity_scores.reshape(-1))


if __name__ == "__main__":
    fire.Fire(generate_wl_kernel_dataset)
