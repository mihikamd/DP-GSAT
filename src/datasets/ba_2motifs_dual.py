# https://github.com/divelab/DIG/blob/dig/dig/xgraph/dataset/syn_dataset.py

import os
import yaml
import torch
import pickle
import numpy as np
import os.path as osp
from pathlib import Path
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, InMemoryDataset, download_url

import networkx as nx
from rdkit import Chem
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx


def read_ba2motif_data(folder: str, prefix):
    with open(os.path.join(folder, f"{prefix}.pkl"), 'rb') as f:
        dense_edges, node_features, graph_labels = pickle.load(f)

    # print(dense_edges.shape) #(1000, 25, 25)
    # print(node_features.shape) # (1000, 25, 10)

    # creating new dense edges, replaces 0,1 with numbered edges, adjacency matrix 1 if in the same row
    numbered_edges = dense_edges.copy()

    dual_dense_edges_list = []
    dual_node_features_list = []
    dual_node_label_lists = []
    dual_features_dim  = node_features.shape[2]*2


    for graph_idx, graph in enumerate(numbered_edges):
        edge_num = 0
        total_edges = int((graph.sum() / 2).item())
        graph -= 2
        dual_node_features = np.zeros((total_edges, dual_features_dim))
        dual_node_label = torch.zeros(total_edges).float()

        for node1 in range(graph.shape[0]):
            for node2 in range(graph.shape[1]):
               if (graph[node1][node2]) == -1 and node1 != node2:
                   graph[node1][node2] = edge_num
                   graph[node2][node1] = edge_num
                   if(node1 >= 20 and node2 >= 20):
                       dual_node_label[edge_num] = 1
                   dual_node_features[edge_num] = np.concatenate((node_features[graph_idx, node1],node_features[graph_idx, node2]), axis=0)
                   edge_num+=1

        dual_dense = np.zeros((edge_num, edge_num))
        for row in graph:
            edges = row[row != -2]
            for i in edges:
                for j in edges:
                    if i!=j:
                        dual_dense[int(i),int(j)] = 1

        dual_dense_edges_list.append(dual_dense)
        dual_node_features_list.append(dual_node_features)
        dual_node_label_lists.append(dual_node_label)

    # creating new node_features
    
    data_list = []
    for graph_idx, graph in enumerate(dual_dense_edges_list):
        x = torch.from_numpy(dual_node_features_list[graph_idx]).float()
        edge_index = dense_to_sparse(torch.from_numpy(dual_dense_edges_list[graph_idx]))[0]
        y = torch.from_numpy(np.where(graph_labels[graph_idx])[0]).reshape(-1, 1).float()

        dual_node_label = dual_node_label_lists[graph_idx]
        dual_edge_label = torch.zeros((edge_index.shape[1]))
        
        for edge_idx in range(edge_index.shape[1]):
            i = edge_index[0, edge_idx]
            j = edge_index[1, edge_idx]
            if dual_node_label[i] == 1 and dual_node_label[j] == 1:
                dual_edge_label[edge_idx] = 1

        # if graph_idx < 10:
        #     edge_att = torch.ones(edge_index.shape[1])
        #     fig, ax = plt.subplots()
        #     fig.canvas.manager.set_window_title(f"Dual Graph {graph_idx}")
                    
        #     visualize_a_graph(edge_index, edge_att, dual_node_label, 'ba_2motifs', ax, coor=None, norm=False, mol_type=None, nodesize=300)
                    
        #     plt.show(block=True)
        #     plt.close(fig)
        #     input("continue to next graph")

        data_list.append(Data(x=x, edge_index=edge_index, y=y, node_label=dual_node_label, edge_label=dual_edge_label))
    return data_list


class SynGraphDataset(InMemoryDataset):
    r"""
    The Synthetic datasets used in
    `Parameterized Explainer for Graph Neural Network <https://arxiv.org/abs/2011.04573>`_.
    It takes Barabási–Albert(BA) graph or balance tree as base graph
    and randomly attachs specific motifs to the base graph.
    Args:
        root (:obj:`str`): Root data directory to save datasets
        name (:obj:`str`): The name of the dataset. Including :obj:`BA_shapes`, BA_grid,
        transform (:obj:`Callable`, :obj:`None`): A function/transform that takes in an
            :class:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (:obj:`Callable`, :obj:`None`):  A function/transform that takes in
            an :class:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'https://github.com/divelab/DIG_storage/raw/main/xgraph/datasets/{}'
    # Format: name: [display_name, url_name, filename]
    names = {
        'ba_shapes': ['BA_shapes', 'BA_shapes.pkl', 'BA_shapes'],
        'ba_community': ['BA_Community', 'BA_Community.pkl', 'BA_Community'],
        'tree_grid': ['Tree_Grid', 'Tree_Grid.pkl', 'Tree_Grid'],
        'tree_cycle': ['Tree_Cycle', 'Tree_Cycles.pkl', 'Tree_Cycles'],
        'ba_2motifs': ['BA_2Motifs', 'BA_2Motifs.pkl', 'BA_2Motifs']
    }

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        super(SynGraphDataset, self).__init__(root, transform, pre_transform)
        self.process()
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return f'{self.names[self.name][2]}.pkl'

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        url = self.url.format(self.names[self.name][1])
        download_url(url, self.raw_dir)

    def process(self):
        if self.name.lower() == 'BA_2Motifs'.lower():
            data_list = read_ba2motif_data(self.raw_dir, self.names[self.name][2])

            if self.pre_filter is not None:
                data_list = [self.get(idx) for idx in range(len(self))]
                data_list = [data for data in data_list if self.pre_filter(data)]
                self.data, self.slices = self.collate(data_list)

            if self.pre_transform is not None:
                data_list = [self.get(idx) for idx in range(len(self))]
                data_list = [self.pre_transform(data) for data in data_list]
                self.data, self.slices = self.collate(data_list)
        else:
            # Read data into huge `Data` list.
            data = self.read_syn_data()
            data = data if self.pre_transform is None else self.pre_transform(data)
            data_list = [data]

        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.names[self.name][0], len(self))

    def read_syn_data(self):
        with open(self.raw_paths[0], 'rb') as f:
            adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix = pickle.load(f)

        x = torch.from_numpy(features).float()
        y = train_mask.reshape(-1, 1) * y_train + val_mask.reshape(-1, 1) * y_val + test_mask.reshape(-1, 1) * y_test
        y = torch.from_numpy(np.where(y)[1])
        edge_index = dense_to_sparse(torch.from_numpy(adj))[0]
        data = Data(x=x, y=y, edge_index=edge_index)
        data.train_mask = torch.from_numpy(train_mask)
        data.val_mask = torch.from_numpy(val_mask)
        data.test_mask = torch.from_numpy(test_mask)
        return data

def visualize_a_graph(edge_index, edge_att, node_label, dataset_name, ax, coor=None, norm=False, mol_type=None, nodesize=300):
    if norm:  # for better visualization
        edge_att = edge_att**10
        edge_att = (edge_att - edge_att.min()) / (edge_att.max() - edge_att.min() + 1e-6)

    if mol_type is None or dataset_name == 'Graph-SST2':
        atom_colors = {0: '#E49D1C', 1: '#FF5357', 2: '#a1c569', 3: '#69c5ba'}
        node_colors = [None for _ in range(node_label.shape[0])]
        for y_idx in range(node_label.shape[0]):
            node_colors[y_idx] = atom_colors[node_label[y_idx].int().tolist()]
    else:
        node_color = ['#29A329', 'lime', '#F0EA00',  'maroon', 'brown', '#E49D1C', '#4970C6', '#FF5357']
        element_idxs = {k: Chem.PeriodicTable.GetAtomicNumber(Chem.GetPeriodicTable(), v) for k, v in mol_type.items()}
        node_colors = [node_color[(v - 1) % len(node_color)] for k, v in element_idxs.items()]

    data = Data(edge_index=edge_index, att=edge_att, y=node_label, num_nodes=node_label.size(0)).to('cpu')
    G = to_networkx(data, node_attrs=['y'], edge_attrs=['att'])

    if coor is None:
        pos = nx.kamada_kawai_layout(G)

    else:
        pos = {idx: each.tolist() for idx, each in enumerate(coor)}

    for source, target, data in G.edges(data=True):
        ax.annotate(
            '', xy=pos[target], xycoords='data', xytext=pos[source],
            textcoords='data', arrowprops=dict(
                arrowstyle="->" if dataset_name == 'Graph-SST2' else '-',
                lw=max(data['att'], 0) * 3,
                alpha=max(data['att'], 0),  # alpha control transparency
                color='black',  # color control color
                shrinkA=np.sqrt(nodesize) / 2.0 + 1,
                shrinkB=np.sqrt(nodesize) / 2.0 + 1,
                connectionstyle='arc3,rad=0.4' if dataset_name == 'Graph-SST2' else 'arc3'
            ))

    if mol_type is not None:
        nx.draw_networkx_labels(G, pos, mol_type, ax=ax)

    if dataset_name != 'Graph-SST2':
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=nodesize, ax=ax)
        nx.draw_networkx_edges(G, pos, width=1, edge_color='gray', arrows=False, alpha=0.1, ax=ax)
    else:
        nx.draw_networkx_edges(G, pos, width=1, edge_color='gray', arrows=False, alpha=0.1, ax=ax, connectionstyle='arc3,rad=0.4')

    labels = {i: str(i) for i in G.nodes}

    nx.draw_networkx_labels(G, pos, labels, font_size=12, ax=ax)

    x_values = [p[0] for p in pos.values()]
    y_values = [p[1] for p in pos.values()]

    x_margin = (max(x_values) - min(x_values)) * 0.2 + 0.1
    y_margin = (max(y_values) - min(y_values)) * 0.2 + 0.1

    ax.set_xlim(min(x_values) - x_margin, max(x_values) + x_margin)
    ax.set_ylim(min(y_values) - y_margin, max(y_values) + y_margin)

    ax.set_aspect('equal')
    ax.axis('off')
