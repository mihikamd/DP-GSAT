import torch
import numpy as np
from torch_geometric.data import Batch
from torch_geometric.utils import degree
from torch_geometric.loader import DataLoader

from ogb.graphproppred import PygGraphPropPredDataset
from datasets import SynGraphDataset, SynGraphDataset_Dual, Mutag, SPMotif, MNIST75sp, graph_sst2, Mutag_Dual

#from trainer import run_one_epoch, update_best_epoch_res, get_viz_idx, visualize_results # MANGO

import matplotlib.pyplot as plt #MANGO
import networkx as nx #MANGO
from torch_geometric.utils import to_networkx #MANGO
import numpy as np #MANGO
from rdkit import Chem #MANGO


def get_data_loaders(data_dir, dataset_name, batch_size, splits, random_state, mutag_x=False):
    multi_label = False
    assert dataset_name in ['ba_2motifs', 'ba_2motifs_dual', 'mutag', 'mutag_dual', 'Graph-SST2', 'mnist',
                            'spmotif_0.5', 'spmotif_0.7', 'spmotif_0.9',
                            'ogbg_molhiv', 'ogbg_moltox21', 'ogbg_molbace',
                            'ogbg_molbbbp', 'ogbg_molclintox', 'ogbg_molsider']

    if dataset_name == 'ba_2motifs':
        dataset = SynGraphDataset(data_dir, 'ba_2motifs')
        split_idx = get_random_split_idx(dataset, splits, random_state=random_state)
        loaders, test_set = get_loaders_and_test_set(batch_size, dataset=dataset, split_idx=split_idx)
        train_set = dataset[split_idx["train"]]

    elif dataset_name == 'ba_2motifs_dual':
        dataset = SynGraphDataset_Dual(data_dir, 'ba_2motifs_dual')
        split_idx = get_random_split_idx(dataset, splits, random_state=random_state)
        loaders, test_set = get_loaders_and_test_set(batch_size, dataset=dataset, split_idx=split_idx)
        train_set = dataset[split_idx["train"]]

    elif dataset_name == 'mutag':
        dataset = Mutag(root=data_dir / 'mutag')
        split_idx = get_random_split_idx(dataset, splits, random_state=random_state, mutag_x=mutag_x)
        loaders, test_set = get_loaders_and_test_set(batch_size, dataset=dataset, split_idx=split_idx)
        print("\U0001F96D HERE GOD DAMNIT IM HERE") #MANGO
        train_set = dataset[split_idx['train']]

    elif dataset_name == 'mutag_dual':
        dataset = Mutag_Dual(root=data_dir / 'mutag_dual') #MANGO
        split_idx = get_random_split_idx(dataset, splits, random_state=random_state, mutag_x=mutag_x)
        loaders, test_set = get_loaders_and_test_set(batch_size, dataset=dataset, split_idx=split_idx) #MANGO
        print("\U0001F96D HERE CRAP FUCKING BUCKETS") #MANGO
        train_set = dataset[split_idx['train']] #MANGO

    elif 'ogbg' in dataset_name:
        dataset = PygGraphPropPredDataset(root=data_dir, name='-'.join(dataset_name.split('_')))
        split_idx = dataset.get_idx_split()
        print('[INFO] Using default splits!')
        loaders, test_set = get_loaders_and_test_set(batch_size, dataset=dataset, split_idx=split_idx)
        train_set = dataset[split_idx['train']]

    elif dataset_name == 'Graph-SST2':
        dataset = graph_sst2.get_dataset(dataset_dir=data_dir, dataset_name='Graph-SST2', task=None)
        dataloader, (train_set, valid_set, test_set) = graph_sst2.get_dataloader(dataset, batch_size=batch_size, degree_bias=True, seed=random_state)
        print('[INFO] Using default splits!')
        loaders = {'train': dataloader['train'], 'valid': dataloader['eval'], 'test': dataloader['test']}
        test_set = dataset  # used for visualization

    elif 'spmotif' in dataset_name:
        b = float(dataset_name.split('_')[-1])
        train_set = SPMotif(root=data_dir / dataset_name, b=b, mode='train')
        valid_set = SPMotif(root=data_dir / dataset_name, b=b, mode='val')
        test_set = SPMotif(root=data_dir / dataset_name, b=b, mode='test')
        print('[INFO] Using default splits!')
        loaders, test_set = get_loaders_and_test_set(batch_size, dataset_splits={'train': train_set, 'valid': valid_set, 'test': test_set})

    elif dataset_name == 'mnist':
        n_train_data, n_val_data = 20000, 5000
        train_val = MNIST75sp(data_dir / 'mnist', mode='train')
        perm_idx = torch.randperm(len(train_val), generator=torch.Generator().manual_seed(random_state))
        train_val = train_val[perm_idx]

        train_set, valid_set = train_val[:n_train_data], train_val[-n_val_data:]
        test_set = MNIST75sp(data_dir / 'mnist', mode='test')
        loaders, test_set = get_loaders_and_test_set(batch_size, dataset_splits={'train': train_set, 'valid': valid_set, 'test': test_set})
        print('[INFO] Using default splits!')

    x_dim = test_set[0].x.shape[1]
    edge_attr_dim = 0 if test_set[0].edge_attr is None else test_set[0].edge_attr.shape[1] #do edge features exist
    if isinstance(test_set, list): #if list of graphs
        num_class = Batch.from_data_list(test_set).y.unique().shape[0] #converts to batched PyG object, extracts y labels then num unique
    elif test_set.data.y.shape[-1] == 1 or len(test_set.data.y.shape) == 1:
        num_class = test_set.data.y.unique().shape[0]
    else:
        num_class = test_set.data.y.shape[-1]
        multi_label = True

    print('[INFO] Calculating degree...')
    # Compute in-degree histogram over training data.
    # deg = torch.zeros(10, dtype=torch.long)
    # for data in train_set:
    #     d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    #     deg += torch.bincount(d, minlength=deg.numel())
    batched_train_set = Batch.from_data_list(train_set)
    d = degree(batched_train_set.edge_index[1], num_nodes=batched_train_set.num_nodes, dtype=torch.long) #degree
    deg = torch.bincount(d, minlength=10) #degree higher than 10 is dropped

    aux_info = {'deg': deg, 'multi_label': multi_label} #multi-label = false
    return loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info


def get_random_split_idx(dataset, splits, random_state=None, mutag_x=False):
    if random_state is not None:
        np.random.seed(random_state)

    print('[INFO] Randomly split dataset!')
    idx = np.arange(len(dataset))
    np.random.shuffle(idx)

    if not mutag_x:
        n_train, n_valid = int(splits['train'] * len(idx)), int(splits['valid'] * len(idx))
        train_idx = idx[:n_train]
        valid_idx = idx[n_train:n_train+n_valid]
        test_idx = idx[n_train+n_valid:]
    else:
        print('[INFO] mutag_x is True!')
        #n_train = int(splits['train'] * len(idx))
        n_train, n_valid = int(splits['train'] * len(idx)), int(splits['valid'] * len(idx))
        train_idx, valid_idx = idx[:n_train], idx[n_train:]
        test_idx = idx[n_train+n_valid:] #MANGO
        #test_idx = [i for i in range(len(dataset)) if (dataset[i].y.squeeze() == 0 and dataset[i].edge_label.sum() > 0)] # MANGO ASK CALLIE WHY ?*
    return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}


def get_loaders_and_test_set(batch_size, dataset=None, split_idx=None, dataset_splits=None):
    if split_idx is not None:
        assert dataset is not None
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=False) #used to be true
        valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=False)
        if len(split_idx["test"]) == 0:
            raise ValueError("split_idx['test'] is empty — cannot create test set.")
        test_set = dataset.copy(split_idx["test"])  # For visualization
    else:
        assert dataset_splits is not None
        train_loader = DataLoader(dataset_splits['train'], batch_size=batch_size, shuffle=False) #used to be true
        valid_loader = DataLoader(dataset_splits['valid'], batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset_splits['test'], batch_size=batch_size, shuffle=False)
        test_set = dataset_splits['test']  # For visualization
    return {'train': train_loader, 'valid': valid_loader, 'test': test_loader}, test_set

#CHAT GPT ALERT MANGOS FALLING

def simple_visualize(test_set, dataset_name, num_graphs=1):
    fig, axes = plt.subplots(1, num_graphs, figsize=(5 * num_graphs, 5))

    for i in range(num_graphs):
        data = test_set[i]
        ax = axes[i] if num_graphs > 1 else axes

        # Optional node labels (for color or text)
        if dataset_name == 'mutag':
            node_dict = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S', 8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'}
            node_types = data.node_type.view(-1)
            mol_type = {k: node_dict[int(v)] for k, v in enumerate(node_types)}

        elif dataset_name == 'Graph-SST2':
            mol_type = {k: v for k, v in enumerate(data.sentence_tokens)}
        elif dataset_name == 'ogbg_molhiv':
            element_idxs = {k: int(data.x[k, 0]) + 1 for k in range(data.num_nodes)}
            mol_type = {k: Chem.PeriodicTable.GetElementSymbol(Chem.GetPeriodicTable(), v) for k, v in element_idxs.items()}
        else:
            mol_type = None

        G = to_networkx(data, to_undirected=True)

        mol_type = {k: v for k, v in mol_type.items() if k in G.nodes}
        # Layout
        pos = nx.kamada_kawai_layout(G)

        # Node colors by label if available
        node_label = getattr(data, 'node_label', None)
        if node_label is not None:
            node_colors = [node_label[i].item() for i in range(data.num_nodes)]
        else:
            node_colors = 'lightblue'

        # Draw
        nx.draw(G, pos, ax=ax, with_labels=False, node_color=node_colors, cmap='Set3', node_size=300, edge_color='gray')

        assert hasattr(data, 'edge_index'), f"Graph {i} missing edge_index"
        assert data.edge_index is not None, f"Graph {i} has edge_index=None"
        assert data.edge_index.ndim == 2 and data.edge_index.shape[0] == 2, \
        f"Graph {i} edge_index shape invalid: {data.edge_index.shape}"
        assert data.num_nodes > 0, f"Graph {i} has no nodes"

        if mol_type:
            nx.draw_networkx_labels(G, pos, labels=mol_type, font_size=8, ax=ax)

        ax.set_title(f"Graph {i}")
    
    # plt.show(block=False)
    # plt.pause(3)  # keep it open for 3 seconds
    # plt.close()
    plt.tight_layout()
    plt.show()

    #CHATGPT ALERT MANGOS LANDING

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

    # calculate Graph positions
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
