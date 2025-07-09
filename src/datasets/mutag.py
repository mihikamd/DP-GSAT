# https://github.com/flyingdoog/PGExplainer/blob/master/MUTAG.ipynb

import yaml
import torch
import numpy as np
import pickle as pkl
from pathlib import Path
from torch_geometric.data import InMemoryDataset, Data

import matplotlib.pyplot as plt #MANGO
import networkx as nx #MANGO
from torch_geometric.utils import to_networkx #MANGO
from rdkit import Chem #MANGO


class Mutag(InMemoryDataset):
    def __init__(self, root):
        super().__init__(root=root)
        self.process() #MANGO
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['Mutagenicity_A.txt', 'Mutagenicity_edge_gt.txt', 'Mutagenicity_edge_labels.txt',
                'Mutagenicity_graph_indicator.txt', 'Mutagenicity_graph_labels.txt', 'Mutagenicity_label_readme.txt',
                'Mutagenicity_node_labels.txt', 'Mutagenicity.pkl']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        raise NotImplementedError

    def process(self):
        with open(self.raw_dir + '/Mutagenicity.pkl', 'rb') as fin:
            _, original_features, original_labels = pkl.load(fin)

        edge_lists, graph_labels, edge_label_lists, node_type_lists = self.get_graph_data()

        data_list = []
        for i in range(original_labels.shape[0]):
            num_nodes = len(node_type_lists[i])
            edge_index = torch.tensor(edge_lists[i], dtype=torch.long).T

            y = torch.tensor(graph_labels[i]).float().reshape(-1, 1)
            x = torch.tensor(original_features[i][:num_nodes]).float()
            assert original_features[i][num_nodes:].sum() == 0
            edge_label = torch.tensor(edge_label_lists[i]).float()
            if y.item() != 0:
                edge_label = torch.zeros_like(edge_label).float()

            node_label = torch.zeros(x.shape[0])
            signal_nodes = list(set(edge_index[:, edge_label.bool()].reshape(-1).tolist()))
            if y.item() == 0:
                node_label[signal_nodes] = 1

            if len(signal_nodes) != 0:
                node_type = torch.tensor(node_type_lists[i])
                node_type = set(node_type[signal_nodes].tolist())
                assert node_type in ({4, 1}, {4, 3}, {4, 1, 3})  # NO or NH

            if y.item() == 0 and len(signal_nodes) == 0:
                continue
            
            if i<5:
                print("\U0001F96D node_label.shape:", node_label.shape)
                try:
                    edge_att = torch.ones(edge_index.shape[1])
                    fig, ax = plt.subplots()
                    fig.canvas.manager.set_window_title(f"Graph {i}")
                    visualize_a_graph(edge_index, edge_att, node_label, 'mutag', ax, coor=None, norm=False, mol_type=None, nodesize=300)
                    
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    
                    plt.show()
                    plt.close(fig)
                except Exception as e:
                    print(f"Error in graph {i}: {e}")
            data_list.append(Data(x=x, y=y, edge_index=edge_index, node_label=node_label, edge_label=edge_label, node_type=torch.tensor(node_type_lists[i])))
            

        edge_att.shape == (edge_index.shape[1],)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_graph_data(self):
        pri = self.raw_dir + '/Mutagenicity_'

        file_edges = pri + 'A.txt'
        # file_edge_labels = pri + 'edge_labels.txt'
        file_edge_labels = pri + 'edge_gt.txt'
        file_graph_indicator = pri + 'graph_indicator.txt'
        file_graph_labels = pri + 'graph_labels.txt'
        file_node_labels = pri + 'node_labels.txt'

        edges = np.loadtxt(file_edges, delimiter=',').astype(np.int32)
        try:
            edge_labels = np.loadtxt(file_edge_labels, delimiter=',').astype(np.int32)
        except Exception as e:
            print(e)
            print('use edge label 0')
            edge_labels = np.zeros(edges.shape[0]).astype(np.int32)

        graph_indicator = np.loadtxt(file_graph_indicator, delimiter=',').astype(np.int32)
        graph_labels = np.loadtxt(file_graph_labels, delimiter=',').astype(np.int32)

        try:
            node_labels = np.loadtxt(file_node_labels, delimiter=',').astype(np.int32)
        except Exception as e:
            print(e)
            print('use node label 0')
            node_labels = np.zeros(graph_indicator.shape[0]).astype(np.int32)

        graph_id = 1
        starts = [1]
        node2graph = {}
        for i in range(len(graph_indicator)):
            if graph_indicator[i] != graph_id:
                graph_id = graph_indicator[i]
                starts.append(i+1)
            node2graph[i+1] = len(starts)-1
        # print(starts)
        # print(node2graph)
        graphid = 0
        edge_lists = []
        edge_label_lists = []
        edge_list = []
        edge_label_list = []
        for (s, t), l in list(zip(edges, edge_labels)):
            sgid = node2graph[s]
            tgid = node2graph[t]
            if sgid != tgid:
                print('edges connecting different graphs, error here, please check.')
                print(s, t, 'graph id', sgid, tgid)
                exit(1)
            gid = sgid
            if gid != graphid:
                edge_lists.append(edge_list)
                edge_label_lists.append(edge_label_list)
                edge_list = []
                edge_label_list = []
                graphid = gid
            start = starts[gid]
            edge_list.append((s-start, t-start))
            edge_label_list.append(l)

        edge_lists.append(edge_list)
        edge_label_lists.append(edge_label_list)

        # node labels
        node_label_lists = []
        graphid = 0
        node_label_list = []
        for i in range(len(node_labels)):
            nid = i+1
            gid = node2graph[nid]
            # start = starts[gid]
            if gid != graphid:
                node_label_lists.append(node_label_list)
                graphid = gid
                node_label_list = []
            node_label_list.append(node_labels[i])
        node_label_lists.append(node_label_list)

        return edge_lists, graph_labels, edge_label_lists, node_label_lists
    

# # https://github.com/flyingdoog/PGExplainer/blob/master/MUTAG.ipynb

# import yaml
# import torch
# import numpy as np
# import pickle as pkl
# from pathlib import Path
# from torch_geometric.data import InMemoryDataset, Data


# class Mutag(InMemoryDataset):
#     def __init__(self, root):
#         super().__init__(root=root)
#         self.process() #MANGO
#         self.data, self.slices = torch.load(self.processed_paths[0])

#     @property
#     def raw_file_names(self):
#         return ['Mutagenicity_A.txt', 'Mutagenicity_edge_gt.txt', 'Mutagenicity_edge_labels.txt',
#                 'Mutagenicity_graph_indicator.txt', 'Mutagenicity_graph_labels.txt', 'Mutagenicity_label_readme.txt',
#                 'Mutagenicity_node_labels.txt', 'Mutagenicity.pkl']

#     @property
#     def processed_file_names(self):
#         return ['data.pt'] #MANGO

#     def download(self):
#         raise NotImplementedError

#     def process(self):
#         with open(self.raw_dir + '/Mutagenicity.pkl', 'rb') as fin:
#             _, original_features, original_labels = pkl.load(fin)

#         dual_edge_lists, graph_labels, dual_node_gt_lists, dual_node_label_lists, dual_node_lists = self.get_graph_data()

#         data_list = []

#         #print("\U0001F96D original_features.shape:", original_features.shape) #MANGO (4337, 418, 14)
#         #print("\U0001F96D dual_edge_lists:", dual_edge_lists) #MANGO, len: 8674, 
#         #print("\U0001F96D original_features.shape:", original_features.shape)

#         max_len = max(len(sublist) for sublist in dual_edge_lists)
#         #print("\U0001F96D max_len:", max_len) #MANGO 

#         dual_features = np.zeros((4337, 203, 28))

#         for gid in range(dual_features.shape[0]):
#             dual_node_list = dual_node_lists[gid]
#             idx = 0
#             for (s,t) in dual_node_list:
#                 if len(dual_node_list) == 0:
#                     print(f"\U0001F96D Skipping gid={gid} due to empty dual_node_list")
#                     continue
#                 #print("\U0001F96D s:",s)
#                 #print("\U0001F96D type(s):",type(s))
#                 assert np.isscalar(s), f"s is not scalar: {s} (type={type(s)})"
#                 assert np.isscalar(t), f"t is not scalar: {t} (type={type(t)})"
#                 f1 = original_features[gid, s, :]
#                 f2 = original_features[gid, t, :]
#                 cat = np.concatenate((f1, f2))
#                 assert cat.shape == (28,), f"Concatenated shape is wrong: {cat.shape}"
#                 dual_features[gid,idx,:] = cat

#         count = 0
#         for i in range(dual_features.shape[0]):
#             num_nodes = len(dual_node_lists[i]) # MANGO
            
#             #print("\U0001F96D i:", i)
#             #print("\U0001F96D num_nodes:", num_nodes)

#             #edge_index = torch.tensor(dual_edge_lists[i], dtype=torch.long).T # MANGO, didnt work bc 2, 2, x dimension


#             edges_np = np.array(dual_edge_lists[i]) 
#             edge_index = torch.tensor(edges_np, dtype=torch.long).T
#             edge_index = edge_index.reshape(2, -1)


#             #print("\U0001F96D edge_index.shape:", edge_index.shape)
#             #print("\U0001F96D edge_index:", edge_index)

#             dual_nodes_index = torch.tensor(dual_node_lists[i], dtype=torch.long).T

#             y = torch.tensor(graph_labels[i]).float().reshape(-1, 1)
#             x = torch.tensor(dual_features[i][:num_nodes]).float() # MANGO removes padding, MANGO PIT
#             #print("\U0001F96D padding:", original_features[i][num_nodes:])
#             #print("\U0001F96D padding sum:", original_features[i][num_nodes:].sum())
#             assert dual_features[i][num_nodes:].sum() == 0
#             #if (original_features[i][num_nodes:].sum() != 0):
#             #    count+=1

#             # MANGO
#             dual_node_gt = torch.tensor(dual_node_gt_lists[i]).float() # MANGO 
#             #if y.item() != 0: # MANGO
#             #    edge_label = torch.zeros_like(edge_label).float() # MANGO zero out edge labels, RAW MANGO

#             node_label = torch.zeros(x.shape[0]) # MANGO copies shape of node features and sets to zero, RAW MANGO
#             #signal_nodes = list(set(dual_nodes_index[:, dual_node_signal.bool()].reshape(-1).tolist())) # MANGO picks out signal nodes from the edge label (for explainer; ground truth nodes), ONLY THE RIPE MANGO
#             #print("\U0001F96D dual_nodes_index:", dual_nodes_index)
#             #print("\U0001F96D edge_index:", edge_index)
            
#             #signal_nodes = list(set(dual_nodes_index[dual_node_gt.astype(bool)].tolist()))
#             signal_nodes = list(set([x - 1 for x in dual_nodes_index[:, dual_node_gt.bool()].reshape(-1).tolist()]))
#             #signal_nodes = list(dual_nodes[dual_node_signal])

#             #print("\U0001F96D dual_gt_list:", dual_node_gt_lists) #MANGO

#             if y.item() == 0:
#                 #print("\U0001F96D signal_nodes:", signal_nodes)
#                 node_label[signal_nodes] = 1 #set signal nodes' label to one if graph label is 0, ONLY THE RIPE MANGO
#                 #print("\U0001F96D signal_nodes:", signal_nodes)

#             # if len(signal_nodes) != 0: #MANGO IDK HERE DOES IT MAKE SENSE TO TAKE THIS OUT cause {4,1} {4,3} etc doesn't apply?*, ROTTEN MANGO?
#             #     node_type = torch.tensor(dual_node_label_lists[i]) # MANGO
#             #     node_type = set(node_type[signal_nodes].tolist())
#             #     assert node_type in ({4, 1}, {4, 3}, {4, 1, 3})  # MANGO NO or NH; Checks that signal nodes are the right node type

#             # if y.item() == 0 and len(signal_nodes) == 0: #SKIP THIS 
#             #     continue

#             edge_label = torch.zeros(edge_index.size(1), dtype=torch.float) # MANGO 
#             if x.shape[0] == 0 or edge_index.shape[1] == 0:
#                 print(f"\U0001F96D Skipping empty graph {i}")
#                 continue
                
#             node_type = torch.tensor(dual_features[i][:num_nodes]).float()
#             #print(f" Total graphs processed: {len(data_list)}")
#             print("\U0001F96D x.shape: ", x.shape) #MANGO
#             print("\U0001F96D y.shape: ", y.shape) #MANGO
#             print("\U0001F96D edge_index.shape:", edge_index.shape) #MANGO
#             print("\U0001F96D node_label.shape:", node_label.shape) #MANGO
#             print("\U0001F96D edge_label.shape:", edge_label.shape) #MANGO
#             print("\U0001F96D node_type.shape:", node_type.shape) #MANGO

#             if i==2:
#                 print("\U0001F96D node_label.shape:", node_label.shape)
#                 edge_att = torch.ones(edge_index.shape[1])
#                 fig, ax = plt.subplots()
#                 visualize_a_graph(edge_index, edge_att, node_label, 'mutag', ax, coor=None, norm=False, mol_type=None, nodesize=300)
#                 plt.show()

#             data_list.append(Data(x=x, y=y, edge_index=edge_index, node_label=node_label, edge_label=edge_label, node_type = node_type)) # MANGO , nodetype?

#         print(f" Total graphs processed final: {len(data_list)}")
#         print("\U0001F96D count:", count)
#         data, slices = self.collate(data_list)
#         torch.save((data, slices), self.processed_paths[0])

#     def get_graph_data(self):
#         pri = self.raw_dir + '/Mutagenicity_'

#         file_edges = pri + 'A.txt' 
#         file_edge_labels = pri + 'edge_labels.txt'
#         file_edge_gt = pri + 'edge_gt.txt'
#         file_graph_indicator = pri + 'graph_indicator.txt'
#         file_graph_labels = pri + 'graph_labels.txt'
#         file_node_labels = pri + 'node_labels.txt'

#         dual_nodes = np.loadtxt(file_edges, delimiter=',').astype(np.int32) # MANGO Loading dual_nodes in adjacency matrix form
#         try:
#             dual_node_labels = np.loadtxt(file_edge_labels, delimiter=',').astype(np.int32)
#         except Exception as e:
#             print(e)
#             print('use edge label 0')
#             dual_node_labels = np.zeros(dual_nodes.shape[0]).astype(np.int32) # MANGO

#         try: # MANGO
#             dual_node_signal = np.loadtxt(file_edge_gt, delimiter=',').astype(np.int32) #MANGO
#         except Exception as e: # MANGO
#             print(e) # MANGO
#             print('use edge gt 0') # MANGO
#             dual_node_signal = np.zeros(dual_nodes.shape[0]).astype(np.int32) # MANGO


#         graph_indicator = np.loadtxt(file_graph_indicator, delimiter=',').astype(np.int32)
#         graph_labels = np.loadtxt(file_graph_labels, delimiter=',').astype(np.int32)

#         dual_graph_indicator = np.zeros((dual_nodes.shape[0], 1)) #MANGO, 
#         for i in (range(dual_nodes.shape[0])): #MANGO
#             assert graph_indicator[(dual_nodes[i][0]-1)] == graph_indicator[(dual_nodes[i][1]-1)]
#             dual_graph_indicator[i] = graph_indicator[(dual_nodes[i][0])-1] #MANGO
#         dual_graph_indicator = dual_graph_indicator.astype(int) # MANGO

#         try:
#             primal_node_labels = np.loadtxt(file_node_labels, delimiter=',').astype(np.int32) # MANGO should these be included in the edge features? MULTIPLE MANGO
#         except Exception as e:
#             print(e)
#             print('use node label 0')
#             primal_node_labels = np.zeros(graph_indicator.shape[0]).astype(np.int32) # MANGO

#         # ATP loaded all files whole MANGO has been eaten

#         dual_edges = [] # MANGO

#         #CHATGPT ALERT ALERT ALERT MANGOS FALLING
#         from collections import defaultdict

#         dual_nodes = np.array(dual_nodes)  # Just in case

#         group_by_first = defaultdict(list)
#         group_by_second = defaultdict(list)

#         for idx, (a, b) in enumerate(dual_nodes):
#             group_by_first[a].append(idx)
#             group_by_second[b].append(idx)

#         dual_edges = []

#         def add_pairs_from_group(group):
#             indices = group
#             length = len(indices)
#             for i in range(length):
#                 for j in range(i + 1, length):
#                     e1 = dual_nodes[indices[i]]
#                     e2 = dual_nodes[indices[j]]
#                     #print("\U0001F96D ENDDDDDD1:", e1[0], e2[0], e1[1], e2[1])
#                     dual_edges.append([e1.tolist(), e2.tolist()])

#         # Add pairs grouped by first element
#         for group in group_by_first.values():
#             if len(group) > 1:
#                 add_pairs_from_group(group)

#         # Add pairs grouped by second element
#         for group in group_by_second.values():
#             if len(group) > 1:
#                 add_pairs_from_group(group)

#         # CHATGPT END THE MANGOS HAVE LANDED

#         dual_edges = np.array(dual_edges) # MANGO

#         #print("\U0001F96D dual_edges:", dual_edges) #MANGO, looks good ATP
#         #print("\U0001F96D len(dual_edges):", len(dual_edges)) #MANGO, looks good ATP, 451808

#         graph_id = 1
#         starts = [1]
#         node2graph = {}
#         #print("\U0001F96D dual_node_signal:", dual_graph_indicator) #MANGO
#         for i in range(len(dual_graph_indicator)): #MANGO
#             if dual_graph_indicator[i] != graph_id: #MANGO 
#                 graph_id = dual_graph_indicator[i] #MANGO
#                 starts.append(i) #MANGO
#             #node2graph[tuple(dual_nodes[i])] = len(starts)-1 #MANGO

#         #Need separate primal starts because normal starts doesn't line up right because of the edges not being numbered 1,2,3 ...
#         graph_id = 1
#         primal_starts = [1]
#         for i in range(len(graph_indicator)): #MANGO
#             if graph_indicator[i] != graph_id: #MANGO 
#                 graph_id = graph_indicator[i] #MANGO
#                 primal_starts.append(i) #MANGO
#             node2graph[i+1] = len(primal_starts)-1 #MANGO, no minus one needed because node2graph only used for comparing nodes against each other
#             #print("\U0001F96D len(primal_starts):", len(primal_starts)) #MANGO, looks good ATP, 451808
#         #Looks all good atp

#         #print("\U0001F96D dual graph indicator:", dual_graph_indicator) #MANGO

#         # print(starts)
#         # print(node2graph)
#         graphid = 0
#         dual_edge_lists = [] # MANGO
#         #edge_label_lists = [] # MANGO
#         dual_edge_list = [] # MANGO
#         #edge_label_list = [] # MANGO
#         for (s, t) in list(dual_edges): #MANGO removed edge labels because dual edges have no gt, SEEDLESS MANGO 
#             sgid = node2graph[s[0]] # MANGO, graph id of start node
#             tgid = node2graph[t[0]] # MANGO, graph id of end node
#             if sgid != tgid:
#                 print('edges connecting different graphs, error here, please check.')
#                 print(s, t, 'graph id', sgid, tgid)
#                 exit(1)
#             gid = sgid
#             if gid != graphid: #create new sublist for next graph
#                 #print("\U0001F96D dual_edge_list:", dual_edge_list) #MANGO
#                 dual_edge_lists.append(dual_edge_list) # MANGO
#                 #edge_label_lists.append(edge_label_list) # MANGO
#                 dual_edge_list = []
#                 #edge_label_list = [] # MANGO
#                 graphid = gid
#             start = primal_starts[gid] # MANGO 
#             #print("\U0001F96D start:", start) #MANGO
#             #print("\U0001F96D s:", s) #MANGO
#             #print("\U0001F96D t:", t) #MANGO
#             dual_edge_list.append((s-start, t-start))
#             #edge_label_list.append(l) # MANGO

#         dual_edge_lists.append(dual_edge_list)
#         #edge_label_lists.append(edge_label_list) # MANGO

#         #CHATGPT ALERT ALERT ALERT MANGOS FALLING

#         dual_node_gt_lists = []
#         dual_node_gt_list = []
#         graphid = 0

#         #print("\U0001F96D dual_node_signal:", dual_node_signal) #MANGO


#         for idx, label in enumerate(dual_node_signal):
#             gid = node2graph[dual_nodes[idx][0]]  # graph ID for this node
    
#             if gid != graphid:
#                 dual_node_gt_lists.append(dual_node_gt_list)
#                 dual_node_gt_list = []
#                 graphid = gid

#             dual_node_gt_list.append(label)

#         dual_node_gt_lists.append(dual_node_gt_list)

#         #CHATGPT END THE MANGOS HAVE LANDED 

        
#         graphid = 0
        

#         # for i in range(len(dual_nodes)):
#         #     gid = node2graph[tuple(dual_nodes[i])]
    
#         #     if gid != graphid:
#         #         dual_nodes_lists.append(dual_node_list)
#         #         dual_node_list = []
#         #         graphid = gid

#         #     start = starts[gid]
#         #     dual_node_list.append(tuple(dual_nodes[i] - start))  # or just dual_nodes[i] if raw
    
#         # dual_nodes_lists.append(dual_node_list)

#         # node labels


#         # print("\U0001F96D dual_nodes size:", len(dual_nodes)) #MANGO
#         dual_node_lists = []
#         dual_node_label_lists = []
#         graphid = 0
#         dual_node_label_list = []
#         dual_node_list = []
#         for i in range(len(dual_node_labels)):
#             # nid = i+1 # MANGO
#             gid = node2graph[dual_nodes[i][0]] # MANGO
#             # start = starts[gid]
#             if gid != graphid:
#                 dual_node_label_lists.append(dual_node_label_list)
#                 dual_node_lists.append(dual_node_list)
#                 graphid = gid
#                 dual_node_label_list = []
#                 dual_node_list = []
#             u, v = dual_nodes[i]
#             gu = graph_indicator[u-1]
#             gv = graph_indicator[v-1]
#             assert gu == gv

#             start = primal_starts[gid]
#             #print("\U0001F96D start:", start) #MANGO
#             #print("\U0001F96D dual_nodes[i]:", dual_nodes[i])
#             dual_node_label_list.append(dual_node_labels[i])
#             dual_node_list.append((dual_nodes[i][0]-start, dual_nodes[i][1]-start))
#         dual_node_label_lists.append(dual_node_label_list)
#         dual_node_lists.append(dual_node_list)

#         # print("\U0001F96D dual_node_label_lists size:", len(dual_node_label_lists)) #MANGO
#         # print("\U0001F96D dual_node_lists size:", len(dual_node_lists)) #MANGO

#         return dual_edge_lists, graph_labels, dual_node_gt_lists, dual_node_label_lists, dual_node_lists # MANGO
    
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
        if len(G.nodes) > 1:
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G)
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

