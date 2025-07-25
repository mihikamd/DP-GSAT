
# Trial of primal and dual together in on class

# https://github.com/flyingdoog/PGExplainer/blob/master/MUTAG.ipynb

import yaml
import torch
import numpy as np
import pickle as pkl
from pathlib import Path
from torch_geometric.data import InMemoryDataset, Data

import matplotlib.pyplot as plt 
import networkx as nx 
from torch_geometric.utils import to_networkx
from rdkit import Chem 
import os

class Mutag(InMemoryDataset):
    def __init__(self, root):
        super().__init__(root=root)
        self.process()
        self.data, self.slices = torch.load(self.processed_paths[0])

    # @property
    # def processed_dir(self):
    #     base = super().processed_dir
    #     return os.path.join(base, 'dual') if self.dual else os.path.join(base, 'primal')


    @property
    def raw_file_names(self):
        return ['Mutagenicity_A.txt', 'Mutagenicity_edge_gt.txt', 'Mutagenicity_edge_labels.txt',
                'Mutagenicity_graph_indicator.txt', 'Mutagenicity_graph_labels.txt', 'Mutagenicity_label_readme.txt',
                'Mutagenicity_node_labels.txt', 'Mutagenicity.pkl']

    @property
    def processed_file_names(self):
        return ['primal_data.pt', 'dual_data.pt']

    def download(self):
        raise NotImplementedError

    def process(self):
        with open(self.raw_dir + '/Mutagenicity.pkl', 'rb') as fin:
            _, original_features, original_labels = pkl.load(fin)

        dual_edge_lists, graph_labels, dual_node_gt_lists, dual_node_label_lists, dual_node_lists = self.get_dualgraph_data()

        data_list = []

        dual_features = np.zeros((4337, 203, 28))

        print("\U0001F96D KACHA MANGO dual_node_lists:", dual_node_lists) #MANGO
        print("\U0001F96D KACHA MANGO len dual_node_lists:", len(dual_node_lists)) #MANGO

        for gid in range(dual_features.shape[0]):
            dual_node_list = dual_node_lists[gid]
            idx = 0
            for (s,t) in dual_node_list:
                assert len(dual_node_list) != 0
                assert np.isscalar(s), f"s is not scalar: {s} (type={type(s)})"
                assert np.isscalar(t), f"t is not scalar: {t} (type={type(t)})"
                f1 = original_features[gid, s, :]
                f2 = original_features[gid, t, :]
                cat = np.concatenate((f1, f2))
                assert cat.shape == (28,), f"Concatenated shape is wrong: {cat.shape}"
                dual_features[gid,idx,:] = cat

        count = 0
        for i in range(dual_features.shape[0]):
            num_nodes = len(dual_node_lists[i]) # MANGO
            print("num_nodes")
            edge_index = torch.tensor(dual_edge_lists[i], dtype=torch.long).T # MANGO, didnt work bc 2, 2, x dimension


            edges_np = np.array(dual_edge_lists[i])
            edges_np -= 1 
            edge_index = torch.tensor(edges_np, dtype=torch.long).T

            #dual_nodes_index = torch.tensor(edges_np, dtype=torch.long).T
            edge_index = edge_index.reshape(2, -1)

            y = torch.tensor(graph_labels[i]).float().reshape(-1, 1)
            x = torch.tensor(dual_features[i][:num_nodes]).float()
            assert dual_features[i][num_nodes:].sum() == 0
            #if (original_features[i][num_nodes:].sum() != 0):
            #    count+=1

            dual_node_gt = torch.tensor(dual_node_gt_lists[i]).float() # MANGO 
            #if y.item() != 0: # MANGO
            #    edge_label = torch.zeros_like(edge_label).float() # MANGO zero out edge labels, RAW MANGO

            node_label = torch.zeros(x.shape[0])
            #signal_nodes = list(set(dual_nodes_index[:, dual_node_signal.bool()].reshape(-1).tolist())) # MANGO picks out signal nodes from the edge label (for explainer; ground truth nodes), ONLY THE RIPE MANGO
            
            #signal_nodes = list(set(dual_nodes_index[dual_node_gt.astype(bool)].tolist()))
            #signal_nodes = list(set([x-1 for x in dual_nodes_index[:, dual_node_gt.bool()].reshape(-1).tolist()]))
            #signal_nodes = list(dual_nodes[dual_node_signal])

            #if y.item() == 0:
            #    node_label[signal_nodes] = 1

            # if len(signal_nodes) != 0: #MANGO IDK HERE DOES IT MAKE SENSE TO TAKE THIS OUT cause {4,1} {4,3} etc doesn't apply?*, ROTTEN MANGO?
            #     node_type = torch.tensor(dual_node_label_lists[i]) # MANGO
            #     node_type = set(node_type[signal_nodes].tolist())
            #     assert node_type in ({4, 1}, {4, 3}, {4, 1, 3})  # MANGO NO or NH; Checks that signal nodes are the right node type

            # if y.item() == 0 and len(signal_nodes) == 0: #SKIP THIS 
            #     continue

            edge_label = torch.zeros(edge_index.size(1), dtype=torch.float)
            if x.shape[0] == 0 or edge_index.shape[1] == 0:
                print(f"\U0001F96D Skipping empty graph {i}")
                continue
                
            node_type = torch.tensor(dual_features[i][:num_nodes]).float()
            print("\U0001F96D x.shape: ", x.shape)
            print("\U0001F96D y.shape: ", y.shape)
            print("\U0001F96D edge_index.shape:", edge_index.shape)
            print("\U0001F96D node_label.shape:", node_label.shape) 
            print("\U0001F96D edge_label.shape:", edge_label.shape)
            print("\U0001F96D node_type.shape:", node_type.shape)

            if i<5:
                print("\U0001F96D node_label.shape:", node_label.shape)
                print("\U0001F96D I:", i)
                #try:
                if i < 5:
                    # edge_index = torch.tensor([[0, 1, 0, 2, 1, 3],
                    #        [1, 0, 2, 0, 3, 1]], dtype=torch.long)
                    num_unique = torch.unique(edge_index).numel()
                    print("unique:", torch.unique(edge_index))
                    print ("\U0001F96D num_unique", num_unique)
                    edge_att = torch.ones(edge_index.shape[1])
                    # node_label = torch.zeros(4)
                    node_label = torch.zeros(num_unique)
                    fig, ax = plt.subplots()
                    fig.canvas.manager.set_window_title(f"Dual Graph v2 {i}")
                    print("\U0001F96D edge_index:", edge_index)
                    print("\U0001F96D edge_att:", edge_att)
                    print("\U0001F96D node_label:", node_label)
                    print("\U0001F96D node_label.size(0)", node_label.size(0))
                    
                    visualize_a_graph(edge_index, edge_att, node_label, 'mutag', ax, coor=None, norm=False, mol_type=None, nodesize=300)
                    
                    plt.show(block=True)
                    plt.close(fig)

                #except Exception as e:
                #    print(f"Error in graph {i}: {e}")

            data_list.append(Data(x=x, y=y, edge_index=edge_index, node_label=node_label, edge_label=edge_label, node_type = node_type))

        # primal_edges = primal_edges - 1

        # print("\U0001F96D edges_np.shape:", edges_np.shape)
        # print("\U0001F96D dual_edge_lists[0]:", dual_edge_lists[0])
        # #edge_index = torch.tensor(edges_np, dtype=torch.long).T

        # num_unique = torch.unique(edge_index).numel()

        # edge_att = torch.ones(edge_index.shape[1])
        # #node_label = torch.zeros(int(dual_node_lists.shape[0]) // 2 + 1)
        # node_label = torch.zeros(num_unique)
        # fig, ax = plt.subplots()
        # fig.canvas.manager.set_window_title("Dual Graph test")
        # print("\U0001F96D edge_index:", edge_index)
        # print("\U0001F96D edge_att:", edge_att)
        # print("\U0001F96D node_label:", node_label)
        # print("\U0001F96D edge_index:", edge_index.max().item())

        # visualize_a_graph(edge_index, edge_att, node_label, 'mutag', ax, coor=None, norm=False, mol_type=None, nodesize=300)
                    
        # plt.show()
        # plt.close(fig)

        # print(f" Total graphs processed final: {len(data_list)}")
        # print("\U0001F96D count:", count)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[1])

        #STARTING PRIMAL

        edge_lists, graph_labels, edge_label_lists, node_type_lists = self.get_primalgraph_data()

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

            data_list.append(Data(x=x, y=y, edge_index=edge_index, node_label=node_label, edge_label=edge_label, node_type=torch.tensor(node_type_lists[i])))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


    def get_dualgraph_data(self):
        pri = self.raw_dir + '/Mutagenicity_'

        file_edges = pri + 'A.txt' 
        file_edge_labels = pri + 'edge_labels.txt'
        file_edge_gt = pri + 'edge_gt.txt'
        file_graph_indicator = pri + 'graph_indicator.txt'
        file_graph_labels = pri + 'graph_labels.txt'
        file_node_labels = pri + 'node_labels.txt'

        dual_nodes = np.loadtxt(file_edges, delimiter=',').astype(np.int32)

        print("\U0001F96D dual_nodes:", dual_nodes)
        print("\U0001F96D dual_nodes.shape:", dual_nodes.shape) # (266894,2)
        #input("continue dual_nodes") # looks good 

        dual_nodes = dual_nodes - 1

        # print("\U0001F96D next dual_nodes.shape:", dual_nodes.shape) 
        # print("\U0001F96D next dual_nodes:", dual_nodes) 

        dual_dict = {tuple(pair): (idx // 2) + 1 for idx, pair in enumerate(dual_nodes)} # for each edge pair duplicate they share one index
        #dual_dict = {tuple(pair): idx + 1 for idx, pair in enumerate(dual_nodes)} 
        print("\U0001F96D next dual_dict:", dual_dict) 
        print("\U0001F96D len(next dual_dict):", len(dual_dict)) # 266894
        #input("continue dual_dict") #looks good

        try:
            dual_node_labels = np.loadtxt(file_edge_labels, delimiter=',').astype(np.int32)
            #dual_node_labels = np.zeros(dual_nodes.shape[0]).astype(np.int32) # DOUBLE MANGO
        except Exception as e:
            print(e)
            print('use edge label 0')
            dual_node_labels = np.zeros(dual_nodes.shape[0]).astype(np.int32) 

        try: 
            dual_node_signal = np.loadtxt(file_edge_gt, delimiter=',').astype(np.int32)
            #dual_node_signal = np.zeros(dual_nodes.shape[0]).astype(np.int32) # DOUBLE MANGO
        except Exception as e: 
            print(e) #
            print('use edge gt 0') 
            dual_node_signal = np.zeros(dual_nodes.shape[0]).astype(np.int32) 

        graph_indicator = np.loadtxt(file_graph_indicator, delimiter=',').astype(np.int32)

        print("\U0001F96D graph_indicator:", graph_indicator)
        print("\U0001F96D graph_indicator.shape:", graph_indicator.shape) #(131488,)
        # input("continue graph_indicator") # looks good

        graph_labels = np.loadtxt(file_graph_labels, delimiter=',').astype(np.int32)

        print("\U0001F96D graph_labels:", graph_labels)
        print("\U0001F96D graph_labels.shape:", graph_labels.shape) # (4337,)
        # input("continue graph_labels") # looks good

        dual_graph_indicator = np.zeros((dual_nodes.shape[0]))
        for i in (range(dual_nodes.shape[0])):
            assert graph_indicator[(dual_nodes[i][0])] == graph_indicator[(dual_nodes[i][1])]
            dual_graph_indicator[i] = graph_indicator[(dual_nodes[i][0])]
        dual_graph_indicator = dual_graph_indicator.astype(int) 
 
        print("\U0001F96D next dual_graph_indicator:", dual_graph_indicator)
        print("\U0001F96D next dual_graph_indicator.shape:", dual_graph_indicator.shape) # (266894,)
        # input("continue dual_graph_indicator") # looks good but checking more

        for i in (range(dual_nodes.shape[0])):
            print("\U0001F96D next dual_graph_indicator, dual node:", dual_graph_indicator[i], ", ", dual_nodes[i])
        # input("continue dual_graph_indicator, dual_nodes") # checked first graph, looks good

        try:
            primal_node_labels = np.loadtxt(file_node_labels, delimiter=',').astype(np.int32) 
        except Exception as e:
            print(e)
            print('use node label 0')
            primal_node_labels = np.zeros(graph_indicator.shape[0]).astype(np.int32)

        dual_edges = []

        #CHATGPT ALERT ALERT ALERT MANGOS FALLING
        from collections import defaultdict

        dual_nodes = np.array(dual_nodes)

        group_by_first = defaultdict(list)
        group_by_second = defaultdict(list)

        for idx, (a, b) in enumerate(dual_nodes):
            group_by_first[a].append(idx)
            group_by_second[b].append(idx)

        print("\U0001F96D group_by_first:", group_by_first)
        # input("continue group_by_first") # looks good, basically group_by_first[1] is the idx of all of the nodes that start with node 1

        print("\U0001F96D group_by_second:", group_by_second)
        # input("continue group_by_second") # looks good, basically group_by_second[1] is the idx of all of the nodes that start with node 1

        dual_edges = []

        def add_pairs_from_group(group):
            indices = group
            length = len(indices)
            for i in range(length):
                for j in range(i + 1, length):
                    e1 = dual_nodes[indices[i]]
                    e2 = dual_nodes[indices[j]]
                    #e1 = indices[i]
                    #e2 = indices[j]
                    dual_edges.append([e1, e2])
                    dual_edges.append([e2, e1])

        for group in group_by_first.values():
            if len(group) > 1:
                add_pairs_from_group(group)

        # for group in group_by_second.values():
        #     if len(group) > 1:
        #         add_pairs_from_group(group)

        print("\U0001F96D dual_edges:", dual_edges)
        print("\U0001F96D len dual_edges:", len(dual_edges)) # 451808
        # input("continue dual_edges") # looks good, all in the right order
        # CHATGPT END THE MANGOS HAVE LANDED

        dual_edges = np.array(dual_edges) # MANGO

        graph_id = 1
        starts = [0]
        node2graph = {}
        for i in range(len(dual_graph_indicator)): 
            if dual_graph_indicator[i] != graph_id: 
                graph_id = dual_graph_indicator[i] 
                starts.append(i) #MANGO
            #node2graph[tuple(dual_nodes[i])] = len(starts)-1 #MANGO

        graph_id = 1
        primal_starts = [0]
        for i in range(len(graph_indicator)): 
            if graph_indicator[i] != graph_id: 
                graph_id = graph_indicator[i] 
                primal_starts.append(i) 
            node2graph[i] = len(primal_starts)-1 

        
        graphid = 0

        dual_node_lists = []
        dual_node_label_lists = []
        graphid = 0
        dual_node_label_list = []
        dual_node_list = []
        for i in range(len(dual_node_labels)):
            # nid = i+1 # MANGO
            gid = node2graph[dual_nodes[i][0]] # MANGO
            # start = starts[gid]
            if gid != graphid:
                dual_node_label_lists.append(dual_node_label_list)
                dual_node_lists.append(dual_node_list)
                graphid = gid
                dual_node_label_list = []
                dual_node_list = []
            u, v = dual_nodes[i]
            gu = graph_indicator[u]
            gv = graph_indicator[v]
            assert gu == gv

            start = primal_starts[gid]
            dual_node_label_list.append(dual_node_labels[i])
            dual_node_list.append((dual_nodes[i][0]-start, dual_nodes[i][1]-start))

            # print("\U0001F96D dual_nodes_list:", dual_node_list) #MANGO
            # print("\U0001F96D len(dual_node_list):", len(dual_node_list)) #MANGO
            # input("continue dual_node_list")
        dual_node_label_lists.append(dual_node_label_list)
        dual_node_lists.append(dual_node_list)

        print("\U0001F96D dual_nodes_lists:", dual_node_lists) #MANGO
        print("\U0001F96D len(dual_node_lists):", len(dual_node_lists)) #MANGO, (4337,)
        # input("continue dual_node_lists") # Looking good, checked graph 1 and graph 2's dual_nodes and seemed to line up.

        graphid = 0
        dual_edge_lists = []
        #edge_label_lists = []
        dual_edge_list = []
        #edge_label_list = [] 
        for (s, t) in list(dual_edges):
            assert node2graph[s[0]] == node2graph[s[1]]
            assert node2graph[t[0]] == node2graph[t[1]]
            sgid = node2graph[s[0]] # MANGO, graph id of start node
            tgid = node2graph[t[0]] # MANGO, graph id of end node
            # print("\U0001F96D s:", s) 
            # print("\U0001F96D t:", t)
            # print("\U0001F96D dual_dict[tuple(s)]:", dual_dict[tuple(s)]) 
            # print("\U0001F96D dual_dict[tuple(t)]:", dual_dict[tuple(t)])
            #sgid = node2graph[s] # MANGO, graph id of start node
            #tgid = node2graph[t] # MANGO, graph id of end node
            if sgid != tgid:
                print('edges connecting different graphs, error here, please check.')
                print(s, t, 'graph id', sgid, tgid)
                exit(1)
            gid = sgid
            if gid != graphid: 
                dual_edge_lists.append(dual_edge_list) # MANGO
                #edge_label_lists.append(edge_label_list) # MANGO
                dual_edge_list = []
                #edge_label_list = [] # MANGO
                graphid = gid
            #start = primal_starts[gid] # MANGO  DOUBLE
            start = primal_starts[gid]
            #dual_edge_list.append((s-start, t-start))
            # print("\U0001F96D start:", start)

            #dual_edge_list.append((dual_dict[tuple(s)]-start, dual_dict[tuple(t)]-start))
            dual_edge_list.append((s-start, t-start))
            #edge_label_list.append(l) # MANGO
            # print("\U0001F96D dual_edge_list:", dual_edge_list) #MANGO
            # print("\U0001F96D len(dual_edge_list):", len(dual_edge_list)) #MANGO
            # input("continue dual_edge_list")

        dual_edge_lists.append(dual_edge_list)


        print("\U0001F96D KACHA MANGO dual_edge_lists:", dual_edge_lists) #MANGO
        print("\U0001F96D KACHA MANGO len dual_edge_lists:", len(dual_edge_lists)) #MANGO
        # input("continue dual_edge_lists") # should be ok?
        #edge_label_lists.append(edge_label_list) # MANGO
        
        #saving dual_nodes and dual_edges with dim 1 and 2 instead of 2 and 4

        dual_single_edge_lists = []
        dual_single_edge_list = []

        for i in range(len(dual_node_lists)):
            dual_dict = {tuple(pair): (idx // 2) + 1 for idx, pair in enumerate(dual_node_lists[i])} # for each edge pair duplicate they share one index
            for j in range(len(dual_edge_lists[i])):
                # print("dual_edge_lists[i][j].shape", dual_edge_lists[i][j].shape)
                # print("dual_edge_lists[i][j]", dual_edge_lists[i][j])
                dual_single_edge_list.append((dual_dict[tuple(dual_edge_lists[i][j][0])], dual_dict[tuple(dual_edge_lists[i][j][1])]))
                # print("\U0001F96D dual_single_edge_list:", dual_single_edge_list) #MANGO
                # print("\U0001F96D len dual_single_edge_list:", len(dual_single_edge_list)) #MANGO
                # input("continue dual_single_edge_list") 
            dual_single_edge_lists.append(dual_single_edge_list)
            dual_single_edge_list = []
            print("\U0001F96D stuck:", i)
        print("\U0001F96D done")
        print("\U0001F96D dual_single_edge_lists[0]:", dual_single_edge_lists[0]) #MANGO
        print("\U0001F96D len dual_single_edge_lists:", len(dual_single_edge_lists)) #MANGO
        input("continue dual_single_edge_lists") # should be ok?

        


        #CHATGPT ALERT ALERT ALERT MANGOS FALLING

        dual_node_gt_lists = []
        dual_node_gt_list = []
        graphid = 0


        for idx, label in enumerate(dual_node_signal):
            gid = node2graph[dual_nodes[idx][0]]  # graph ID for this node
    
            if gid != graphid:
                dual_node_gt_lists.append(dual_node_gt_list)
                dual_node_gt_list = []
                graphid = gid

            dual_node_gt_list.append(label)

        dual_node_gt_lists.append(dual_node_gt_list)

        #CHATGPT END THE MANGOS HAVE LANDED 
        
        return dual_single_edge_lists, graph_labels, dual_node_gt_lists, dual_node_label_lists, dual_node_lists # MANGO
    
    def get_primalgraph_data(self):
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

    #assert node_label.size(0) >= edge_index.max().item() + 1, "node_label too short"

    data = Data(edge_index=edge_index, att=edge_att, y=node_label, num_nodes=node_label.size(0)).to('cpu')
    G = to_networkx(data, node_attrs=['y'], edge_attrs=['att'])

    # from itertools import combinations

    # def repel_nodes(pos, min_dist=0.05):
    #     for i, j in combinations(pos, 2):
    #         pi, pj = pos[i], pos[j]
    #         dist = np.linalg.norm(pi - pj)
    #         if dist < min_dist:
    #             direction = (pi - pj) / (dist + 1e-6)
    #             shift = 0.5 * (min_dist - dist) * direction
    #             pos[i] += shift
    #             pos[j] -= shift
    #     return pos

    #pos = nx.kamada_kawai_layout(G)
    #pos = repel_nodes(pos, min_dist=0.1)

    # calculate Graph positions
    if coor is None:
        pos = nx.kamada_kawai_layout(G)
        #pos = nx.spring_layout(G, seed=42)
        # pos = repel_nodes(pos, min_dist=0.1)

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

    # Get all x and y positions from the layout
    x_values = [p[0] for p in pos.values()]
    y_values = [p[1] for p in pos.values()]

    # Compute padding and limits
    x_margin = (max(x_values) - min(x_values)) * 0.2 + 0.1
    y_margin = (max(y_values) - min(y_values)) * 0.2 + 0.1

    ax.set_xlim(min(x_values) - x_margin, max(x_values) + x_margin)
    ax.set_ylim(min(y_values) - y_margin, max(y_values) + y_margin)

    ax.set_aspect('equal') #this
    ax.axis('off') #this





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
#         self.data, self.slices = torch.load(self.processed_paths[0])

#     @property
#     def raw_file_names(self):
#         return ['Mutagenicity_A.txt', 'Mutagenicity_edge_gt.txt', 'Mutagenicity_edge_labels.txt',
#                 'Mutagenicity_graph_indicator.txt', 'Mutagenicity_graph_labels.txt', 'Mutagenicity_label_readme.txt',
#                 'Mutagenicity_node_labels.txt', 'Mutagenicity.pkl']

#     @property
#     def processed_file_names(self):
#         return ['data.pt']

#     def download(self):
#         raise NotImplementedError

#     def process(self):
#         with open(self.raw_dir + '/Mutagenicity.pkl', 'rb') as fin:
#             _, original_features, original_labels = pkl.load(fin)

#         edge_lists, graph_labels, edge_label_lists, node_type_lists = self.get_primalgraph_data()

#         data_list = []
#         for i in range(original_labels.shape[0]):
#             num_nodes = len(node_type_lists[i])
#             edge_index = torch.tensor(edge_lists[i], dtype=torch.long).T

#             y = torch.tensor(graph_labels[i]).float().reshape(-1, 1)
#             x = torch.tensor(original_features[i][:num_nodes]).float()
#             assert original_features[i][num_nodes:].sum() == 0
#             edge_label = torch.tensor(edge_label_lists[i]).float()
#             if y.item() != 0:
#                 edge_label = torch.zeros_like(edge_label).float()

#             node_label = torch.zeros(x.shape[0])
#             signal_nodes = list(set(edge_index[:, edge_label.bool()].reshape(-1).tolist()))
#             if y.item() == 0:
#                 node_label[signal_nodes] = 1

#             if len(signal_nodes) != 0:
#                 node_type = torch.tensor(node_type_lists[i])
#                 node_type = set(node_type[signal_nodes].tolist())
#                 assert node_type in ({4, 1}, {4, 3}, {4, 1, 3})  # NO or NH

#             if y.item() == 0 and len(signal_nodes) == 0:
#                 continue

#             data_list.append(Data(x=x, y=y, edge_index=edge_index, node_label=node_label, edge_label=edge_label, node_type=torch.tensor(node_type_lists[i])))

#         data, slices = self.collate(data_list)
#         torch.save((data, slices), self.processed_paths[0])

#     def get_primalgraph_data(self):
#         pri = self.raw_dir + '/Mutagenicity_'

#         file_edges = pri + 'A.txt'
#         # file_edge_labels = pri + 'edge_labels.txt'
#         file_edge_labels = pri + 'edge_gt.txt'
#         file_graph_indicator = pri + 'graph_indicator.txt'
#         file_graph_labels = pri + 'graph_labels.txt'
#         file_node_labels = pri + 'node_labels.txt'

#         edges = np.loadtxt(file_edges, delimiter=',').astype(np.int32)
#         try:
#             edge_labels = np.loadtxt(file_edge_labels, delimiter=',').astype(np.int32)
#         except Exception as e:
#             print(e)
#             print('use edge label 0')
#             edge_labels = np.zeros(edges.shape[0]).astype(np.int32)

#         graph_indicator = np.loadtxt(file_graph_indicator, delimiter=',').astype(np.int32)
#         graph_labels = np.loadtxt(file_graph_labels, delimiter=',').astype(np.int32)

#         try:
#             node_labels = np.loadtxt(file_node_labels, delimiter=',').astype(np.int32)
#         except Exception as e:
#             print(e)
#             print('use node label 0')
#             node_labels = np.zeros(graph_indicator.shape[0]).astype(np.int32)

#         graph_id = 1
#         starts = [1]
#         node2graph = {}
#         for i in range(len(graph_indicator)):
#             if graph_indicator[i] != graph_id:
#                 graph_id = graph_indicator[i]
#                 starts.append(i+1)
#             node2graph[i+1] = len(starts)-1
#         # print(starts)
#         # print(node2graph)
#         graphid = 0
#         edge_lists = []
#         edge_label_lists = []
#         edge_list = []
#         edge_label_list = []
#         for (s, t), l in list(zip(edges, edge_labels)):
#             sgid = node2graph[s]
#             tgid = node2graph[t]
#             if sgid != tgid:
#                 print('edges connecting different graphs, error here, please check.')
#                 print(s, t, 'graph id', sgid, tgid)
#                 exit(1)
#             gid = sgid
#             if gid != graphid:
#                 edge_lists.append(edge_list)
#                 edge_label_lists.append(edge_label_list)
#                 edge_list = []
#                 edge_label_list = []
#                 graphid = gid
#             start = starts[gid]
#             edge_list.append((s-start, t-start))
#             edge_label_list.append(l)

#         edge_lists.append(edge_list)
#         edge_label_lists.append(edge_label_list)

#         # node labels
#         node_label_lists = []
#         graphid = 0
#         node_label_list = []
#         for i in range(len(node_labels)):
#             nid = i+1
#             gid = node2graph[nid]
#             # start = starts[gid]
#             if gid != graphid:
#                 node_label_lists.append(node_label_list)
#                 graphid = gid
#                 node_label_list = []
#             node_label_list.append(node_labels[i])
#         node_label_lists.append(node_label_list)

#         return edge_lists, graph_labels, edge_label_lists, node_label_lists


# ========================================================


# print(dual_edges)
        # dual_edge_indicator = np.zeros((dual_edges.shape[0]))
        # for i in (range(dual_edges.shape[0])):
        #     print("dual_graph_indic: ", dual_graph_indicator[(dual_edges[i][0][0]-1)], ", ", dual_graph_indicator[(dual_edges[i][0][1]-1)], ", ", dual_graph_indicator[(dual_edges[i][1][0]-1)], ", ", dual_graph_indicator[(dual_edges[i][1][1]-1)])
        #     assert dual_graph_indicator[(dual_edges[i][0][0]-1)] == dual_graph_indicator[(dual_edges[i][0][1]-1)] == dual_graph_indicator[(dual_edges[i][1][0]-1)] == dual_graph_indicator[(dual_edges[i][1][1]-1)]
        #     dual_edge_indicator[i] = dual_graph_indicator[dual_edges[i][0][0]-1]
        # dual_graph_indicator = dual_graph_indicator.astype(int) 

        # dual_edge_indicator = np.array(dual_edge_indicator)
        # dual_edges = np.array(dual_edges)

        # # Get sorted indices
        # sorted_indices = np.argsort(dual_edge_indicator)

        # # Reorder both arrays
        # dual_edge_indicator = dual_edge_indicator[sorted_indices]
        # dual_edges = dual_edges[sorted_indices]

        # print("\U0001F96D dual_edge_indicator", dual_edge_indicator)
        # print("\U0001F96D dual_edges", dual_edges)
        # input("continue1")

        # graph_id = 1
        # edge_starts = [1]
        # for (s,t) in dual_edges: 
        #     if graph_indicator[s[0]] != graph_id: 
        #         graph_id = graph_indicator[s[0]] 
        #         edge_starts.append(dual_dict[tuple(s)]) 
        #     #node2graph[index] = len(primal_starts)-1 
        # graph_id = 1
        # edge_starts = [1]
        # for i in range(len(dual_edge_indicator)): 
        #     if dual_edge_indicator[i] != graph_id: 
        #         graph_id = dual_edge_indicator[i] 
        #         edge_starts.append(i) 
        #     #node2graph[i] = len(primal_starts)-1 

        # print('dual_edge_indicator', dual_edge_indicator)
        # input("continue")

        # print("edge_starts", edge_starts)
        # input("continue")


#=========================
# # https://github.com/flyingdoog/PGExplainer/blob/master/MUTAG.ipynb

# import yaml
# import torch
# import numpy as np
# import pickle as pkl
# from pathlib import Path
# from torch_geometric.data import InMemoryDataset, Data

import matplotlib.pyplot as plt #MANGO
import networkx as nx #MANGO
from torch_geometric.utils import to_networkx #MANGO
from rdkit import Chem #MANGO


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
#         return ['data.pt']

#     def download(self):
#         raise NotImplementedError

#     def process(self):
#         with open(self.raw_dir + '/Mutagenicity.pkl', 'rb') as fin:
#             _, original_features, original_labels = pkl.load(fin)

#         edge_lists, graph_labels, edge_label_lists, node_type_lists = self.get_graph_data()

#         data_list = []
#         for i in range(original_labels.shape[0]):
#             num_nodes = len(node_type_lists[i])
#             edge_index = torch.tensor(edge_lists[i], dtype=torch.long).T

#             y = torch.tensor(graph_labels[i]).float().reshape(-1, 1)
#             x = torch.tensor(original_features[i][:num_nodes]).float()
#             assert original_features[i][num_nodes:].sum() == 0
#             edge_label = torch.tensor(edge_label_lists[i]).float()
#             if y.item() != 0:
#                 edge_label = torch.zeros_like(edge_label).float()

#             node_label = torch.zeros(x.shape[0])
#             signal_nodes = list(set(edge_index[:, edge_label.bool()].reshape(-1).tolist()))
#             if y.item() == 0:
#                 node_label[signal_nodes] = 1

#             if len(signal_nodes) != 0:
#                 node_type = torch.tensor(node_type_lists[i])
#                 node_type = set(node_type[signal_nodes].tolist())
#                 assert node_type in ({4, 1}, {4, 3}, {4, 1, 3})  # NO or NH

#             if y.item() == 0 and len(signal_nodes) == 0:
#                 continue #so not all graphs are added
            
#             if i<5:
#                 print("\U0001F96D node_label.shape:", node_label.shape)
#                 print("\U0001F96D I:", i)
#                 try:
#                     edge_index = torch.tensor([[0, 1, 0, 2, 1, 3],
#                            [1, 0, 2, 0, 3, 1]], dtype=torch.long)
#                     edge_att = torch.ones(edge_index.shape[1])
#                     node_label = torch.zeros(4)
#                     fig, ax = plt.subplots()
#                     fig.canvas.manager.set_window_title(f"Primal Graph {i}")
#                     print("\U0001F96D edge_index:", edge_index)
#                     print("\U0001F96D edge_att:", edge_att)
#                     print("\U0001F96D node_label:", node_label)
                    
#                     visualize_a_graph(edge_index, edge_att, node_label, 'mutag', ax, coor=None, norm=False, mol_type=None, nodesize=300)
                    
#                     plt.show()
#                     plt.close(fig)
#                 except Exception as e:
#                     print(f"Error in graph {i}: {e}")
#             data_list.append(Data(x=x, y=y, edge_index=edge_index, node_label=node_label, edge_label=edge_label, node_type=torch.tensor(node_type_lists[i])))
            

#         #edge_att.shape == (edge_index.shape[1],)
#         data, slices = self.collate(data_list)
#         torch.save((data, slices), self.processed_paths[0])

#     def get_graph_data(self):
#         pri = self.raw_dir + '/Mutagenicity_'

#         file_edges = pri + 'A.txt'
#         # file_edge_labels = pri + 'edge_labels.txt'
#         file_edge_labels = pri + 'edge_gt.txt'
#         file_graph_indicator = pri + 'graph_indicator.txt'
#         file_graph_labels = pri + 'graph_labels.txt'
#         file_node_labels = pri + 'node_labels.txt'

#         edges = np.loadtxt(file_edges, delimiter=',').astype(np.int32)
#         try:
#             edge_labels = np.loadtxt(file_edge_labels, delimiter=',').astype(np.int32)
#         except Exception as e:
#             print(e)
#             print('use edge label 0')
#             edge_labels = np.zeros(edges.shape[0]).astype(np.int32)

#         graph_indicator = np.loadtxt(file_graph_indicator, delimiter=',').astype(np.int32)
#         graph_labels = np.loadtxt(file_graph_labels, delimiter=',').astype(np.int32)

#         try:
#             node_labels = np.loadtxt(file_node_labels, delimiter=',').astype(np.int32)
#         except Exception as e:
#             print(e)
#             print('use node label 0')
#             node_labels = np.zeros(graph_indicator.shape[0]).astype(np.int32)

#         graph_id = 1
#         starts = [1]
#         node2graph = {}
#         for i in range(len(graph_indicator)):
#             if graph_indicator[i] != graph_id:
#                 graph_id = graph_indicator[i]
#                 starts.append(i+1)
#             node2graph[i+1] = len(starts)-1
#         # print(starts)
#         # print(node2graph)
#         graphid = 0
#         edge_lists = []
#         edge_label_lists = []
#         edge_list = []
#         edge_label_list = []
#         for (s, t), l in list(zip(edges, edge_labels)):
#             sgid = node2graph[s]
#             tgid = node2graph[t]
#             if sgid != tgid:
#                 print('edges connecting different graphs, error here, please check.')
#                 print(s, t, 'graph id', sgid, tgid)
#                 exit(1)
#             gid = sgid
#             if gid != graphid:
#                 edge_lists.append(edge_list)
#                 edge_label_lists.append(edge_label_list)
#                 edge_list = []
#                 edge_label_list = []
#                 graphid = gid
#             start = starts[gid]
#             edge_list.append((s-start, t-start))
#             edge_label_list.append(l)

#         edge_lists.append(edge_list)
#         edge_label_lists.append(edge_label_list)

#         # node labels
#         node_label_lists = []
#         graphid = 0
#         node_label_list = []
#         for i in range(len(node_labels)):
#             nid = i+1
#             gid = node2graph[nid]
#             # start = starts[gid]
#             if gid != graphid:
#                 node_label_lists.append(node_label_list)
#                 graphid = gid
#                 node_label_list = []
#             node_label_list.append(node_labels[i])
#         node_label_lists.append(node_label_list)

#         return edge_lists, graph_labels, edge_label_lists, node_label_lists
    

# https://github.com/flyingdoog/PGExplainer/blob/master/MUTAG.ipynb

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

#             if i<5:
#                 print("\U0001F96D node_label.shape:", node_label.shape)
#                 print("\U0001F96D I:", i)
#                 try:
#                     edge_att = torch.ones(edge_index.shape[1])
#                     fig, ax = plt.subplots()
#                     fig.canvas.manager.set_window_title(f"Dual Graph {i}")
#                     visualize_a_graph(edge_index, edge_att, node_label, 'mutag', ax, coor=None, norm=False, mol_type=None, nodesize=300)
                    
#                     plt.show()
#                     plt.close(fig)
#                 except Exception as e:
#                     print(f"Error in graph {i}: {e}")

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


# https://github.com/flyingdoog/PGExplainer/blob/master/MUTAG.ipynb

import yaml
import torch
import numpy as np
import pickle as pkl
from pathlib import Path
from torch_geometric.data import InMemoryDataset, Data


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
        return ['data.pt'] #MANGO

    def download(self):
        raise NotImplementedError

    def process(self):
        with open(self.raw_dir + '/Mutagenicity.pkl', 'rb') as fin:
            _, original_features, original_labels = pkl.load(fin)

        dual_edge_lists, graph_labels, dual_node_gt_lists, dual_node_label_lists, dual_node_lists = self.get_graph_data()

        data_list = []

        #print("\U0001F96D original_features.shape:", original_features.shape) #MANGO (4337, 418, 14)
        #print("\U0001F96D dual_edge_lists:", dual_edge_lists) #MANGO, len: 8674, 
        #print("\U0001F96D original_features.shape:", original_features.shape)

        max_len = max(len(sublist) for sublist in dual_edge_lists)
        #print("\U0001F96D max_len:", max_len) #MANGO 

        dual_features = np.zeros((4337, 203, 28))

        for gid in range(dual_features.shape[0]):
            dual_node_list = dual_node_lists[gid]
            idx = 0
            for (s,t) in dual_node_list:
                if len(dual_node_list) == 0:
                    print(f"\U0001F96D Skipping gid={gid} due to empty dual_node_list")
                    continue
                #print("\U0001F96D s:",s)
                #print("\U0001F96D type(s):",type(s))
                assert np.isscalar(s), f"s is not scalar: {s} (type={type(s)})"
                assert np.isscalar(t), f"t is not scalar: {t} (type={type(t)})"
                f1 = original_features[gid, s, :]
                f2 = original_features[gid, t, :]
                cat = np.concatenate((f1, f2))
                assert cat.shape == (28,), f"Concatenated shape is wrong: {cat.shape}"
                dual_features[gid,idx,:] = cat

        count = 0
        for i in range(dual_features.shape[0]):
            num_nodes = len(dual_node_lists[i]) # MANGO
            
            #print("\U0001F96D i:", i)
            #print("\U0001F96D num_nodes:", num_nodes)

            #edge_index = torch.tensor(dual_edge_lists[i], dtype=torch.long).T # MANGO, didnt work bc 2, 2, x dimension


            edges_np = np.array(dual_edge_lists[i]) 
            edge_index = torch.tensor(edges_np, dtype=torch.long).T
            edge_index = edge_index.reshape(2, -1)


            #print("\U0001F96D edge_index.shape:", edge_index.shape)
            #print("\U0001F96D edge_index:", edge_index)

            dual_nodes_index = torch.tensor(dual_node_lists[i], dtype=torch.long).T

            y = torch.tensor(graph_labels[i]).float().reshape(-1, 1)
            x = torch.tensor(dual_features[i][:num_nodes]).float() # MANGO removes padding, MANGO PIT
            #print("\U0001F96D padding:", original_features[i][num_nodes:])
            #print("\U0001F96D padding sum:", original_features[i][num_nodes:].sum())
            assert dual_features[i][num_nodes:].sum() == 0
            #if (original_features[i][num_nodes:].sum() != 0):
            #    count+=1

            # MANGO
            dual_node_gt = torch.tensor(dual_node_gt_lists[i]).float() # MANGO 
            #if y.item() != 0: # MANGO
            #    edge_label = torch.zeros_like(edge_label).float() # MANGO zero out edge labels, RAW MANGO

            node_label = torch.zeros(x.shape[0]) # MANGO copies shape of node features and sets to zero, RAW MANGO
            #signal_nodes = list(set(dual_nodes_index[:, dual_node_signal.bool()].reshape(-1).tolist())) # MANGO picks out signal nodes from the edge label (for explainer; ground truth nodes), ONLY THE RIPE MANGO
            #print("\U0001F96D dual_nodes_index:", dual_nodes_index)
            #print("\U0001F96D edge_index:", edge_index)
            
            #signal_nodes = list(set(dual_nodes_index[dual_node_gt.astype(bool)].tolist()))
            signal_nodes = list(set([x - 1 for x in dual_nodes_index[:, dual_node_gt.bool()].reshape(-1).tolist()]))
            #signal_nodes = list(dual_nodes[dual_node_signal])

            #print("\U0001F96D dual_gt_list:", dual_node_gt_lists) #MANGO

            if y.item() == 0:
                #print("\U0001F96D signal_nodes:", signal_nodes)
                node_label[signal_nodes] = 1 #set signal nodes' label to one if graph label is 0, ONLY THE RIPE MANGO
                #print("\U0001F96D signal_nodes:", signal_nodes)

            # if len(signal_nodes) != 0: #MANGO IDK HERE DOES IT MAKE SENSE TO TAKE THIS OUT cause {4,1} {4,3} etc doesn't apply?*, ROTTEN MANGO?
            #     node_type = torch.tensor(dual_node_label_lists[i]) # MANGO
            #     node_type = set(node_type[signal_nodes].tolist())
            #     assert node_type in ({4, 1}, {4, 3}, {4, 1, 3})  # MANGO NO or NH; Checks that signal nodes are the right node type

            # if y.item() == 0 and len(signal_nodes) == 0: #SKIP THIS 
            #     continue

            edge_label = torch.zeros(edge_index.size(1), dtype=torch.float) # MANGO 
            if x.shape[0] == 0 or edge_index.shape[1] == 0:
                print(f"\U0001F96D Skipping empty graph {i}")
                continue
                
            node_type = torch.tensor(dual_features[i][:num_nodes]).float()
            #print(f" Total graphs processed: {len(data_list)}")
            print("\U0001F96D x.shape: ", x.shape) #MANGO
            print("\U0001F96D y.shape: ", y.shape) #MANGO
            print("\U0001F96D edge_index.shape:", edge_index.shape) #MANGO
            print("\U0001F96D node_label.shape:", node_label.shape) #MANGO
            print("\U0001F96D edge_label.shape:", edge_label.shape) #MANGO
            print("\U0001F96D node_type.shape:", node_type.shape) #MANGO

            if i<5:
                print("\U0001F96D node_label.shape:", node_label.shape)
                print("\U0001F96D I:", i)
                try:
                    edge_index = torch.tensor([[0, 1, 0, 2, 1, 3],
                           [1, 0, 2, 0, 3, 1]], dtype=torch.long)
                    edge_att = torch.ones(edge_index.shape[1])
                    node_label = torch.zeros(4)
                    fig, ax = plt.subplots()
                    fig.canvas.manager.set_window_title(f"Primal Graph {i}")
                    print("\U0001F96D edge_index:", edge_index)
                    print("\U0001F96D edge_att:", edge_att)
                    print("\U0001F96D node_label:", node_label)
                    
                    visualize_a_graph(edge_index, edge_att, node_label, 'mutag', ax, coor=None, norm=False, mol_type=None, nodesize=300)
                    
                    plt.show()
                    plt.close(fig)
                except Exception as e:
                    print(f"Error in graph {i}: {e}")

            data_list.append(Data(x=x, y=y, edge_index=edge_index, node_label=node_label, edge_label=edge_label, node_type = node_type)) # MANGO , nodetype?

        print(f" Total graphs processed final: {len(data_list)}")
        print("\U0001F96D count:", count)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_graph_data(self):
        pri = self.raw_dir + '/Mutagenicity_'

        file_edges = pri + 'A.txt' 
        file_edge_labels = pri + 'edge_labels.txt'
        file_edge_gt = pri + 'edge_gt.txt'
        file_graph_indicator = pri + 'graph_indicator.txt'
        file_graph_labels = pri + 'graph_labels.txt'
        file_node_labels = pri + 'node_labels.txt'

        dual_nodes = np.loadtxt(file_edges, delimiter=',').astype(np.int32) # MANGO Loading dual_nodes in adjacency matrix form

        print("\U0001F96D dual_nodes.shape:", dual_nodes.shape) #MANGO,
        print("\U0001F96D dual_nodes:", dual_nodes) #MANGO,

        # dual_nodes = np.array([ #DOUBLE MANGO
        #     [1, 2],
        #     [2, 1],
        #     [1, 3],
        #     [3, 1]
        # ])

        unique_elements = np.unique(dual_nodes)

        # Map each to an index
        dual_dict = {element: idx for idx, element in enumerate(unique_elements)}

        try:
            dual_node_labels = np.loadtxt(file_edge_labels, delimiter=',').astype(np.int32)
            #dual_node_labels = np.zeros(dual_nodes.shape[0]).astype(np.int32) # DOUBLE MANGO
        except Exception as e:
            print(e)
            print('use edge label 0')
            dual_node_labels = np.zeros(dual_nodes.shape[0]).astype(np.int32) # MANGO

        try: # MANGO
            dual_node_signal = np.loadtxt(file_edge_gt, delimiter=',').astype(np.int32) #MANGO
            #dual_node_signal = np.zeros(dual_nodes.shape[0]).astype(np.int32) # DOUBLE MANGO
        except Exception as e: # MANGO
            print(e) # MANGO
            print('use edge gt 0') # MANGO
            dual_node_signal = np.zeros(dual_nodes.shape[0]).astype(np.int32) # MANGO


        graph_indicator = np.loadtxt(file_graph_indicator, delimiter=',').astype(np.int32)

        print("\U0001F96D graph_indicator.shape:", graph_indicator.shape) #MANGO,
        print("\U0001F96D graph_indicator:", graph_indicator) #MANGO,
        #graph_indicator = np.
        graph_labels = np.loadtxt(file_graph_labels, delimiter=',').astype(np.int32)

        

        dual_graph_indicator = np.zeros((dual_nodes.shape[0], 1)) #MANGO, 
        for i in (range(dual_nodes.shape[0])): #MANGO
            assert graph_indicator[(dual_nodes[i][0]-1)] == graph_indicator[(dual_nodes[i][1]-1)]
            dual_graph_indicator[i] = graph_indicator[(dual_nodes[i][0])-1] #MANGO
        dual_graph_indicator = dual_graph_indicator.astype(int) # MANGO

        try:
            primal_node_labels = np.loadtxt(file_node_labels, delimiter=',').astype(np.int32) # MANGO should these be included in the edge features? MULTIPLE MANGO
        except Exception as e:
            print(e)
            print('use node label 0')
            primal_node_labels = np.zeros(graph_indicator.shape[0]).astype(np.int32) # MANGO

        # ATP loaded all files whole MANGO has been eaten

        dual_edges = [] # MANGO

        #CHATGPT ALERT ALERT ALERT MANGOS FALLING
        from collections import defaultdict

        dual_nodes = np.array(dual_nodes)  # Just in case

        group_by_first = defaultdict(list)
        group_by_second = defaultdict(list)

        for idx, (a, b) in enumerate(dual_nodes):
            group_by_first[a].append(idx)
            group_by_second[b].append(idx)

        dual_edges = []

        def add_pairs_from_group(group):
            indices = group
            length = len(indices)
            for i in range(length):
                for j in range(i + 1, length):
                    # e1 = dual_nodes[indices[i]]
                    # e2 = dual_nodes[indices[j]]
                    e1 = indices[i]
                    e2 = indices[j]
                    #print("\U0001F96D ENDDDDDD1:", e1[0], e2[0], e1[1], e2[1])
                    dual_edges.append([e1.tolist(), e2.tolist()])

        # Add pairs grouped by first element
        for group in group_by_first.values():
            if len(group) > 1:
                add_pairs_from_group(group)

        # Add pairs grouped by second element
        for group in group_by_second.values():
            if len(group) > 1:
                add_pairs_from_group(group)

        # CHATGPT END THE MANGOS HAVE LANDED

        dual_edges = np.array(dual_edges) # MANGO

        #print("\U0001F96D dual_edges:", dual_edges) #MANGO, looks good ATP
        #print("\U0001F96D len(dual_edges):", len(dual_edges)) #MANGO, looks good ATP, 451808

        graph_id = 1
        starts = [1]
        node2graph = {}
        #print("\U0001F96D dual_node_signal:", dual_graph_indicator) #MANGO
        for i in range(len(dual_graph_indicator)): #MANGO
            if dual_graph_indicator[i] != graph_id: #MANGO 
                graph_id = dual_graph_indicator[i] #MANGO
                starts.append(i) #MANGO
            #node2graph[tuple(dual_nodes[i])] = len(starts)-1 #MANGO

        #Need separate primal starts because normal starts doesn't line up right because of the edges not being numbered 1,2,3 ...
        graph_id = 1
        primal_starts = [1]
        for i in range(len(graph_indicator)): #MANGO
            if graph_indicator[i] != graph_id: #MANGO 
                graph_id = graph_indicator[i] #MANGO
                primal_starts.append(i) #MANGO
            node2graph[i+1] = len(primal_starts)-1 #MANGO, no minus one needed because node2graph only used for comparing nodes against each other
            #print("\U0001F96D len(primal_starts):", len(primal_starts)) #MANGO, looks good ATP, 451808
        #Looks all good atp

        #print("\U0001F96D dual graph indicator:", dual_graph_indicator) #MANGO

        # print(starts)
        # print(node2graph)
        graphid = 0
        dual_edge_lists = [] # MANGO
        #edge_label_lists = [] # MANGO
        dual_edge_list = [] # MANGO
        #edge_label_list = [] # MANGO
        for (s, t) in list(dual_edges): #MANGO removed edge labels because dual edges have no gt, SEEDLESS MANGO 
            # sgid = node2graph[s[0]] # MANGO, graph id of start node
            # tgid = node2graph[t[0]] # MANGO, graph id of end node
            sgid = node2graph[s] # MANGO, graph id of start node
            tgid = node2graph[t] # MANGO, graph id of end node
            if sgid != tgid:
                print('edges connecting different graphs, error here, please check.')
                print(s, t, 'graph id', sgid, tgid)
                exit(1)
            gid = sgid
            if gid != graphid: #create new sublist for next graph
                #print("\U0001F96D dual_edge_list:", dual_edge_list) #MANGO
                dual_edge_lists.append(dual_edge_list) # MANGO
                #edge_label_lists.append(edge_label_list) # MANGO
                dual_edge_list = []
                #edge_label_list = [] # MANGO
                graphid = gid
            start = primal_starts[gid] # MANGO 
            #print("\U0001F96D start:", start) #MANGO
            #print("\U0001F96D s:", s) #MANGO
            #print("\U0001F96D t:", t) #MANGO
            dual_edge_list.append((s-start, t-start))
            #edge_label_list.append(l) # MANGO

        dual_edge_lists.append(dual_edge_list)

        print("\U0001F96D KACHA MANGO dual_edge_lists:", dual_edge_lists) #MANGO
        print("\U0001F96D KACHA MANGO dual_edge_lists.shape:", dual_edge_lists.shape) #MANGO
        #edge_label_lists.append(edge_label_list) # MANGO

        #CHATGPT ALERT ALERT ALERT MANGOS FALLING

        dual_node_gt_lists = []
        dual_node_gt_list = []
        graphid = 0

        #print("\U0001F96D dual_node_signal:", dual_node_signal) #MANGO


        for idx, label in enumerate(dual_node_signal):
            gid = node2graph[dual_nodes[idx][0]]  # graph ID for this node
    
            if gid != graphid:
                dual_node_gt_lists.append(dual_node_gt_list)
                dual_node_gt_list = []
                graphid = gid

            dual_node_gt_list.append(label)

        dual_node_gt_lists.append(dual_node_gt_list)

        #CHATGPT END THE MANGOS HAVE LANDED 

        
        graphid = 0
        

        # for i in range(len(dual_nodes)):
        #     gid = node2graph[tuple(dual_nodes[i])]
    
        #     if gid != graphid:
        #         dual_nodes_lists.append(dual_node_list)
        #         dual_node_list = []
        #         graphid = gid

        #     start = starts[gid]
        #     dual_node_list.append(tuple(dual_nodes[i] - start))  # or just dual_nodes[i] if raw
    
        # dual_nodes_lists.append(dual_node_list)

        # node labels


        # print("\U0001F96D dual_nodes size:", len(dual_nodes)) #MANGO
        dual_node_lists = []
        dual_node_label_lists = []
        graphid = 0
        dual_node_label_list = []
        dual_node_list = []
        for i in range(len(dual_node_labels)):
            # nid = i+1 # MANGO
            gid = node2graph[dual_nodes[i][0]] # MANGO
            # start = starts[gid]
            if gid != graphid:
                dual_node_label_lists.append(dual_node_label_list)
                dual_node_lists.append(dual_node_list)
                graphid = gid
                dual_node_label_list = []
                dual_node_list = []
            u, v = dual_nodes[i]
            gu = graph_indicator[u-1]
            gv = graph_indicator[v-1]
            assert gu == gv

            start = primal_starts[gid]
            #print("\U0001F96D start:", start) #MANGO
            #print("\U0001F96D dual_nodes[i]:", dual_nodes[i])
            dual_node_label_list.append(dual_node_labels[i])
            dual_node_list.append((dual_nodes[i][0]-start, dual_nodes[i][1]-start))
        dual_node_label_lists.append(dual_node_label_list)
        dual_node_lists.append(dual_node_list)

        # print("\U0001F96D dual_node_label_lists size:", len(dual_node_label_lists)) #MANGO
        # print("\U0001F96D dual_node_lists size:", len(dual_node_lists)) #MANGO

        return dual_edge_lists, graph_labels, dual_node_gt_lists, dual_node_label_lists, dual_node_lists # MANGO
    
#============================
    
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




# https://github.com/flyingdoog/PGExplainer/blob/master/MUTAG.ipynb

import yaml
import torch
import numpy as np
import pickle as pkl
from pathlib import Path
from torch_geometric.data import InMemoryDataset, Data


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
        return ['dual_data.pt']

    def download(self):
        raise NotImplementedError

    def process(self):
        with open(self.raw_dir + '/Mutagenicity.pkl', 'rb') as fin:
            _, original_features, original_labels = pkl.load(fin)
        #print("\U0001F96D original_features shape:", original_features.shape) #MANGO
        #print("\U0001F96D original_features[417]:", original_features[417]) #MANGO
        #print("\U0001F96D original_labels:", original_labels) #MANGO
        edge_lists, graph_labels, edge_label_lists, node_type_lists = self.get_graph_data()

        #print("\U0001F96D original_features shape:", original_features.shape) #MANGO

        node_label_lists = np.array(node_type_lists) #MANGO
        #print("\U0001F96D node_label_lists shape:", len(node_label_lists)) #MANGO

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

            data_list.append(Data(x=x, y=y, edge_index=edge_index, node_label=node_label, edge_label=edge_label, node_type=torch.tensor(node_type_lists[i])))

        data, slices = self.collate(data_list)
        #torch.save((data, slices), self.processed_paths[0])

    def get_graph_data(self):
        pri = self.raw_dir + '/Mutagenicity_'

        file_edges = pri + 'A.txt'
        file_edge_labels_true = pri + 'edge_labels.txt' #MANGO
        file_edge_labels = pri + 'edge_gt.txt' #checking explanation accuracy
        file_graph_indicator = pri + 'graph_indicator.txt'
        file_graph_labels = pri + 'graph_labels.txt'
        file_node_labels = pri + 'node_labels.txt'

        dual_nodes = np.loadtxt(file_edges, delimiter=',').astype(np.int32) #MANGO
        #print("\U0001F96D dual nodes: ", dual_nodes) #MANGO
        dual_node_labels = np.loadtxt(file_edge_labels_true, delimiter=",").astype(np.int32) #MANGO

        try:
            edge_labels = np.loadtxt(file_edge_labels, delimiter=',').astype(np.int32)
        except Exception as e:
            print(e)
            print('use edge label 0')
            edge_labels = np.zeros(dual_nodes.shape[0]).astype(np.int32)

        graph_indicator = np.loadtxt(file_graph_indicator, delimiter=',').astype(np.int32)

        dual_graph_labels = np.loadtxt(file_graph_labels, delimiter=',').astype(np.int32) #MANGO
        #print("\U0001F96D graph_labels.shape: ", dual_graph_labels.shape) #MANGO
        

        try:
            node_labels = np.loadtxt(file_node_labels, delimiter=',').astype(np.int32)
        except Exception as e:
            print(e)
            print('use node label 0')
            node_labels = np.zeros(graph_indicator.shape[0]).astype(np.int32)

        #print("\U0001F96D graph_indicator.shape: ", graph_indicator.shape) #MANGO
        
        unique, counts = np.unique(graph_indicator, return_counts=True) #MANGO
        max_count = np.max(counts) #MANGO
        most_common_value = unique[np.argmax(counts)] #MANGO

        #print("\U0001F96D MAX, MOST COMMON: ", max_count, most_common_value) #MANGO

        #print("\U0001F96D node_labels.shape: ", node_labels.shape) #MANGO
        #print("\U0001F96D dual_nodes.shape: ", dual_nodes.shape) #MANGO
        #print("\U0001F96D graph_indicator: ", graph_indicator) #MANGO
        
        dual_graph_indicator = np.zeros((dual_nodes.shape[0], 1)) #MANGO
        for i in (range(dual_nodes.shape[0])): #MANGO
            dual_graph_indicator[i] = graph_indicator[(dual_nodes[i][0]-1)] #MANGO
        dual_graph_indicator = dual_graph_indicator.astype(int)

        #print("\U0001F96D dual_graph_indicator: ", dual_graph_indicator) #MANGO

       
        dual_edges = []

        # n = len(dual_nodes)
        # for i in range(n):
        #     for j in range(i + 1, n):
        #         e1, e2 = dual_nodes[i], dual_nodes[j]
        #         if e1[0] == e2[0] or e1[1] == e2[1]:
        #             print("\U0001F96D ENDDDDDD1:", e1[0], " ", e2[0], " ", e1[1], " ", e2[1]) #MANGO
        #             dual_edges.append([e1.tolist(), e2.tolist()])

        #CHAT GPT ALERT ALERT ALERT MANGOS
        from collections import defaultdict

        dual_nodes = np.array(dual_nodes)  # Just in case

        group_by_first = defaultdict(list)
        group_by_second = defaultdict(list)

        for idx, (a, b) in enumerate(dual_nodes):
            group_by_first[a].append(idx)
            group_by_second[b].append(idx)

        dual_edges = []

        def add_pairs_from_group(group):
            indices = group
            length = len(indices)
            for i in range(length):
                for j in range(i + 1, length):
                    e1 = dual_nodes[indices[i]]
                    e2 = dual_nodes[indices[j]]
                    #print("\U0001F96D ENDDDDDD1:", e1[0], e2[0], e1[1], e2[1])
                    dual_edges.append([e1.tolist(), e2.tolist()])

        # Add pairs grouped by first element
        for group in group_by_first.values():
            if len(group) > 1:
                add_pairs_from_group(group)

        # Add pairs grouped by second element
        for group in group_by_second.values():
            if len(group) > 1:
                add_pairs_from_group(group)

        # CHATGPT END MANGOS

        dual_edges = np.array(dual_edges)

        graph_id = 1
        starts = [1]
        node2graph = {}

        for i in range(len(dual_graph_indicator)): #MANGO
            if dual_graph_indicator[i] != graph_id: #MANGO 
                graph_id = dual_graph_indicator[i] #MANGO
                starts.append(i) #MANGO
            node2graph[tuple(dual_nodes[i])] = len(starts)-1 #MANGO

        # print(starts)
        # print(node2graph)

        graphid = 0
        edge_lists = []
        edge_label_lists = []
        edge_list = []
        edge_label_list = []

        for (s, t), l in list(zip(dual_edges, edge_labels)): #MANGO , leaving edge_labels bc its gt anyway and all zero
            sgid = node2graph[tuple(s)] #MANGO 
            tgid = node2graph[tuple(t)] #MANGO
            if sgid != tgid: #MANGO
                print('edges connecting different graphs, error here, please check.') #MANGO
                print(s, t, 'graph id', sgid, tgid) #MANGO
                exit(1) #MANGO
            gid = sgid #MANGO
            if gid != graphid: #MANGO
                edge_lists.append(edge_list) #MANGO
                edge_label_lists.append(edge_label_list) #MANGO
                edge_list = [] #MANGO
                edge_label_list = []
                graphid = gid
            start = starts[gid]
            edge_list.append(((s[0]-start,s[1]-start), (t[0]-start, t[1]-start)))
            edge_label_list.append(l)

        edge_lists.append(edge_list)
        edge_label_lists.append(edge_label_list)

        # node labels
        node_label_lists = []
        graphid = 0
        node_label_list = []
        for i in range(len(dual_node_labels)):
            #nid = i+1
            gid = node2graph[tuple(dual_nodes[i])]
            # start = starts[gid]
            if gid != graphid:
                node_label_lists.append(node_label_list)
                graphid = gid
                node_label_list = []
            node_label_list.append(dual_node_labels[i])
        node_label_lists.append(node_label_list)

        #print("\U0001F96D ENDDDDDD2:") #MANGO

        return edge_lists, dual_graph_labels, edge_label_lists, node_label_lists

