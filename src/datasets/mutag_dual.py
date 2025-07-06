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

